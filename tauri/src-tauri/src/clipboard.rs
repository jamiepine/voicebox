//! Snapshot / write / restore helpers around the system clipboard.
//!
//! Used by the auto-paste flow: before synthesising the paste accelerator
//! into a foreign app we need to (1) remember what the user had on the
//! clipboard, (2) stage our transcribed text, (3) paste, (4) put the
//! original contents back. Missing step 4 turns every dictation into a
//! silent clipboard-stomp.
//!
//! On **macOS** the snapshot walks `NSPasteboard.pasteboardItems` and
//! copies every `(UTI, data)` pair into an owned `Vec<u8>`, so restore
//! rebuilds the full multi-type payload — not just the plain-text
//! fallback. Images, styled text, file-reference lists all survive the
//! round-trip.
//!
//! On **Windows** the snapshot walks `EnumClipboardFormats` and copies the
//! HGLOBAL payload for every advertised format. GDI-handle formats (DIB
//! bitmap, metafile, enhanced metafile, palette), owner-display variants,
//! and the private-/GDI-object format ranges are skipped — those can't be
//! round-tripped across processes without synthesising the underlying
//! kernel/GDI objects, which isn't worth the complexity for a dictation
//! clipboard guard. CF_UNICODETEXT, CF_HDROP, CF_DIB (bitmap data in
//! memory, not a handle), CF_DIBV5, and every registered format (HTML
//! Format, Rich Text Format, FileGroupDescriptor, etc.) all survive.
//!
//! On **macOS** every entry point manages its own `NSAutoreleasePool`
//! because the Tauri command runtime threads don't have one by default —
//! without it, every autoreleased `NSString` / `NSData` we touch would
//! leak for the life of the process. On Windows, HGLOBAL ownership
//! transfers to the clipboard on `SetClipboardData` success, so we only
//! free handles we allocated but didn't hand off.

#[cfg(target_os = "macos")]
use objc::runtime::Object;
#[cfg(target_os = "macos")]
use objc::{class, msg_send, sel, sel_impl};

/// One full-fidelity snapshot of the general pasteboard. Hold on to the value
/// until the paste has landed, then pass it to [`restore_clipboard`].
#[derive(Debug, Clone)]
pub struct ClipboardSnapshot {
    /// Outer vec: pasteboard items. Inner: `(uti, raw bytes)` per type. We
    /// store the raw UTI string and the raw `NSData` payload so we can rebuild
    /// the item with `setData:forType:` without interpreting the contents.
    items: Vec<Vec<(String, Vec<u8>)>>,
    /// `NSPasteboard.changeCount` at the moment of capture. Incremented by AppKit
    /// on every mutation from any process, so a caller can decide whether a
    /// restore is still safe (change_count == expected) or whether someone
    /// else wrote to the clipboard in the interim and we should back off.
    change_count: i64,
}

impl ClipboardSnapshot {
    pub fn change_count(&self) -> i64 {
        self.change_count
    }

    pub fn item_count(&self) -> usize {
        self.items.len()
    }
}

#[cfg(target_os = "macos")]
type Id = *mut Object;

/// RAII wrapper so the pool drains even on early return / `?` propagation.
#[cfg(target_os = "macos")]
struct AutoreleasePool {
    pool: Id,
}

#[cfg(target_os = "macos")]
impl AutoreleasePool {
    unsafe fn new() -> Self {
        let pool: Id = msg_send![class!(NSAutoreleasePool), alloc];
        let pool: Id = msg_send![pool, init];
        Self { pool }
    }
}

#[cfg(target_os = "macos")]
impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.pool, drain];
        }
    }
}

/// Build an autoreleased `NSString` from a Rust `&str` without scanning for
/// interior nulls (which is what `initWithUTF8String:` would require).
#[cfg(target_os = "macos")]
unsafe fn ns_string(s: &str) -> Id {
    // NSUTF8StringEncoding = 4.
    let obj: Id = msg_send![class!(NSString), alloc];
    let obj: Id = msg_send![
        obj,
        initWithBytes: s.as_ptr()
        length: s.len()
        encoding: 4u64
    ];
    let _: () = msg_send![obj, autorelease];
    obj
}

#[cfg(target_os = "macos")]
unsafe fn ns_string_to_rust(s: Id) -> Option<String> {
    if s.is_null() {
        return None;
    }
    let bytes: *const i8 = msg_send![s, UTF8String];
    if bytes.is_null() {
        return None;
    }
    std::ffi::CStr::from_ptr(bytes)
        .to_str()
        .ok()
        .map(|x| x.to_owned())
}

#[cfg(target_os = "macos")]
unsafe fn general_pasteboard() -> Result<Id, String> {
    let pb: Id = msg_send![class!(NSPasteboard), generalPasteboard];
    if pb.is_null() {
        return Err("NSPasteboard generalPasteboard returned nil".into());
    }
    Ok(pb)
}

/// Read the pasteboard's current change count without snapshotting contents.
///
/// AppKit increments this every time any process writes to the general
/// pasteboard, so it's a cheap way to detect "did someone clobber my staged
/// text before the paste landed?".
#[cfg(target_os = "macos")]
pub fn current_change_count() -> Result<i64, String> {
    unsafe {
        let _pool = AutoreleasePool::new();
        let pb = general_pasteboard()?;
        let c: i64 = msg_send![pb, changeCount];
        Ok(c)
    }
}

/// Capture every item on the general pasteboard into an owned snapshot.
#[cfg(target_os = "macos")]
pub fn save_clipboard() -> Result<ClipboardSnapshot, String> {
    unsafe {
        let _pool = AutoreleasePool::new();
        let pb = general_pasteboard()?;
        let change_count: i64 = msg_send![pb, changeCount];

        let items: Id = msg_send![pb, pasteboardItems];
        if items.is_null() {
            return Ok(ClipboardSnapshot {
                items: Vec::new(),
                change_count,
            });
        }

        let count: usize = msg_send![items, count];
        let mut saved: Vec<Vec<(String, Vec<u8>)>> = Vec::with_capacity(count);

        for i in 0..count {
            let item: Id = msg_send![items, objectAtIndex: i];
            if item.is_null() {
                continue;
            }
            let types: Id = msg_send![item, types];
            if types.is_null() {
                continue;
            }
            let type_count: usize = msg_send![types, count];
            let mut pairs: Vec<(String, Vec<u8>)> = Vec::with_capacity(type_count);
            for j in 0..type_count {
                let t: Id = msg_send![types, objectAtIndex: j];
                let Some(type_str) = ns_string_to_rust(t) else {
                    continue;
                };
                let data: Id = msg_send![item, dataForType: t];
                if data.is_null() {
                    // Type advertised but no concrete data (lazy provider).
                    // Skipping is safer than trying to force it to materialise.
                    continue;
                }
                let length: usize = msg_send![data, length];
                let bytes_ptr: *const u8 = msg_send![data, bytes];
                let bytes = if bytes_ptr.is_null() || length == 0 {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(bytes_ptr, length).to_vec()
                };
                pairs.push((type_str, bytes));
            }
            saved.push(pairs);
        }

        Ok(ClipboardSnapshot {
            items: saved,
            change_count,
        })
    }
}

/// Replace the pasteboard contents with a single plain-text string. Returns
/// the post-write change count so a later restore can verify nothing else
/// touched the clipboard in between.
#[cfg(target_os = "macos")]
pub fn write_text(text: &str) -> Result<i64, String> {
    unsafe {
        let _pool = AutoreleasePool::new();
        let pb = general_pasteboard()?;
        let _new_count: i64 = msg_send![pb, clearContents];

        let ns_text = ns_string(text);
        // `public.utf8-plain-text` is the raw UTI behind `NSPasteboardTypeString`
        // and works for every text-aware paste target we care about.
        let ns_type = ns_string("public.utf8-plain-text");
        let ok: bool = msg_send![pb, setString: ns_text forType: ns_type];
        if !ok {
            return Err("NSPasteboard setString:forType: returned NO".into());
        }

        let after: i64 = msg_send![pb, changeCount];
        Ok(after)
    }
}

/// Rebuild the pasteboard from a snapshot, replacing whatever is on it now.
///
/// Does not consult the change count — callers that want safe restore should
/// compare [`current_change_count`] against the value returned by
/// [`write_text`] first.
#[cfg(target_os = "macos")]
pub fn restore_clipboard(snapshot: &ClipboardSnapshot) -> Result<(), String> {
    unsafe {
        let _pool = AutoreleasePool::new();
        let pb = general_pasteboard()?;
        let _: i64 = msg_send![pb, clearContents];

        if snapshot.items.is_empty() {
            return Ok(());
        }

        let array: Id = msg_send![class!(NSMutableArray), array];

        for pairs in &snapshot.items {
            let item: Id = msg_send![class!(NSPasteboardItem), alloc];
            let item: Id = msg_send![item, init];
            let _: () = msg_send![item, autorelease];

            for (uti, bytes) in pairs {
                let ns_type = ns_string(uti);
                let data: Id = msg_send![
                    class!(NSData),
                    dataWithBytes: bytes.as_ptr()
                    length: bytes.len()
                ];
                let _ok: bool = msg_send![item, setData: data forType: ns_type];
            }

            let _: () = msg_send![array, addObject: item];
        }

        let ok: bool = msg_send![pb, writeObjects: array];
        if !ok {
            return Err("NSPasteboard writeObjects: returned NO".into());
        }
        Ok(())
    }
}

#[cfg(target_os = "windows")]
mod win {
    //! Windows clipboard implementation.
    //!
    //! The snapshot is structured so it mirrors the macOS `Vec<Vec<_>>`
    //! shape: a single outer "item" holding one `(format-name, bytes)`
    //! pair per enumerated format. Windows has no notion of multiple
    //! pasteboard items, so there's always exactly one or zero outer
    //! entries — enough to keep `item_count()` meaningful without
    //! fan-out.
    //!
    //! Format IDs are serialised as strings so the snapshot type can stay
    //! platform-neutral. Predefined formats use their canonical
    //! identifier (`"CF_UNICODETEXT"`, `"CF_HDROP"`, `"CF_DIB"`, …);
    //! registered formats use their string name from
    //! `GetClipboardFormatNameW` (`"HTML Format"`, `"Rich Text
    //! Format"`, …). Restore reverses the mapping with a lookup table
    //! for the predefined IDs and `RegisterClipboardFormatW` for the
    //! rest.
    //!
    //! Skipped format classes:
    //! - CF_BITMAP (2), CF_METAFILEPICT (3), CF_PALETTE (9),
    //!   CF_ENHMETAFILE (14) — HGLOBAL's actually an HBITMAP /
    //!   HENHMETAFILE, not raw memory. Rebuilding them across processes
    //!   is possible but not worth it for clipboard stashing.
    //! - CF_OWNERDISPLAY (0x80) and the CF_DSPxxx variants (0x81–0x8E) —
    //!   the owner draws these on demand. No data to snapshot.
    //! - CF_PRIVATEFIRST..CF_PRIVATELAST (0x200–0x2FF) — app-private,
    //!   meaningless to restore from a different process.
    //! - CF_GDIOBJFIRST..CF_GDIOBJLAST (0x300–0x3FF) — GDI handles.
    //!
    //! Text formats that Windows auto-synthesises (CF_TEXT, CF_OEMTEXT,
    //! CF_LOCALE) are also skipped during save: `SetClipboardData` on
    //! CF_UNICODETEXT regenerates them lazily on restore.

    use std::thread;
    use std::time::Duration;

    use windows::core::PCWSTR;
    use windows::Win32::Foundation::{GlobalFree, HANDLE, HGLOBAL, HWND};
    use windows::Win32::System::DataExchange::{
        CloseClipboard, EmptyClipboard, EnumClipboardFormats, GetClipboardData,
        GetClipboardFormatNameW, GetClipboardSequenceNumber, OpenClipboard,
        RegisterClipboardFormatW, SetClipboardData,
    };
    use windows::Win32::System::Memory::{
        GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock, GLOBAL_ALLOC_FLAGS,
    };

    // `windows` 0.62 doesn't re-export every predefined clipboard format
    // under a stable feature flag, so the values are pinned inline.
    // These numbers are ABI-stable back to Windows 3.1 — verified against
    // winuser.h.
    pub const CF_TEXT: u32 = 1;
    pub const CF_BITMAP: u32 = 2;
    pub const CF_METAFILEPICT: u32 = 3;
    pub const CF_SYLK: u32 = 4;
    pub const CF_DIF: u32 = 5;
    pub const CF_TIFF: u32 = 6;
    pub const CF_OEMTEXT: u32 = 7;
    pub const CF_DIB: u32 = 8;
    pub const CF_PALETTE: u32 = 9;
    pub const CF_PENDATA: u32 = 10;
    pub const CF_RIFF: u32 = 11;
    pub const CF_WAVE: u32 = 12;
    pub const CF_UNICODETEXT: u32 = 13;
    pub const CF_ENHMETAFILE: u32 = 14;
    pub const CF_HDROP: u32 = 15;
    pub const CF_LOCALE: u32 = 16;
    pub const CF_DIBV5: u32 = 17;
    pub const CF_OWNERDISPLAY: u32 = 0x0080;
    pub const CF_DSPTEXT: u32 = 0x0081;
    pub const CF_DSPBITMAP: u32 = 0x0082;
    pub const CF_DSPMETAFILEPICT: u32 = 0x0083;
    pub const CF_DSPENHMETAFILE: u32 = 0x008E;
    pub const CF_PRIVATEFIRST: u32 = 0x0200;
    pub const CF_PRIVATELAST: u32 = 0x02FF;
    pub const CF_GDIOBJFIRST: u32 = 0x0300;
    pub const CF_GDIOBJLAST: u32 = 0x03FF;

    /// `GlobalAlloc` movable-memory flag — `GMEM_MOVEABLE` (0x0002).
    /// Required for HGLOBAL handles destined for `SetClipboardData`; fixed
    /// allocations are rejected.
    const GMEM_MOVEABLE: GLOBAL_ALLOC_FLAGS = GLOBAL_ALLOC_FLAGS(0x0002);

    /// Map a predefined clipboard format ID to its canonical identifier
    /// string. Registered formats (IDs >= 0xC000) aren't handled here —
    /// the caller resolves those via `GetClipboardFormatNameW`.
    pub fn predefined_name(id: u32) -> Option<&'static str> {
        Some(match id {
            CF_TEXT => "CF_TEXT",
            CF_BITMAP => "CF_BITMAP",
            CF_METAFILEPICT => "CF_METAFILEPICT",
            CF_SYLK => "CF_SYLK",
            CF_DIF => "CF_DIF",
            CF_TIFF => "CF_TIFF",
            CF_OEMTEXT => "CF_OEMTEXT",
            CF_DIB => "CF_DIB",
            CF_PALETTE => "CF_PALETTE",
            CF_PENDATA => "CF_PENDATA",
            CF_RIFF => "CF_RIFF",
            CF_WAVE => "CF_WAVE",
            CF_UNICODETEXT => "CF_UNICODETEXT",
            CF_ENHMETAFILE => "CF_ENHMETAFILE",
            CF_HDROP => "CF_HDROP",
            CF_LOCALE => "CF_LOCALE",
            CF_DIBV5 => "CF_DIBV5",
            CF_OWNERDISPLAY => "CF_OWNERDISPLAY",
            CF_DSPTEXT => "CF_DSPTEXT",
            CF_DSPBITMAP => "CF_DSPBITMAP",
            CF_DSPMETAFILEPICT => "CF_DSPMETAFILEPICT",
            CF_DSPENHMETAFILE => "CF_DSPENHMETAFILE",
            _ => return None,
        })
    }

    /// Reverse of [`predefined_name`].
    pub fn predefined_id(name: &str) -> Option<u32> {
        Some(match name {
            "CF_TEXT" => CF_TEXT,
            "CF_BITMAP" => CF_BITMAP,
            "CF_METAFILEPICT" => CF_METAFILEPICT,
            "CF_SYLK" => CF_SYLK,
            "CF_DIF" => CF_DIF,
            "CF_TIFF" => CF_TIFF,
            "CF_OEMTEXT" => CF_OEMTEXT,
            "CF_DIB" => CF_DIB,
            "CF_PALETTE" => CF_PALETTE,
            "CF_PENDATA" => CF_PENDATA,
            "CF_RIFF" => CF_RIFF,
            "CF_WAVE" => CF_WAVE,
            "CF_UNICODETEXT" => CF_UNICODETEXT,
            "CF_ENHMETAFILE" => CF_ENHMETAFILE,
            "CF_HDROP" => CF_HDROP,
            "CF_LOCALE" => CF_LOCALE,
            "CF_DIBV5" => CF_DIBV5,
            "CF_OWNERDISPLAY" => CF_OWNERDISPLAY,
            "CF_DSPTEXT" => CF_DSPTEXT,
            "CF_DSPBITMAP" => CF_DSPBITMAP,
            "CF_DSPMETAFILEPICT" => CF_DSPMETAFILEPICT,
            "CF_DSPENHMETAFILE" => CF_DSPENHMETAFILE,
            _ => return None,
        })
    }

    /// Returns true for predefined formats whose payload is a GDI handle
    /// or owner-display sentinel rather than plain memory — callers must
    /// skip these during snapshot because GlobalSize/GlobalLock wouldn't
    /// return usable bytes.
    pub fn is_skipped_format(id: u32) -> bool {
        matches!(
            id,
            CF_BITMAP
                | CF_METAFILEPICT
                | CF_PALETTE
                | CF_ENHMETAFILE
                | CF_OWNERDISPLAY
                | CF_DSPTEXT
                | CF_DSPBITMAP
                | CF_DSPMETAFILEPICT
                | CF_DSPENHMETAFILE
        ) || (CF_PRIVATEFIRST..=CF_PRIVATELAST).contains(&id)
            || (CF_GDIOBJFIRST..=CF_GDIOBJLAST).contains(&id)
    }

    /// Auto-synthesised formats that Windows regenerates from
    /// CF_UNICODETEXT on demand. Safe to skip during save; restore
    /// lets `SetClipboardData(CF_UNICODETEXT)` re-derive them.
    pub fn is_auto_synthesised(id: u32) -> bool {
        matches!(id, CF_TEXT | CF_OEMTEXT | CF_LOCALE)
    }

    /// RAII wrapper around `OpenClipboard` / `CloseClipboard`.
    ///
    /// The clipboard is a global exclusive resource — only one process at
    /// a time holds the handle. `OpenClipboard` fails with
    /// ERROR_ACCESS_DENIED when another process is mid-paste; the retry
    /// loop here absorbs the common transient case without bubbling a
    /// user-visible error.
    pub struct ClipboardGuard;

    impl ClipboardGuard {
        pub fn open() -> Result<Self, String> {
            const MAX_ATTEMPTS: usize = 10;
            const RETRY_DELAY: Duration = Duration::from_millis(10);
            let mut last_err: Option<windows::core::Error> = None;
            for _ in 0..MAX_ATTEMPTS {
                let result = unsafe { OpenClipboard(Some(HWND(std::ptr::null_mut()))) };
                match result {
                    Ok(()) => return Ok(Self),
                    Err(e) => {
                        last_err = Some(e);
                        thread::sleep(RETRY_DELAY);
                    }
                }
            }
            Err(format!(
                "OpenClipboard failed after {} retries ({:?}). Another process likely holds the clipboard open.",
                MAX_ATTEMPTS, last_err
            ))
        }
    }

    impl Drop for ClipboardGuard {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseClipboard();
            }
        }
    }

    /// Read the full payload for `format` from the currently open
    /// clipboard into an owned `Vec<u8>`. Returns `Ok(None)` when the
    /// clipboard advertises the format but provides no concrete data
    /// (delay-rendered format that's never been realised).
    pub fn read_format_bytes(format: u32) -> Result<Option<Vec<u8>>, String> {
        unsafe {
            let handle = GetClipboardData(format)
                .map_err(|e| format!("GetClipboardData({format}) failed: {e}"))?;
            if handle.is_invalid() {
                return Ok(None);
            }
            let hglobal = HGLOBAL(handle.0);
            let size = GlobalSize(hglobal);
            if size == 0 {
                return Ok(Some(Vec::new()));
            }
            let ptr = GlobalLock(hglobal);
            if ptr.is_null() {
                return Err(format!(
                    "GlobalLock returned null for format {format} (size {size})"
                ));
            }
            let bytes = std::slice::from_raw_parts(ptr as *const u8, size).to_vec();
            let _ = GlobalUnlock(hglobal);
            Ok(Some(bytes))
        }
    }

    /// Look up the name for a registered format ID (>= 0xC000). Returns
    /// `None` for unnamed predefined IDs — the caller should have used
    /// [`predefined_name`] first.
    pub fn registered_name(id: u32) -> Option<String> {
        let mut buf = [0u16; 256];
        let len = unsafe { GetClipboardFormatNameW(id, &mut buf) };
        if len <= 0 {
            return None;
        }
        String::from_utf16(&buf[..len as usize]).ok()
    }

    /// Allocate a movable HGLOBAL, copy `bytes` in, return the handle
    /// ready for `SetClipboardData`. On success ownership transfers to
    /// the clipboard; on failure the caller must `GlobalFree`.
    pub fn allocate_global(bytes: &[u8]) -> Result<HGLOBAL, String> {
        if bytes.is_empty() {
            // `GlobalAlloc(_, 0)` returns NULL, which `SetClipboardData`
            // would then reject as an invalid handle. Pad to one byte so
            // the format still round-trips (the receiving app already
            // has to handle zero-content payloads via GlobalSize).
            return allocate_global(&[0u8]);
        }
        unsafe {
            let hglobal = GlobalAlloc(GMEM_MOVEABLE, bytes.len())
                .map_err(|e| format!("GlobalAlloc({}) failed: {e}", bytes.len()))?;
            let ptr = GlobalLock(hglobal);
            if ptr.is_null() {
                let _ = GlobalFree(Some(hglobal));
                return Err("GlobalLock returned null after GlobalAlloc".into());
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
            let _ = GlobalUnlock(hglobal);
            Ok(hglobal)
        }
    }

    /// Push one format's payload onto the currently open clipboard.
    /// On `SetClipboardData` success the HGLOBAL becomes the clipboard's
    /// responsibility — do not free. On failure, free it ourselves.
    pub fn put_format(format: u32, bytes: &[u8]) -> Result<(), String> {
        let hglobal = allocate_global(bytes)?;
        let handle = HANDLE(hglobal.0);
        unsafe {
            match SetClipboardData(format, Some(handle)) {
                Ok(_) => Ok(()),
                Err(e) => {
                    let _ = GlobalFree(Some(hglobal));
                    Err(format!("SetClipboardData({format}) failed: {e}"))
                }
            }
        }
    }

    /// UTF-16 encode `s` with a trailing null code unit and push it as
    /// CF_UNICODETEXT.
    pub fn put_unicode_text(s: &str) -> Result<(), String> {
        let mut utf16: Vec<u16> = s.encode_utf16().collect();
        utf16.push(0);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                utf16.as_ptr() as *const u8,
                utf16.len() * std::mem::size_of::<u16>(),
            )
        };
        put_format(CF_UNICODETEXT, bytes)
    }

    /// Walk every format currently on the clipboard. `EnumClipboardFormats(0)`
    /// returns the first; each subsequent call with the previous format
    /// returns the next, until it returns 0 (or an error).
    pub fn enumerate_formats() -> Vec<u32> {
        let mut out = Vec::new();
        let mut current = 0u32;
        loop {
            let next = unsafe { EnumClipboardFormats(current) };
            if next == 0 {
                break;
            }
            out.push(next);
            current = next;
        }
        out
    }

    /// Resolve a snapshot's format name back to the u32 format ID.
    /// Registered names (anything not predefined) go through
    /// `RegisterClipboardFormatW`, which is idempotent — the same name
    /// yields the same ID within a Windows session.
    pub fn resolve_format_id(name: &str) -> Result<u32, String> {
        if let Some(id) = predefined_id(name) {
            return Ok(id);
        }
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        let id = unsafe { RegisterClipboardFormatW(PCWSTR(wide.as_ptr())) };
        if id == 0 {
            return Err(format!("RegisterClipboardFormatW failed for {name:?}"));
        }
        Ok(id)
    }

    pub fn sequence_number() -> u32 {
        unsafe { GetClipboardSequenceNumber() }
    }

    pub fn empty() -> Result<(), String> {
        unsafe { EmptyClipboard().map_err(|e| format!("EmptyClipboard failed: {e}")) }
    }
}

#[cfg(target_os = "windows")]
pub fn current_change_count() -> Result<i64, String> {
    Ok(win::sequence_number() as i64)
}

#[cfg(target_os = "windows")]
pub fn save_clipboard() -> Result<ClipboardSnapshot, String> {
    let change_count = win::sequence_number() as i64;
    let _guard = win::ClipboardGuard::open()?;

    let formats = win::enumerate_formats();
    let mut pairs: Vec<(String, Vec<u8>)> = Vec::with_capacity(formats.len());
    for id in formats {
        if win::is_skipped_format(id) || win::is_auto_synthesised(id) {
            continue;
        }
        let name = match win::predefined_name(id) {
            Some(n) => n.to_string(),
            None => match win::registered_name(id) {
                Some(n) => n,
                None => continue,
            },
        };
        match win::read_format_bytes(id) {
            Ok(Some(bytes)) => pairs.push((name, bytes)),
            Ok(None) => {}
            Err(_) => {
                // Single-format read failure (delay-render that never
                // materialises, ACL-restricted format, etc.) shouldn't
                // abort the whole snapshot — drop this format and keep
                // going so the user's other clipboard contents still
                // survive the round-trip.
                continue;
            }
        }
    }

    let items = if pairs.is_empty() {
        Vec::new()
    } else {
        vec![pairs]
    };

    Ok(ClipboardSnapshot {
        items,
        change_count,
    })
}

#[cfg(target_os = "windows")]
pub fn write_text(text: &str) -> Result<i64, String> {
    let _guard = win::ClipboardGuard::open()?;
    win::empty()?;
    win::put_unicode_text(text)?;
    // `GetClipboardSequenceNumber` reflects the post-write value as soon
    // as `SetClipboardData` returns.
    Ok(win::sequence_number() as i64)
}

#[cfg(target_os = "windows")]
pub fn restore_clipboard(snapshot: &ClipboardSnapshot) -> Result<(), String> {
    let _guard = win::ClipboardGuard::open()?;
    win::empty()?;

    for pairs in &snapshot.items {
        for (name, bytes) in pairs {
            let id = match win::resolve_format_id(name) {
                Ok(id) => id,
                Err(_) => continue,
            };
            // Per-format failures here also don't abort the whole
            // restore — better to get the user's text content back even
            // if a weird custom format can't be rehydrated.
            let _ = win::put_format(id, bytes);
        }
    }
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn current_change_count() -> Result<i64, String> {
    Err("clipboard snapshot is not yet implemented on this platform".into())
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn save_clipboard() -> Result<ClipboardSnapshot, String> {
    Err("clipboard snapshot is not yet implemented on this platform".into())
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn write_text(_text: &str) -> Result<i64, String> {
    Err("clipboard snapshot is not yet implemented on this platform".into())
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn restore_clipboard(_snapshot: &ClipboardSnapshot) -> Result<(), String> {
    Err("clipboard snapshot is not yet implemented on this platform".into())
}
