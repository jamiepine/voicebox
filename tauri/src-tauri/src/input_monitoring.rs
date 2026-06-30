//! Platform permission gate for the global keyboard tap.
//!
//! On macOS 10.15+, creating a CGEventTap that observes keyboard events
//! requires the host process to be listed under System Settings → Privacy &
//! Security → Input Monitoring. Without that trust, keytap's `Tap` returns
//! a permission error and no key events ever flow through the chord engine.
//!
//! The relevant TCC pair lives in IOKit, mirroring `AXIsProcessTrusted` /
//! `AXIsProcessTrustedWithOptions` on the Accessibility side:
//!
//!   - `IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)` — read the current
//!     grant without prompting. We call this from the Captures settings UI
//!     so the row can show "granted" / "missing" without surprising the user.
//!   - `IOHIDRequestAccess(kIOHIDRequestTypeListenEvent)` — fire the
//!     "Voicebox would like to receive keystrokes from any application"
//!     dialog and add Voicebox to the Input Monitoring pane (toggle off).
//!     Returns true when access is already granted; otherwise returns false
//!     and queues the prompt. The user still has to flip the toggle on; this
//!     just gets us into the list.
//!
//! `enable_hotkey` calls `request` on first invocation so the prompt fires
//! from a deterministic, user-initiated point (the Captures toggle) instead
//! of as a side-effect of keytap's `Tap` creating its CGEventTap.
//!
//! Windows / Linux don't gate keyboard taps behind a TCC-style permission,
//! so those branches return `true`.

#[cfg(target_os = "macos")]
mod ffi {
    use std::os::raw::c_uint;

    /// `kIOHIDRequestTypeListenEvent` from `<IOKit/hidsystem/IOHIDLib.h>` —
    /// the request-type discriminator for "I want to read keyboard / mouse
    /// events created by other processes."
    pub const REQUEST_TYPE_LISTEN_EVENT: c_uint = 1;

    /// `kIOHIDAccessTypeGranted` from `IOHIDLib.h`. The other values are
    /// `Denied = 1` and `Unknown = 2`; we only ever care about the granted
    /// case so they don't get their own constants.
    pub const ACCESS_TYPE_GRANTED: c_uint = 0;

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        /// Returns the current access state as an `IOHIDAccessType` enum
        /// (Granted=0, Denied=1, Unknown=2). No prompt side-effect.
        ///
        /// Declared as `c_uint` rather than `bool`: the C signature returns
        /// the full enum, and reading a 3-valued enum into Rust's 1-bit
        /// `bool` is undefined behaviour that silently inverts our gate.
        pub fn IOHIDCheckAccess(request_type: c_uint) -> c_uint;

        /// Returns true when access is already granted; otherwise queues
        /// the system prompt and returns false synchronously. Safe to call
        /// repeatedly — once the entry exists in the Input Monitoring pane
        /// macOS won't re-prompt. Real `Boolean` (UInt8) return on the C
        /// side, so `bool` here is correct.
        pub fn IOHIDRequestAccess(request_type: c_uint) -> bool;
    }
}

#[cfg(target_os = "macos")]
pub fn is_trusted() -> bool {
    unsafe { ffi::IOHIDCheckAccess(ffi::REQUEST_TYPE_LISTEN_EVENT) == ffi::ACCESS_TYPE_GRANTED }
}

/// Fire the Input Monitoring prompt if not already granted. Returns the
/// current grant state; a `false` here means the prompt was queued and the
/// user needs to flip the toggle in System Settings before key events flow.
#[cfg(target_os = "macos")]
pub fn request() -> bool {
    unsafe { ffi::IOHIDRequestAccess(ffi::REQUEST_TYPE_LISTEN_EVENT) }
}

#[cfg(not(target_os = "macos"))]
pub fn is_trusted() -> bool {
    true
}

#[cfg(not(target_os = "macos"))]
pub fn request() -> bool {
    true
}
