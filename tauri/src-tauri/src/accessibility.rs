//! Platform permission gate for the auto-paste pipeline.
//!
//! On macOS, posting synthetic keyboard events and reading focused-UI state
//! via the AX API both require the host process to be listed under System
//! Settings → Privacy & Security → Accessibility. Without that trust,
//! `CGEventPost` silently drops events and `AXUIElementCopyAttributeValue`
//! returns an error. We surface a boolean check up front so the paste
//! pipeline can short-circuit with a clear "grant permission" message
//! instead of running through the full save → write → post → restore dance
//! with nothing to show for it.
//!
//! Windows has no equivalent user-facing permission — `SendInput` and
//! UIAutomation work for any non-elevated target out of the box. (UAC /
//! UIPI still blocks sending input *into* an elevated target window from a
//! non-elevated process, but that's per-target, not a global switch, and
//! there's no Settings pane to send users to.) So the Windows branch just
//! returns `true`.

#[cfg(target_os = "macos")]
mod ffi {
    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        /// Returns true when the current process is listed in Accessibility.
        /// No prompt side-effect.
        pub fn AXIsProcessTrusted() -> bool;
    }
}

#[cfg(target_os = "macos")]
pub fn is_trusted() -> bool {
    unsafe { ffi::AXIsProcessTrusted() }
}

#[cfg(target_os = "windows")]
pub fn is_trusted() -> bool {
    true
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn is_trusted() -> bool {
    false
}
