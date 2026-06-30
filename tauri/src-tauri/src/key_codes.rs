//! Stable string ↔ `keytap::Key` mapping for chord persistence.
//!
//! The frontend captures keypresses through the browser keyboard API (which
//! exposes `event.code` like `"MetaRight"`, `"AltRight"`, `"Space"`, `"KeyA"`)
//! and stores chords in capture_settings as JSON arrays of canonical names.
//! On the way back the same names need to round-trip into `keytap::Key`
//! variants the chord engine actually matches against.
//!
//! Input strings follow the W3C `KeyboardEvent.code` identifiers exactly —
//! `"MetaRight"`, `"AltRight"`, `"KeyA"`, `"Digit0"`, `"ArrowUp"`, … —
//! which is also what the browser emits natively, so on-disk chords
//! round-trip without translation on the frontend side. Legacy aliases
//! (`"Alt"` / `"AltGr"` / `"Num0"` / `"UpArrow"` / …) are accepted too so
//! older capture_settings rows written before the keytap swap keep working.

use keytap::Key;

/// Resolve a canonical key name to its `keytap::Key`. Returns `None` for
/// names that don't have a corresponding variant — the command surface
/// rejects those so we never silently drop keys from a chord.
pub fn key_from_str(name: &str) -> Option<Key> {
    Some(match name {
        // Modifiers — left/right distinction matters for chord defaults.
        "AltLeft" | "Alt" => Key::AltLeft,
        "AltRight" | "AltGr" => Key::AltRight,
        "ControlLeft" => Key::ControlLeft,
        "ControlRight" => Key::ControlRight,
        "MetaLeft" => Key::MetaLeft,
        "MetaRight" => Key::MetaRight,
        "ShiftLeft" => Key::ShiftLeft,
        "ShiftRight" => Key::ShiftRight,
        "CapsLock" => Key::CapsLock,

        // Whitespace / navigation
        "Space" => Key::Space,
        "Tab" => Key::Tab,
        "Enter" | "Return" => Key::Enter,
        "Backspace" => Key::Backspace,
        "Delete" => Key::Delete,
        "Escape" => Key::Escape,
        "Insert" => Key::Insert,
        "Home" => Key::Home,
        "End" => Key::End,
        "PageUp" => Key::PageUp,
        "PageDown" => Key::PageDown,
        "ArrowUp" | "UpArrow" => Key::ArrowUp,
        "ArrowDown" | "DownArrow" => Key::ArrowDown,
        "ArrowLeft" | "LeftArrow" => Key::ArrowLeft,
        "ArrowRight" | "RightArrow" => Key::ArrowRight,

        // Function row
        "F1" => Key::F1, "F2" => Key::F2, "F3" => Key::F3, "F4" => Key::F4,
        "F5" => Key::F5, "F6" => Key::F6, "F7" => Key::F7, "F8" => Key::F8,
        "F9" => Key::F9, "F10" => Key::F10, "F11" => Key::F11, "F12" => Key::F12,

        // Digits
        "Digit0" | "Num0" => Key::Digit0,
        "Digit1" | "Num1" => Key::Digit1,
        "Digit2" | "Num2" => Key::Digit2,
        "Digit3" | "Num3" => Key::Digit3,
        "Digit4" | "Num4" => Key::Digit4,
        "Digit5" | "Num5" => Key::Digit5,
        "Digit6" | "Num6" => Key::Digit6,
        "Digit7" | "Num7" => Key::Digit7,
        "Digit8" | "Num8" => Key::Digit8,
        "Digit9" | "Num9" => Key::Digit9,

        // Letters — browser emits "KeyA"; keytap uses the bare letter.
        "KeyA" => Key::A, "KeyB" => Key::B, "KeyC" => Key::C,
        "KeyD" => Key::D, "KeyE" => Key::E, "KeyF" => Key::F,
        "KeyG" => Key::G, "KeyH" => Key::H, "KeyI" => Key::I,
        "KeyJ" => Key::J, "KeyK" => Key::K, "KeyL" => Key::L,
        "KeyM" => Key::M, "KeyN" => Key::N, "KeyO" => Key::O,
        "KeyP" => Key::P, "KeyQ" => Key::Q, "KeyR" => Key::R,
        "KeyS" => Key::S, "KeyT" => Key::T, "KeyU" => Key::U,
        "KeyV" => Key::V, "KeyW" => Key::W, "KeyX" => Key::X,
        "KeyY" => Key::Y, "KeyZ" => Key::Z,

        // Punctuation / symbols
        "Backquote" | "BackQuote" => Key::Backtick,
        "Minus" => Key::Minus,
        "Equal" => Key::Equal,
        "BracketLeft" | "LeftBracket" => Key::BracketLeft,
        "BracketRight" | "RightBracket" => Key::BracketRight,
        "Semicolon" | "SemiColon" => Key::Semicolon,
        "Quote" => Key::Quote,
        "Backslash" | "BackSlash" => Key::Backslash,
        "Comma" => Key::Comma,
        "Period" | "Dot" => Key::Period,
        "Slash" => Key::Slash,

        _ => return None,
    })
}
