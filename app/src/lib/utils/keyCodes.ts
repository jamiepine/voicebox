/**
 * Stable key-name vocabulary shared with the Rust `key_codes` module.
 *
 * The chord persistence layer stores keytap `Key` variant names ("MetaRight",
 * "AltGr", "KeyA", …) so the same array round-trips losslessly between
 * the picker UI, the SQLite settings row, and the global hotkey listener.
 *
 * This module owns the conversions between three vocabularies:
 *   - browser `KeyboardEvent` (`event.code` like "MetaRight" / "AltRight")
 *   - canonical chord key names (matches keytap variants)
 *   - human display labels ("⌘", "⌥", "A", …)
 */

/**
 * Map a `KeyboardEvent` to the canonical key name we persist. Returns
 * `null` for keys we don't support in chords (dead keys, IME composition,
 * etc.).
 *
 * Browser quirk: right-Option on macOS is reported as `"AltRight"`; keytap
 * calls it `"AltGr"`. Normalize to keytap's name so the Rust side recognizes
 * it without an aliasing layer.
 */
export function canonicalKeyFromEvent(event: KeyboardEvent): string | null {
  const code = event.code;
  if (!code) return null;
  switch (code) {
    case 'AltLeft':
      return 'Alt';
    case 'AltRight':
      return 'AltGr';
    case 'BracketLeft':
      return 'LeftBracket';
    case 'BracketRight':
      return 'RightBracket';
    case 'Semicolon':
      return 'SemiColon';
    case 'Backslash':
      return 'BackSlash';
    case 'Backquote':
      return 'BackQuote';
    case 'Period':
      return 'Dot';
    case 'Enter':
      return 'Return';
    case 'ArrowUp':
      return 'UpArrow';
    case 'ArrowDown':
      return 'DownArrow';
    case 'ArrowLeft':
      return 'LeftArrow';
    case 'ArrowRight':
      return 'RightArrow';
    default:
      // Browser names like "MetaRight", "MetaLeft", "ControlLeft",
      // "ShiftRight", "Space", "KeyA", "Digit1", "F5" all match the
      // keytap variant names directly.
      if (
        /^(Meta|Control|Shift)(Left|Right)$/.test(code) ||
        /^Key[A-Z]$/.test(code) ||
        /^Digit[0-9]$/.test(code) ||
        /^F([1-9]|1[0-2])$/.test(code) ||
        ['Space', 'Tab', 'Backspace', 'Delete', 'Escape', 'Insert',
          'Home', 'End', 'PageUp', 'PageDown', 'CapsLock', 'Function',
          'Minus', 'Equal', 'Quote', 'Comma', 'Slash'].includes(code)
      ) {
        return code;
      }
      return null;
  }
}

const PLATFORM_IS_MAC =
  typeof navigator !== 'undefined' && /mac/i.test(navigator.platform);

export function defaultChordKeys(mode: 'push' | 'toggle'): string[] {
  const base = PLATFORM_IS_MAC
    ? ['MetaRight', 'AltGr']
    : ['ControlRight', 'ShiftRight'];
  return mode === 'toggle' ? [...base, 'Space'] : base;
}

/**
 * Pretty label for a canonical key name. Picks platform-appropriate
 * modifier glyphs so macOS users see ⌘ and Windows/Linux users see Win.
 */
export function displayLabelForKey(name: string): string {
  switch (name) {
    case 'MetaLeft':
    case 'MetaRight':
      return PLATFORM_IS_MAC ? '⌘' : 'Win';
    case 'Alt':
      return PLATFORM_IS_MAC ? '⌥' : 'Alt';
    case 'AltGr':
      return PLATFORM_IS_MAC ? '⌥' : 'AltGr';
    case 'ControlLeft':
    case 'ControlRight':
      return PLATFORM_IS_MAC ? '⌃' : 'Ctrl';
    case 'ShiftLeft':
    case 'ShiftRight':
      return PLATFORM_IS_MAC ? '⇧' : 'Shift';
    case 'CapsLock':
      return '⇪';
    case 'Function':
      return 'fn';
    case 'Space':
      return 'Space';
    case 'Tab':
      return '⇥';
    case 'Return':
      return '↵';
    case 'Backspace':
      return '⌫';
    case 'Delete':
      return '⌦';
    case 'Escape':
      return 'Esc';
    case 'UpArrow':
      return '↑';
    case 'DownArrow':
      return '↓';
    case 'LeftArrow':
      return '←';
    case 'RightArrow':
      return '→';
  }
  if (/^Key([A-Z])$/.test(name)) return name.slice(3);
  if (/^Num([0-9])$/.test(name)) return name.slice(3);
  if (/^F([1-9]|1[0-2])$/.test(name)) return name;
  return name;
}

/**
 * Side-aware suffix to disambiguate left vs right modifier variants
 * — the tiny "R" badge that lets a user see the chord defaults to the
 * right-hand keys.
 */
export function modifierSideHint(name: string): 'L' | 'R' | null {
  if (name === 'MetaRight' || name === 'AltGr' || name === 'ControlRight' || name === 'ShiftRight') {
    return 'R';
  }
  if (name === 'MetaLeft' || name === 'Alt' || name === 'ControlLeft' || name === 'ShiftLeft') {
    return 'L';
  }
  return null;
}

/**
 * Sort a chord's keys so the kbd pills always render in a predictable
 * order: modifiers first (Ctrl, Opt, Shift, Cmd), main key last. Matches
 * how every macOS shortcut docs list the keys.
 */
const SORT_ORDER: Record<string, number> = {
  ControlLeft: 0, ControlRight: 0,
  Alt: 1, AltGr: 1,
  ShiftLeft: 2, ShiftRight: 2,
  MetaLeft: 3, MetaRight: 3,
  Function: 4,
  CapsLock: 5,
};

export function sortChordKeys(keys: string[]): string[] {
  return [...keys].sort((a, b) => {
    const sa = SORT_ORDER[a] ?? 99;
    const sb = SORT_ORDER[b] ?? 99;
    if (sa !== sb) return sa - sb;
    return a.localeCompare(b);
  });
}
