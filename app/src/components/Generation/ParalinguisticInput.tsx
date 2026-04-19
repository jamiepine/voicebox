/**
 * ParalinguisticInput — a contentEditable rich text input that renders
 * Chatterbox Turbo paralinguistic tags (e.g. [laugh]) as inline badges.
 *
 * Trigger: typing "/" opens an autocomplete dropdown.
 * Paste:   pasting text with [tag] patterns auto-converts to badges.
 * Output:  serializes badges back to plain [tag] text for the API.
 */

import { AnimatePresence, motion } from 'framer-motion';
import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { cn } from '@/lib/utils/cn';

// ── Tag definitions ─────────────────────────────────────────────────
const PARALINGUISTIC_TAGS = [
  { tag: '[laugh]', label: 'laugh', emoji: '\u{1F602}' },
  { tag: '[chuckle]', label: 'chuckle', emoji: '\u{1F60F}' },
  { tag: '[gasp]', label: 'gasp', emoji: '\u{1F62E}' },
  { tag: '[cough]', label: 'cough', emoji: '\u{1F637}' },
  { tag: '[sigh]', label: 'sigh', emoji: '\u{1F614}' },
  { tag: '[groan]', label: 'groan', emoji: '\u{1F629}' },
  { tag: '[sniff]', label: 'sniff', emoji: '\u{1F443}' },
  { tag: '[shush]', label: 'shush', emoji: '\u{1F92B}' },
  { tag: '[clear throat]', label: 'clear throat', emoji: '\u{1F64A}' },
] as const;

const TAG_REGEX = /\[(laugh|chuckle|gasp|cough|sigh|groan|sniff|shush|clear throat)\]/gi;

// Data attribute used to identify badge spans in the DOM
const BADGE_ATTR = 'data-ptag';

// ── Helpers ─────────────────────────────────────────────────────────

/** Build an inline badge <span> for a tag. */
function makeBadgeHTML(tag: string): string {
  const entry = PARALINGUISTIC_TAGS.find((t) => t.tag.toLowerCase() === tag.toLowerCase());
  const label = entry?.label ?? tag.replace(/[[\]]/g, '');
  const emoji = entry?.emoji ?? '';
  // Non-editable inline badge. Zero-width spaces around it let the
  // caret sit on either side so the user can type before/after.
  return `\u200B<span ${BADGE_ATTR}="${tag}" contenteditable="false" class="ptag-badge">${emoji ? `${emoji}\u00A0` : ''}${label}</span>\u200B`;
}

/** Convert plain text with [tag] patterns into HTML with badge spans. */
function textToHTML(text: string): string {
  // Escape HTML entities first
  const escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  // Replace tag patterns with badge HTML
  return escaped.replace(TAG_REGEX, (match) => makeBadgeHTML(match));
}

/** Serialize the contentEditable innerHTML back to plain text with [tag] syntax. */
function htmlToText(container: HTMLElement): string {
  let result = '';
  for (const node of container.childNodes) {
    if (node.nodeType === Node.TEXT_NODE) {
      // Strip zero-width spaces we added around badges
      result += (node.textContent ?? '').replace(/\u200B/g, '');
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      const el = node as HTMLElement;
      if (el.hasAttribute(BADGE_ATTR)) {
        result += el.getAttribute(BADGE_ATTR) ?? '';
      } else if (el.tagName === 'BR') {
        result += '\n';
      } else {
        // Recurse for nested elements (e.g. spans from paste)
        result += htmlToText(el);
      }
    }
  }
  return result;
}

/** Get the text content from the current caret position back to the last
 *  whitespace or start of container, to detect the "/" trigger. */
function getWordBeforeCaret(_container: HTMLElement): { word: string; range: Range | null } {
  const sel = window.getSelection();
  if (!sel || sel.rangeCount === 0) return { word: '', range: null };
  const range = sel.getRangeAt(0).cloneRange();
  range.collapse(true);

  // Walk backwards from caret through the text node
  const textNode = range.startContainer;
  if (textNode.nodeType !== Node.TEXT_NODE) return { word: '', range: null };
  const text = textNode.textContent ?? '';
  const offset = range.startOffset;

  let start = offset;
  while (
    start > 0 &&
    text[start - 1] !== ' ' &&
    text[start - 1] !== '\n' &&
    text[start - 1] !== '\u00A0'
  ) {
    start--;
  }

  const word = text.slice(start, offset);
  const wordRange = document.createRange();
  wordRange.setStart(textNode, start);
  wordRange.setEnd(textNode, offset);

  return { word, range: wordRange };
}

// ── Component ───────────────────────────────────────────────────────

export interface ParalinguisticInputProps {
  value?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
  style?: React.CSSProperties;
  onClick?: () => void;
  onFocus?: () => void;
}

export interface ParalinguisticInputRef {
  focus: () => void;
  element: HTMLDivElement | null;
}

export const ParalinguisticInput = forwardRef<ParalinguisticInputRef, ParalinguisticInputProps>(
  function ParalinguisticInput(
    { value, onChange, placeholder, disabled, className, style, onClick, onFocus },
    ref,
  ) {
    const editorRef = useRef<HTMLDivElement>(null);
    const [showMenu, setShowMenu] = useState(false);
    const [menuFilter, setMenuFilter] = useState('');
    const [menuIndex, setMenuIndex] = useState(0);
    const [menuPosition, setMenuPosition] = useState<{ bottom: number; left: number }>({
      bottom: 0,
      left: 0,
    });
    const triggerRangeRef = useRef<Range | null>(null);
    const lastSerializedRef = useRef<string>('');
    const isComposingRef = useRef(false);

    useImperativeHandle(ref, () => ({
      focus: () => editorRef.current?.focus(),
      element: editorRef.current,
    }));

    // Filtered tag list for the autocomplete menu
    const filteredTags = PARALINGUISTIC_TAGS.filter((t) =>
      t.label.toLowerCase().includes(menuFilter.toLowerCase()),
    );

    // ── Sync external value → editor ──────────────────────────────
    useEffect(() => {
      const el = editorRef.current;
      if (!el) return;
      // Only update DOM if the external value differs from what we last emitted
      if (value !== undefined && value !== lastSerializedRef.current) {
        lastSerializedRef.current = value;
        el.innerHTML = value ? textToHTML(value) : '';
      }
    }, [value]);

    // ── Emit plain-text value on input ────────────────────────────
    const emitChange = useCallback(() => {
      const el = editorRef.current;
      if (!el || !onChange) return;
      const text = htmlToText(el);
      lastSerializedRef.current = text;
      onChange(text);
    }, [onChange]);

    // ── Insert a tag badge at the caret ───────────────────────────
    const insertTag = useCallback(
      (tag: string) => {
        const el = editorRef.current;
        if (!el) return;

        // Delete the /filter text
        const wordRange = triggerRangeRef.current;
        if (wordRange) {
          wordRange.deleteContents();
        }

        // Insert badge HTML
        const temp = document.createElement('span');
        temp.innerHTML = makeBadgeHTML(tag);
        const frag = document.createDocumentFragment();
        let lastNode: Node | null = null;
        while (temp.firstChild) {
          lastNode = frag.appendChild(temp.firstChild);
        }

        const sel = window.getSelection();
        if (sel && sel.rangeCount > 0) {
          const range = sel.getRangeAt(0);
          range.deleteContents();
          range.insertNode(frag);

          // Move caret after the badge
          if (lastNode) {
            const newRange = document.createRange();
            newRange.setStartAfter(lastNode);
            newRange.collapse(true);
            sel.removeAllRanges();
            sel.addRange(newRange);
          }
        }

        setShowMenu(false);
        setMenuFilter('');
        emitChange();
        el.focus();
      },
      [emitChange],
    );

    // ── Handle keydown for autocomplete navigation ────────────────
    const handleKeyDown = useCallback(
      (e: React.KeyboardEvent) => {
        if (showMenu) {
          if (filteredTags.length === 0) {
            if (e.key === 'Escape') {
              e.preventDefault();
              setShowMenu(false);
            }
            return;
          }
          if (e.key === 'ArrowDown') {
            e.preventDefault();
            setMenuIndex((i) => (i + 1) % filteredTags.length);
          } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setMenuIndex((i) => (i - 1 + filteredTags.length) % filteredTags.length);
          } else if (e.key === 'Enter' || e.key === 'Tab') {
            e.preventDefault();
            if (filteredTags[menuIndex]) {
              insertTag(filteredTags[menuIndex].tag);
            }
          } else if (e.key === 'Escape') {
            e.preventDefault();
            setShowMenu(false);
          }
        } else {
          // Prevent Enter from creating <div> blocks in contentEditable
          if (e.key === 'Enter' && !e.shiftKey) {
            // Let the form handle submit
          }
        }
      },
      [showMenu, filteredTags, menuIndex, insertTag],
    );

    // ── Handle input (check for / trigger) ────────────────────────
    const handleInput = useCallback(() => {
      if (isComposingRef.current) return;
      const el = editorRef.current;
      if (!el) return;

      const { word, range } = getWordBeforeCaret(el);

      if (word.startsWith('/')) {
        const filter = word.slice(1); // strip the /
        setMenuFilter(filter);
        setMenuIndex(0);
        triggerRangeRef.current = range;

        // Position the menu above the caret using viewport coords (portalled)
        const sel = window.getSelection();
        if (sel && sel.rangeCount > 0) {
          const rect = sel.getRangeAt(0).getBoundingClientRect();
          setMenuPosition({
            bottom: window.innerHeight - rect.top + 4,
            left: rect.left,
          });
        }

        setShowMenu(true);
      } else {
        setShowMenu(false);
      }

      emitChange();
    }, [emitChange]);

    // ── Handle paste — convert [tag] patterns to badges ───────────
    const handlePaste = useCallback(
      (e: React.ClipboardEvent) => {
        e.preventDefault();
        const text = e.clipboardData.getData('text/plain');
        if (!text) return;

        const el = editorRef.current;
        if (!el) return;

        const html = textToHTML(text);

        // Insert at caret
        const sel = window.getSelection();
        if (sel && sel.rangeCount > 0) {
          const range = sel.getRangeAt(0);
          range.deleteContents();
          const temp = document.createElement('div');
          temp.innerHTML = html;
          const frag = document.createDocumentFragment();
          let lastNode: Node | null = null;
          while (temp.firstChild) {
            lastNode = frag.appendChild(temp.firstChild);
          }
          range.insertNode(frag);
          if (lastNode) {
            const newRange = document.createRange();
            newRange.setStartAfter(lastNode);
            newRange.collapse(true);
            sel.removeAllRanges();
            sel.addRange(newRange);
          }
        }

        emitChange();
      },
      [emitChange],
    );

    // ── Show placeholder ──────────────────────────────────────────
    const isEmpty = !value || value.trim() === '';

    return (
      <div className="relative">
        {/* Placeholder */}
        {isEmpty && placeholder && (
          <div
            className="pointer-events-none absolute inset-0 text-sm text-muted-foreground/60 px-3 py-2 select-none"
            aria-hidden
          >
            {placeholder}
          </div>
        )}

        {/* Editable area */}
        <div
          ref={editorRef}
          contentEditable={!disabled}
          suppressContentEditableWarning
          role={disabled ? undefined : 'textbox'}
          aria-multiline={disabled ? undefined : true}
          aria-placeholder={placeholder}
          aria-disabled={disabled}
          tabIndex={disabled ? -1 : 0}
          className={cn(
            'min-h-[32px] text-sm whitespace-pre-wrap break-words outline-none',
            '[&_.ptag-badge]:inline-flex [&_.ptag-badge]:items-center [&_.ptag-badge]:rounded-full',
            '[&_.ptag-badge]:bg-accent/20 [&_.ptag-badge]:text-accent [&_.ptag-badge]:border [&_.ptag-badge]:border-accent/30',
            '[&_.ptag-badge]:px-2 [&_.ptag-badge]:py-0 [&_.ptag-badge]:text-xs [&_.ptag-badge]:font-medium',
            '[&_.ptag-badge]:mx-0.5 [&_.ptag-badge]:select-none [&_.ptag-badge]:cursor-default',
            '[&_.ptag-badge]:align-baseline',
            disabled && 'opacity-50 cursor-not-allowed',
            className,
          )}
          style={style}
          onInput={!disabled ? handleInput : undefined}
          onKeyDown={!disabled ? handleKeyDown : undefined}
          onPaste={!disabled ? handlePaste : undefined}
          onClick={!disabled ? onClick : undefined}
          onFocus={!disabled ? onFocus : undefined}
          onBlur={() => {
            setShowMenu(false);
            triggerRangeRef.current = null;
          }}
          onCompositionStart={() => {
            isComposingRef.current = true;
          }}
          onCompositionEnd={() => {
            isComposingRef.current = false;
            handleInput();
          }}
        />

        {/* Autocomplete dropdown — portalled to body, positioned above the caret */}
        {showMenu &&
          filteredTags.length > 0 &&
          createPortal(
            <AnimatePresence>
              <motion.div
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 4 }}
                transition={{ duration: 0.12 }}
                className="fixed z-[9999] min-w-[200px] max-h-[280px] overflow-y-auto rounded-lg border border-border bg-popover shadow-lg"
                style={{
                  bottom: menuPosition.bottom,
                  left: menuPosition.left,
                }}
              >
                {filteredTags.map((t, i) => (
                  <button
                    key={t.tag}
                    type="button"
                    className={cn(
                      'flex items-center gap-2 w-full px-3 py-1.5 text-sm text-left transition-colors',
                      i === menuIndex
                        ? 'bg-accent/20 text-accent-foreground'
                        : 'text-popover-foreground hover:bg-muted/50',
                    )}
                    onMouseDown={(e) => {
                      e.preventDefault(); // Keep focus in editor
                      insertTag(t.tag);
                    }}
                    onMouseEnter={() => setMenuIndex(i)}
                  >
                    <span className="text-base leading-none">{t.emoji}</span>
                    <span>{t.label}</span>
                    <span className="ml-auto text-xs text-muted-foreground font-mono">{t.tag}</span>
                  </button>
                ))}
              </motion.div>
            </AnimatePresence>,
            document.body,
          )}
      </div>
    );
  },
);
