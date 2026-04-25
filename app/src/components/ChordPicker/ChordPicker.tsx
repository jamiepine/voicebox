import { Keyboard } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  canonicalKeyFromEvent,
  displayLabelForKey,
  modifierSideHint,
  sortChordKeys,
} from '@/lib/utils/keyCodes';
import { cn } from '@/lib/utils/cn';

interface ChordPickerProps {
  open: boolean;
  /** Title shown in the modal — caller picks "push-to-talk" vs "toggle". */
  title: string;
  description?: string;
  /** The chord currently saved, shown as the starting state. */
  initialKeys: string[];
  onSave: (keys: string[]) => void;
  onCancel: () => void;
}

/**
 * Modal that captures a key chord from the browser keyboard. Tracks the
 * peak set of keys held during the session so the user can release
 * before clicking Save (otherwise they'd be saving while still holding
 * the shortcut, which is awkward).
 *
 * Browser limitation: we can only capture keys while Voicebox has key
 * focus, so the picker pulls focus to a hidden capture surface inside
 * the dialog. The actual chord runs through the Rust global hook —
 * this picker only writes the configuration the hook reads.
 */
export function ChordPicker({
  open,
  title,
  description,
  initialKeys,
  onSave,
  onCancel,
}: ChordPickerProps) {
  const { t } = useTranslation();
  // Currently held set, peak set captured this session, and "is the user
  // mid-chord?". We freeze the peak when they release everything so the
  // Save button can read a stable value.
  const [pressed, setPressed] = useState<Set<string>>(new Set());
  const [captured, setCaptured] = useState<string[]>(initialKeys);
  const [unsupportedAttempt, setUnsupportedAttempt] = useState<string | null>(null);
  const captureRef = useRef<HTMLDivElement>(null);

  // Reset every time the modal re-opens — otherwise the previous picker
  // session's peak set leaks into the next open and confuses the user.
  useEffect(() => {
    if (open) {
      setPressed(new Set());
      setCaptured(initialKeys);
      setUnsupportedAttempt(null);
      // Defer focus to the next paint so the dialog is mounted.
      const timeoutId = window.setTimeout(() => captureRef.current?.focus(), 50);
      return () => window.clearTimeout(timeoutId);
    }
    return;
  }, [open, initialKeys]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Esc reaches the dialog's onOpenChange and closes the modal — let
      // it pass through unmodified.
      if (event.key === 'Escape') return;
      // Tab cycles focus inside the dialog; capturing it would trap the
      // user. Same for the dialog's own keyboard interactions.
      if (event.key === 'Tab') return;

      const canonical = canonicalKeyFromEvent(event);
      if (!canonical) {
        setUnsupportedAttempt(event.code || event.key || 'unknown');
        event.preventDefault();
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      setUnsupportedAttempt(null);

      setPressed((prev) => {
        if (prev.has(canonical)) return prev;
        const next = new Set(prev);
        next.add(canonical);
        setCaptured((prevCaptured) => {
          const candidate = sortChordKeys(Array.from(next));
          // First key in a fresh sequence replaces the peak — otherwise a
          // user trying to swap a longer saved chord for a shorter one is
          // stuck because their candidate never beats the seed length.
          if (prev.size === 0) return candidate;
          return candidate.length >= prevCaptured.length ? candidate : prevCaptured;
        });
        return next;
      });
    },
    [],
  );

  const handleKeyUp = useCallback((event: KeyboardEvent) => {
    if (event.key === 'Escape' || event.key === 'Tab') return;
    const canonical = canonicalKeyFromEvent(event);
    if (!canonical) return;
    event.preventDefault();
    setPressed((prev) => {
      if (!prev.has(canonical)) return prev;
      const next = new Set(prev);
      next.delete(canonical);
      return next;
    });
  }, []);

  // Wire global listeners only while open. Capture phase so Voicebox's
  // own command palette / global shortcuts don't swallow the chord first.
  useEffect(() => {
    if (!open) return;
    window.addEventListener('keydown', handleKeyDown, true);
    window.addEventListener('keyup', handleKeyUp, true);
    return () => {
      window.removeEventListener('keydown', handleKeyDown, true);
      window.removeEventListener('keyup', handleKeyUp, true);
    };
  }, [open, handleKeyDown, handleKeyUp]);

  const displayKeys = pressed.size > 0
    ? sortChordKeys(Array.from(pressed))
    : captured;

  const canSave = captured.length > 0;

  return (
    <Dialog open={open} onOpenChange={(next) => { if (!next) onCancel(); }}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          {description ? <DialogDescription>{description}</DialogDescription> : null}
        </DialogHeader>

        <div
          ref={captureRef}
          tabIndex={-1}
          className="rounded-lg border border-border bg-muted/30 p-6 outline-none focus:ring-2 focus:ring-accent"
        >
          <div className="flex flex-col items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Keyboard className="h-3.5 w-3.5" />
              {pressed.size > 0 ? t('captures.chord.capturing') : t('captures.chord.pressShortcut')}
            </div>
            <div className="flex flex-wrap items-center justify-center gap-1.5 min-h-[2.5rem]">
              {displayKeys.length === 0 ? (
                <span className="text-sm text-muted-foreground italic">
                  {t('captures.chord.noKeys')}
                </span>
              ) : (
                displayKeys.map((k) => <ChordKey key={k} name={k} />)
              )}
            </div>
            {unsupportedAttempt ? (
              <p className="text-xs text-destructive">
                {t('captures.chord.unsupported', { key: unsupportedAttempt })}
              </p>
            ) : null}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            {t('common.cancel')}
          </Button>
          <Button onClick={() => onSave(captured)} disabled={!canSave}>
            {t('common.save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ChordKey({ name }: { name: string }) {
  const side = modifierSideHint(name);
  return (
    <span
      className={cn(
        'relative inline-flex items-center justify-center h-8 min-w-[2rem] px-2',
        'rounded-md border border-border bg-background font-mono text-sm font-medium',
        'shadow-sm text-foreground',
      )}
    >
      {displayLabelForKey(name)}
      {side ? (
        <span className="absolute -top-1 -right-1 h-3.5 min-w-[0.875rem] px-0.5 rounded-sm bg-accent text-[8px] font-bold leading-none flex items-center justify-center text-accent-foreground">
          {side}
        </span>
      ) : null}
    </span>
  );
}
