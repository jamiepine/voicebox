import { Loader2, Play, Square } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { VoicePreviewState } from '@/lib/hooks/useVoicePreview';
import { cn } from '@/lib/utils/cn';

interface VoicePreviewButtonProps {
  /** Voice name, used for the accessible label. */
  name: string;
  state: VoicePreviewState;
  onToggle: () => void;
  className?: string;
}

/** Play / stop control for a voice's sample clip. Safe to place inside a clickable row or a form. */
export function VoicePreviewButton({ name, state, onToggle, className }: VoicePreviewButtonProps) {
  const { t } = useTranslation();
  const label =
    state === 'idle'
      ? t('generation.voicePicker.preview', { name })
      : t('generation.voicePicker.stopPreview', { name });

  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={(event) => {
        event.stopPropagation();
        onToggle();
      }}
      className={cn(
        'flex h-7 w-7 shrink-0 items-center justify-center rounded-full border transition-colors',
        state === 'idle'
          ? 'border-border bg-card text-muted-foreground hover:bg-accent hover:text-accent-foreground'
          : 'border-accent bg-accent text-accent-foreground',
        className,
      )}
    >
      {state === 'loading' ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
      ) : state === 'playing' ? (
        <Square className="h-3 w-3 fill-current" />
      ) : (
        <Play className="h-3.5 w-3.5 fill-current" />
      )}
    </button>
  );
}
