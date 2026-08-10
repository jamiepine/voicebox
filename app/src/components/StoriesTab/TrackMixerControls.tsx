import { Headphones, VolumeX, Waves } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Slider } from '@/components/ui/slider';
import type { StoryTrackResponse } from '@/lib/api/types';
import { cn } from '@/lib/utils/cn';

interface TrackMixerControlsProps {
  /** Lane index. Lanes are sparse integers and may be negative. */
  index: number;
  /** Undefined when the lane has no settings row — it mixes at unity gain. */
  track?: StoryTrackResponse;
  /** Other lane indices, for the duck-under menu. */
  otherTracks: number[];
  /** True when any lane in the story is soloed, so this one may be implicitly silent. */
  anySoloed: boolean;
  onChange: (patch: Partial<StoryTrackResponse>) => void;
}

const DEFAULTS = { volume: 1, muted: false, soloed: false, duck_under_track: null } as const;

/**
 * Mute / solo / volume / ducking for one timeline lane.
 *
 * A lane without a settings row is not a special case — it renders these same
 * controls at their defaults and creates the row on first change. That mirrors
 * the mixer, which treats a missing row as defaults rather than as exempt.
 */
export function TrackMixerControls({
  index,
  track,
  otherTracks,
  anySoloed,
  onChange,
}: TrackMixerControlsProps) {
  const { t } = useTranslation();

  const volume = track?.volume ?? DEFAULTS.volume;
  const muted = track?.muted ?? DEFAULTS.muted;
  const soloed = track?.soloed ?? DEFAULTS.soloed;
  const duckUnder = track?.duck_under_track ?? DEFAULTS.duck_under_track;

  // Silent because something *else* is soloed — worth showing differently from
  // an explicit mute, so the user knows why a lane went quiet.
  const dimmedBySolo = anySoloed && !soloed && !muted;

  // The slider drives local state while dragging and only persists on
  // release. Writing on every step fired dozens of PUTs per drag, each
  // invalidating the track query, and out-of-order responses snapped the
  // thumb backwards mid-gesture. Same approach as ClipVolumePopover.
  const [localVolume, setLocalVolume] = useState(volume);
  // Re-sync when the persisted value changes from elsewhere, or when this
  // row is reused for a different lane.
  useEffect(() => {
    setLocalVolume(volume);
  }, [volume]);

  const patch = (changes: Partial<StoryTrackResponse>) =>
    onChange({ volume, muted, soloed, duck_under_track: duckUnder, ...changes });

  return (
    <div className="flex items-center gap-1">
      <Button
        variant="ghost"
        size="icon"
        className={cn('h-5 w-5', muted && 'text-destructive')}
        aria-label={t('storyTracks.muteTrack', { index })}
        aria-pressed={muted}
        onClick={() => patch({ muted: !muted })}
      >
        <VolumeX className="h-3 w-3" />
      </Button>

      <Button
        variant="ghost"
        size="icon"
        className={cn('h-5 w-5', soloed && 'text-accent', dimmedBySolo && 'opacity-40')}
        aria-label={t('storyTracks.soloTrack', { index })}
        aria-pressed={soloed}
        onClick={() => patch({ soloed: !soloed })}
      >
        <Headphones className="h-3 w-3" />
      </Button>

      <Slider
        value={[localVolume]}
        min={0}
        max={2}
        step={0.05}
        className="w-16"
        aria-label={t('storyTracks.volumeTrack', { index })}
        onValueChange={([next]) => setLocalVolume(next)}
        onValueCommit={([next]) => patch({ volume: next })}
      />

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className={cn('h-5 w-5', duckUnder !== null && 'text-accent')}
            aria-label={t('storyTracks.duck')}
          >
            <Waves className="h-3 w-3" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuLabel>{t('storyTracks.duckUnder')}</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            disabled={duckUnder === null}
            onClick={() => patch({ duck_under_track: null })}
          >
            {t('storyTracks.duckOff')}
          </DropdownMenuItem>
          {otherTracks.map((other) => (
            <DropdownMenuItem
              key={other}
              disabled={other === duckUnder}
              onClick={() => patch({ duck_under_track: other })}
            >
              {t('storyTracks.trackNumber', { index: other })}
            </DropdownMenuItem>
          ))}
          {otherTracks.length === 0 && (
            <DropdownMenuItem disabled>{t('storyTracks.noOtherTracks')}</DropdownMenuItem>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <span className="w-8 shrink-0 text-right text-[10px] tabular-nums text-muted-foreground">
        {Math.round(volume * 100)}%
      </span>
    </div>
  );
}
