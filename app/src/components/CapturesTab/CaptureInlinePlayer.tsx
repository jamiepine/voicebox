import { Loader2, Pause, Play } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils/cn';
import { debug } from '@/lib/utils/debug';

function formatDuration(ms?: number | null): string {
  if (!ms || ms < 0) return '0:00';
  const total = Math.round(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

export function CaptureInlinePlayer({
  audioUrl,
  fallbackDurationMs,
  className,
}: {
  audioUrl: string;
  fallbackDurationMs?: number | null;
  className?: string;
}) {
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const container = waveformRef.current;
    if (!container) return;

    const root = document.documentElement;
    const cssHsla = (varName: string, alpha: number) => {
      const value = getComputedStyle(root).getPropertyValue(varName).trim();
      if (!value) return '';
      const [h, s, l] = value.split(/\s+/);
      if (!h || !s || !l) return '';
      return `hsla(${h}, ${s}, ${l}, ${alpha})`;
    };

    const ws = WaveSurfer.create({
      container,
      waveColor: cssHsla('--muted-foreground', 1),
      progressColor: cssHsla('--accent', 1),
      cursorColor: 'transparent',
      barWidth: 2,
      barRadius: 2,
      barGap: 2,
      height: 40,
      normalize: true,
      interact: true,
      dragToSeek: { debounceTime: 0 },
      mediaControls: false,
      backend: 'WebAudio',
    });

    ws.on('ready', () => {
      setDuration(ws.getDuration());
      setIsLoading(false);
      setError(null);
    });
    ws.on('play', () => setIsPlaying(true));
    ws.on('pause', () => setIsPlaying(false));
    ws.on('finish', () => {
      setIsPlaying(false);
      setCurrentTime(ws.getDuration());
    });
    ws.on('timeupdate', (t) => setCurrentTime(t));
    ws.on('seeking', (t) => setCurrentTime(t));
    ws.on('error', (err) => {
      debug.error('Inline waveform error', err);
      setError(err instanceof Error ? err.message : String(err));
      setIsLoading(false);
    });

    wavesurferRef.current = ws;

    return () => {
      try {
        ws.destroy();
      } catch (err) {
        debug.error('Failed to destroy inline waveform', err);
      }
      wavesurferRef.current = null;
    };
  }, []);

  useEffect(() => {
    const ws = wavesurferRef.current;
    if (!ws) return;
    setIsLoading(true);
    setError(null);
    setCurrentTime(0);
    setDuration(0);
    setIsPlaying(false);
    try {
      if (ws.isPlaying()) ws.pause();
      ws.seekTo(0);
    } catch (err) {
      debug.error('Failed to reset inline waveform before load', err);
    }
    ws.load(audioUrl).catch((err) => {
      debug.error('Inline waveform load failed', err);
      setError(err instanceof Error ? err.message : String(err));
      setIsLoading(false);
    });
  }, [audioUrl]);

  const handlePlayPause = () => {
    const ws = wavesurferRef.current;
    if (!ws || isLoading) return;
    if (ws.isPlaying()) {
      ws.pause();
    } else {
      ws.play().catch((err) => {
        debug.error('Inline play failed', err);
        setError(err instanceof Error ? err.message : String(err));
      });
    }
  };

  const displayMs =
    duration > 0
      ? Math.round((isPlaying || currentTime > 0 ? currentTime : duration) * 1000)
      : (fallbackDurationMs ?? 0);

  return (
    <div className={cn('flex items-center gap-4', className)}>
      <Button
        size="icon"
        variant="outline"
        className="h-10 w-10 rounded-full shrink-0"
        onClick={handlePlayPause}
        disabled={isLoading || !!error}
        aria-label={isPlaying ? 'Pause' : 'Play'}
      >
        {isLoading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : isPlaying ? (
          <Pause className="h-4 w-4 fill-current" />
        ) : (
          <Play className="h-4 w-4 ml-0.5 fill-current" />
        )}
      </Button>
      <div ref={waveformRef} className="flex-1 min-w-0 h-10 select-none" />
      <span className="text-xs tabular-nums text-muted-foreground font-medium shrink-0">
        {error ? '—' : formatDuration(displayMs)}
      </span>
    </div>
  );
}
