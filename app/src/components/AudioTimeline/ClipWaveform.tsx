import { useEffect, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { cn } from '@/lib/utils/cn';

interface ClipWaveformProps {
  audioUrl: string;
  width: number;
  durationMs: number;
  trimStartMs?: number;
  trimEndMs?: number;
  height?: number;
  className?: string;
}

export function ClipWaveform({
  audioUrl,
  width,
  durationMs,
  trimStartMs = 0,
  trimEndMs = 0,
  height = 28,
  className,
}: ClipWaveformProps) {
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);

  const effectiveDurationMs = durationMs - trimStartMs - trimEndMs;
  const fullWaveformWidth =
    effectiveDurationMs > 0 ? (width / effectiveDurationMs) * durationMs : width;
  const offsetX = effectiveDurationMs > 0 ? (trimStartMs / durationMs) * fullWaveformWidth : 0;

  useEffect(() => {
    if (!waveformRef.current || fullWaveformWidth < 20) return;

    const root = document.documentElement;
    const getCSSVar = (varName: string) => {
      const value = getComputedStyle(root).getPropertyValue(varName).trim();
      return value ? `hsl(${value})` : '';
    };
    const waveColor = getCSSVar('--accent-foreground');

    const mediaElement = document.createElement('audio');
    mediaElement.muted = true;
    mediaElement.preload = 'metadata';

    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      media: mediaElement,
      waveColor,
      progressColor: waveColor,
      cursorWidth: 0,
      barWidth: 1,
      barRadius: 1,
      barGap: 1,
      height,
      normalize: true,
      interact: false,
    });

    wavesurferRef.current = wavesurfer;
    wavesurfer.load(audioUrl).catch(() => {
      // Visual-only waveform; playback is handled by the owning timeline.
    });

    return () => {
      wavesurfer.destroy();
      wavesurferRef.current = null;
    };
  }, [audioUrl, fullWaveformWidth, height]);

  return (
    <div className={cn('h-full w-full overflow-hidden opacity-60', className)}>
      <div
        ref={waveformRef}
        className="h-full"
        style={{
          width: `${fullWaveformWidth}px`,
          transform: `translateX(-${offsetX}px)`,
        }}
      />
    </div>
  );
}
