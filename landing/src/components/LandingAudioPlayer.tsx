'use client';

import { Pause, Play, Repeat, Volume2, VolumeX } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// Shared ref so the unmute button can unlock WaveSurfer's audio on iOS Safari
// Must call .play() on WaveSurfer's actual media element during a user gesture
let sharedWaveSurfer: WaveSurfer | null = null;
let audioUnlocked = false;

export function unlockAudioContext() {
  if (audioUnlocked) return;
  audioUnlocked = true;

  // Unlock WaveSurfer's internal audio element
  // Skip if already playing — the context is already unlocked and the
  // play/pause/reset dance would destroy the active playback.
  if (sharedWaveSurfer && !sharedWaveSurfer.isPlaying()) {
    const media = sharedWaveSurfer.getMediaElement();
    if (media) {
      media.muted = true;
      media
        .play()
        .then(() => {
          media.pause();
          media.muted = false;
          media.currentTime = 0;
        })
        .catch(() => {});
    }
  }

  // Also unlock a standalone AudioContext as fallback
  try {
    const ctx = new (
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    )();
    const buffer = ctx.createBuffer(1, 1, 22050);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.start(0);
  } catch {
    // Silently fail
  }
}

interface LandingAudioPlayerProps {
  audioUrl: string;
  title: string;
  playing: boolean;
  muted: boolean;
  onFinish: () => void;
  onClose: () => void;
}

export function LandingAudioPlayer({
  audioUrl,
  title,
  playing,
  muted,
  onFinish,
  onClose,
}: LandingAudioPlayerProps) {
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.75);
  const [isLooping, setIsLooping] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const onFinishRef = useRef(onFinish);
  onFinishRef.current = onFinish;
  const playingRef = useRef(playing);
  playingRef.current = playing;
  const mutedRef = useRef(muted);
  mutedRef.current = muted;

  // Initialize WaveSurfer
  useEffect(() => {
    const initWaveSurfer = () => {
      const container = waveformRef.current;
      if (!container) {
        setTimeout(initWaveSurfer, 50);
        return;
      }

      const rect = container.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        setTimeout(initWaveSurfer, 50);
        return;
      }

      // Clean up existing instance
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
      }

      const root = document.documentElement;
      const getCSSVar = (varName: string) => {
        const value = getComputedStyle(root).getPropertyValue(varName).trim();
        return value ? `hsl(${value})` : '';
      };

      const ws = WaveSurfer.create({
        container,
        waveColor: getCSSVar('--muted'),
        progressColor: getCSSVar('--accent'),
        cursorColor: getCSSVar('--accent'),
        barWidth: 2,
        barRadius: 2,
        height: 80,
        normalize: true,
        interact: true,
        mediaControls: false,
      });

      ws.on('ready', () => {
        setDuration(ws.getDuration());
        ws.setVolume(mutedRef.current ? 0 : volume);
        setIsReady(true);
      });

      ws.on('play', () => {
        console.log('[Player] play event');
        setIsPlaying(true);
      });
      ws.on('pause', () => {
        console.log('[Player] pause event');
        setIsPlaying(false);
      });

      ws.on('timeupdate', (time: number) => {
        setCurrentTime(Math.min(time, ws.getDuration()));
      });

      let didFinish = false;
      ws.on('finish', () => {
        if (didFinish) return;
        didFinish = true;
        console.log(
          '[Player] finish event, currentTime:',
          ws.getCurrentTime(),
          'duration:',
          ws.getDuration(),
        );
        setIsPlaying(false);
        onFinishRef.current();
      });

      ws.load(audioUrl);
      wavesurferRef.current = ws;
      sharedWaveSurfer = ws;
    };

    setIsReady(false);
    setCurrentTime(0);
    setDuration(0);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setTimeout(initWaveSurfer, 10);
      });
    });

    return () => {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioUrl]);

  // Respond to external play/stop signals
  useEffect(() => {
    const ws = wavesurferRef.current;
    console.log('[Player] effect', { playing, isReady, hasWs: !!ws });
    if (!ws || !isReady) return;

    if (playing) {
      // Resume the AudioContext first (required for iOS Safari after unlock)
      const backend = ws.getMediaElement();
      if (backend && 'context' in backend) {
        const ctx = (backend as unknown as { context: AudioContext }).context;
        if (ctx?.state === 'suspended') ctx.resume();
      }
      ws.play()
        .then(() => {
          console.log('[Player] play succeeded');
        })
        .catch((e: Error) => {
          if (e.name === 'NotAllowedError') {
            console.warn('[Player] Autoplay blocked by browser — waiting for user gesture');
          } else {
            console.error('[Player] play failed', e);
          }
        });
    } else {
      ws.pause();
    }
  }, [playing, isReady]);

  // Sync volume and muted state
  useEffect(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.setVolume(muted ? 0 : volume);
    }
  }, [volume, muted]);

  const handlePlayPause = useCallback(() => {
    if (!wavesurferRef.current) return;
    wavesurferRef.current.playPause();
  }, []);

  return (
    <div className="absolute bottom-0 left-0 right-0 border-t border-border bg-background/95 backdrop-blur supports-backdrop-filter:bg-background/60 z-30">
      <div className="px-4 py-3 flex flex-col md:flex-row md:items-center gap-2 md:gap-4">
        {/* Waveform — full width row on mobile, inline on desktop */}
        <div className="min-w-0 min-h-[60px] md:min-h-[80px] md:flex-1 md:order-2">
          <div ref={waveformRef} className="w-full h-full min-h-[60px] md:min-h-[80px]" />
        </div>

        {/* Controls row */}
        <div className="flex items-center gap-3 md:contents">
          {/* Play/Pause */}
          <button
            onClick={handlePlayPause}
            disabled={!isReady}
            className="h-10 w-10 rounded-full bg-accent flex items-center justify-center shrink-0 disabled:opacity-50 md:order-1 shadow-lg"
          >
            {isPlaying ? (
              <Pause className="h-5 w-5 text-accent-foreground fill-accent-foreground" />
            ) : (
              <Play className="h-5 w-5 ml-0.5 text-accent-foreground fill-accent-foreground" />
            )}
          </button>

          {/* Time */}
          <div className="flex items-center gap-1 text-sm text-muted-foreground shrink-0 md:order-3">
            <span className="font-mono text-xs">{formatDuration(currentTime)}</span>
            <span className="text-xs">/</span>
            <span className="font-mono text-xs">{formatDuration(duration)}</span>
          </div>

          {/* Title */}
          {title && (
            <div className="text-sm font-medium truncate max-w-[200px] shrink-0 hidden lg:block md:order-4">
              {title}
            </div>
          )}

          {/* Loop */}
          <button
            onClick={() => setIsLooping(!isLooping)}
            className={`h-8 w-8 flex items-center justify-center rounded-sm shrink-0 hover:bg-muted md:order-5 ${
              isLooping ? 'text-foreground' : 'text-muted-foreground'
            }`}
          >
            <Repeat className="h-4 w-4" />
          </button>

          {/* Volume */}
          <div className="flex items-center gap-2 shrink-0 w-[140px] md:order-6 mr-3">
            <button
              onClick={() => setVolume(volume > 0 ? 0 : 0.75)}
              className="h-8 w-8 flex items-center justify-center hover:bg-muted rounded-sm"
            >
              {volume > 0 ? (
                <Volume2 className="h-4 w-4 text-muted-foreground" />
              ) : (
                <VolumeX className="h-4 w-4 text-muted-foreground" />
              )}
            </button>
            <input
              type="range"
              min={0}
              max={100}
              value={volume * 100}
              onChange={(e) => setVolume(Number(e.target.value) / 100)}
              className="flex-1 h-1 appearance-none bg-muted rounded-full accent-foreground cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-foreground"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
