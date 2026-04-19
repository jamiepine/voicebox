import { useQuery } from '@tanstack/react-query';
import { Pause, Play, Repeat, Volume2, VolumeX, X } from 'lucide-react';
import { useEffect, useId, useMemo, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { apiClient } from '@/lib/api/client';
import { formatAudioDuration } from '@/lib/utils/audio';
import { debug } from '@/lib/utils/debug';
import { usePlatform } from '@/platform/PlatformContext';
import { usePlayerStore } from '@/stores/playerStore';

export function AudioPlayer() {
  const platform = usePlatform();
  const volumeLabelId = useId();
  const {
    audioUrl,
    audioId,
    profileId,
    isPlaying,
    currentTime,
    duration,
    volume,
    isLooping,
    shouldRestart,
    setIsPlaying,
    setCurrentTime,
    setDuration,
    setVolume,
    toggleLoop,
    clearRestartFlag,
    reset,
  } = usePlayerStore();

  // Check if profile has assigned channels (for native audio routing)
  const { data: profileChannels } = useQuery({
    queryKey: ['profile-channels', profileId],
    queryFn: () => {
      if (!profileId) return { channel_ids: [] };
      return apiClient.getProfileChannels(profileId);
    },
    enabled: !!profileId && platform.metadata.isTauri,
  });

  const { data: channels } = useQuery({
    queryKey: ['channels'],
    queryFn: () => apiClient.listChannels(),
    enabled: !!profileChannels && profileChannels.channel_ids.length > 0,
  });

  // Determine if we should use native playback
  const useNativePlayback = useMemo(() => {
    if (!platform.metadata.isTauri || !profileChannels || !channels) {
      return false;
    }

    const assignedChannels = channels.filter((ch) => profileChannels.channel_ids.includes(ch.id));

    // Use native playback if any assigned channel has non-default devices
    const shouldUseNative = assignedChannels.some(
      (ch) => ch.device_ids.length > 0 && !ch.is_default,
    );

    return shouldUseNative;
  }, [profileChannels, channels, platform.metadata.isTauri]);

  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const loadingRef = useRef(false);
  const previousAudioIdRef = useRef<string | null>(null);
  const hasInitializedRef = useRef(false);
  const isUsingNativePlaybackRef = useRef(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [wsReady, setWsReady] = useState(false);

  // Create WaveSurfer once when the player becomes visible (audioUrl is set).
  // This instance is reused for all subsequent audio loads - never destroyed until unmount.
  useEffect(() => {
    if (!audioUrl) return;
    if (wavesurferRef.current) return; // already created

    const initWaveSurfer = () => {
      const container = waveformRef.current;
      if (!container) {
        setTimeout(initWaveSurfer, 50);
        return;
      }

      const rect = container.getBoundingClientRect();
      const style = window.getComputedStyle(container);
      const isVisible =
        rect.width > 0 &&
        rect.height > 0 &&
        style.display !== 'none' &&
        style.visibility !== 'hidden';

      if (!isVisible) {
        setTimeout(initWaveSurfer, 50);
        return;
      }

      debug.log('Creating WaveSurfer instance', {
        width: rect.width,
        height: rect.height,
      });

      try {
        const root = document.documentElement;
        const getCSSVar = (varName: string) => {
          const value = getComputedStyle(root).getPropertyValue(varName).trim();
          return value ? `hsl(${value})` : '';
        };

        const wavesurfer = WaveSurfer.create({
          container,
          waveColor: getCSSVar('--muted'),
          progressColor: getCSSVar('--accent'),
          cursorColor: getCSSVar('--accent'),
          cursorWidth: 3,
          barWidth: 2,
          barRadius: 2,
          height: 80,
          normalize: true,
          interact: true,
          dragToSeek: { debounceTime: 0 },
          mediaControls: false,
          backend: 'WebAudio',
        });

        // Wire up event handlers (these persist for the lifetime of the instance)
        wavesurfer.on('timeupdate', (time) => {
          const dur = usePlayerStore.getState().duration;
          if (dur > 0 && time >= dur) {
            setCurrentTime(dur);
            const loop = usePlayerStore.getState().isLooping;
            if (loop) {
              wavesurfer.seekTo(0);
              wavesurfer.play().catch((err) => debug.error('Loop play failed:', err));
            } else {
              wavesurfer.pause();
              setIsPlaying(false);
            }
            return;
          }
          setCurrentTime(time);
        });

        wavesurfer.on('ready', () => {
          const dur = wavesurfer.getDuration();
          setDuration(dur);
          loadingRef.current = false;
          setIsLoading(false);
          setError(null);
          debug.log('Audio ready, duration:', dur);

          wavesurfer.setVolume(usePlayerStore.getState().volume);
          wavesurfer.setMuted(false);

          // Auto-play if the flag is set (story mode advance or explicit play)
          const shouldAutoPlayNow = usePlayerStore.getState().shouldAutoPlay;
          if (shouldAutoPlayNow) {
            usePlayerStore.getState().clearAutoPlayFlag();
            wavesurfer.play().catch((err) => {
              debug.error('Failed to autoplay:', err);
            });
          } else {
            debug.log('Skipping auto-play - shouldAutoPlay is false');
          }
        });

        wavesurfer.on('play', () => setIsPlaying(true));
        wavesurfer.on('pause', () => {
          setIsPlaying(false);
          setCurrentTime(wavesurfer.getCurrentTime());
        });

        wavesurfer.on('seeking', (time) => setCurrentTime(time));

        // Mute audio during drag-to-seek to prevent popping from the WebAudio
        // backend's hard stop/start cycle on each seek. Unmute with a short
        // fade-in when the drag ends.
        const seekMedia = wavesurfer.getMediaElement() as any;
        const seekGain: GainNode | null = seekMedia?.getGainNode?.() ?? null;
        if (seekGain) {
          const ctx = seekGain.context as AudioContext;
          wavesurfer.on('dragstart', () => {
            seekGain.gain.cancelScheduledValues(ctx.currentTime);
            seekGain.gain.setTargetAtTime(0, ctx.currentTime, 0.002);
          });
          wavesurfer.on('dragend', () => {
            seekGain.gain.cancelScheduledValues(ctx.currentTime);
            seekGain.gain.setTargetAtTime(1, ctx.currentTime, 0.01);
          });
        }
        wavesurfer.on('finish', () => {
          const loop = usePlayerStore.getState().isLooping;
          if (loop) {
            wavesurfer.seekTo(0);
            wavesurfer.play().catch((err) => debug.error('Loop play failed:', err));
          } else {
            setIsPlaying(false);
            const onFinish = usePlayerStore.getState().onFinish;
            if (onFinish) onFinish();
          }
        });

        wavesurfer.on('error', (err) => {
          debug.error('WaveSurfer error:', err);
          setIsLoading(false);
          setError(`Audio error: ${err instanceof Error ? err.message : String(err)}`);
        });

        wavesurfer.on('loading', (percent) => {
          setIsLoading(true);
          if (percent === 100) setIsLoading(false);
        });

        wavesurferRef.current = wavesurfer;
        setWsReady(true);
        debug.log('WaveSurfer created successfully');
      } catch (err) {
        debug.error('Failed to create WaveSurfer:', err);
        setError(
          `Failed to initialize waveform: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    };

    let rafId: number;
    rafId = requestAnimationFrame(() => {
      initWaveSurfer();
    });

    return () => {
      cancelAnimationFrame(rafId);
    };
    // Only run on mount-like conditions. audioUrl is here so we create the instance
    // when the player first appears, but we guard against re-creation above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioUrl, setIsPlaying, setDuration, setCurrentTime]);

  // Destroy WaveSurfer only on unmount
  useEffect(() => {
    return () => {
      if (wavesurferRef.current) {
        debug.log('Destroying WaveSurfer instance (unmount)');
        try {
          wavesurferRef.current.destroy();
        } catch (err) {
          debug.error('Error destroying WaveSurfer:', err);
        }
        wavesurferRef.current = null;
        setWsReady(false);
      }
    };
  }, []);

  // Load audio when URL changes (reuses the existing WaveSurfer instance)
  useEffect(() => {
    const wavesurfer = wavesurferRef.current;
    if (!wavesurfer || !wsReady) return;

    if (!audioUrl) {
      // No audio - pause and reset
      wavesurfer.pause();
      wavesurfer.seekTo(0);
      loadingRef.current = false;
      setIsLoading(false);
      setDuration(0);
      setCurrentTime(0);
      setError(null);
      isUsingNativePlaybackRef.current = false;
      return;
    }

    // Reset native playback state
    isUsingNativePlaybackRef.current = false;
    wavesurfer.setMuted(false);
    wavesurfer.setVolume(usePlayerStore.getState().volume);

    // Stop current playback and reset position before loading new audio.
    // With the WebAudio backend, pause() accumulates playedDuration internally.
    // seekTo(0) resets it so the new track starts from the beginning.
    debug.log('Loading new audio URL:', audioUrl);
    try {
      if (wavesurfer.isPlaying()) {
        wavesurfer.pause();
      }
      wavesurfer.seekTo(0);
    } catch (err) {
      debug.error('Error resetting before load:', err);
    }

    loadingRef.current = true;
    setIsLoading(true);
    setError(null);
    setCurrentTime(0);
    setDuration(0);

    wavesurfer
      .load(audioUrl)
      .then(() => {
        debug.log('Audio loaded into WaveSurfer');
        loadingRef.current = false;
      })
      .catch((err) => {
        debug.error('Failed to load audio:', err);
        loadingRef.current = false;
        setIsLoading(false);
        setError(`Failed to load audio: ${err instanceof Error ? err.message : String(err)}`);
      });
  }, [audioUrl, wsReady, setCurrentTime, setDuration]);

  // Sync play/pause state (only when user clicks play/pause button, not auto-sync)
  // This effect is kept for external state changes but should be minimal
  useEffect(() => {
    if (!wavesurferRef.current || duration === 0) return;

    if (isPlaying && wavesurferRef.current.isPlaying() === false) {
      wavesurferRef.current.play().catch((error) => {
        debug.error('Failed to play:', error);
        setIsPlaying(false);
        setError(`Playback error: ${error instanceof Error ? error.message : String(error)}`);
      });
    } else if (!isPlaying && wavesurferRef.current.isPlaying()) {
      wavesurferRef.current.pause();
    }
  }, [isPlaying, setIsPlaying, duration]);

  // Sync volume
  useEffect(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.setVolume(volume);
    }
  }, [volume]);

  // Mark as initialized when audio is ready, reset when audioId changes
  useEffect(() => {
    if (duration > 0 && audioId) {
      hasInitializedRef.current = true;
    }
    // Reset initialization flag when audioId changes to a new audio
    if (audioId !== previousAudioIdRef.current && previousAudioIdRef.current !== null) {
      hasInitializedRef.current = false;
    }
    if (audioId !== null) {
      previousAudioIdRef.current = audioId;
    }
  }, [duration, audioId]);

  // Handle restart flag - when history item is clicked again, restart from beginning
  useEffect(() => {
    const wavesurfer = wavesurferRef.current;
    if (!wavesurfer || !shouldRestart || duration === 0) {
      return;
    }

    debug.log('Restarting current audio from beginning');
    wavesurfer.seekTo(0);
    wavesurfer.play().catch((error) => {
      debug.error('Failed to play after restart:', error);
      setIsPlaying(false);
      setError(`Playback error: ${error instanceof Error ? error.message : String(error)}`);
    });

    clearRestartFlag();
  }, [shouldRestart, duration, setIsPlaying, clearRestartFlag]);

  // Auto-play is handled exclusively in the WaveSurfer 'ready' event handler.
  // A separate effect here would race with the ready event since the WebAudio
  // backend needs to fully decode the audio before play() works correctly.

  // Spacebar to play/pause (capture phase so it fires before focused elements)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      // Ignore if user is typing in an input/textarea
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.isContentEditable) {
        return;
      }
      if (audioUrl && duration > 0 && wavesurferRef.current) {
        e.preventDefault();
        e.stopPropagation();
        if (wavesurferRef.current.isPlaying()) {
          wavesurferRef.current.pause();
        } else {
          wavesurferRef.current.play().catch((err) => debug.error('Spacebar play failed:', err));
        }
      }
    };
    document.addEventListener('keydown', onKeyDown, true);
    return () => document.removeEventListener('keydown', onKeyDown, true);
  }, [audioUrl, duration]);

  const handlePlayPause = async () => {
    // Standard WaveSurfer playback (works for both normal and native playback modes)
    // When using native playback, WaveSurfer is muted but still controls visualization
    if (!wavesurferRef.current) {
      debug.error('WaveSurfer not initialized');
      return;
    }

    // Check if audio is loaded
    if (duration === 0 && !isLoading) {
      debug.error('Audio not loaded yet');
      setError('Audio not loaded. Please wait...');
      return;
    }

    // If using native playback
    if (useNativePlayback && audioUrl && profileChannels && channels) {
      if (isPlaying) {
        // Pause: stop native playback and pause WaveSurfer visualization
        try {
          platform.audio.stopPlayback();
          debug.log('Stopped native audio playback');
        } catch (error) {
          debug.error('Failed to stop native playback:', error);
        }
        wavesurferRef.current.pause();
        return;
      }

      // Play: trigger native playback
      try {
        // Stop any existing native playback first
        try {
          platform.audio.stopPlayback();
        } catch (_error) {
          // Ignore errors when stopping (might not be playing)
          debug.log('No existing playback to stop');
        }

        // Collect all device IDs from assigned channels
        const assignedChannels = channels.filter((ch) =>
          profileChannels.channel_ids.includes(ch.id),
        );
        const deviceIds = assignedChannels.flatMap((ch) => ch.device_ids);

        if (deviceIds.length > 0) {
          // Fetch audio data
          const response = await fetch(audioUrl);
          const audioData = new Uint8Array(await response.arrayBuffer());

          // Play via native audio
          await platform.audio.playToDevices(audioData, deviceIds);

          // Mark that we're using native playback
          isUsingNativePlaybackRef.current = true;

          // Mute WaveSurfer and start it for visualization
          wavesurferRef.current.setVolume(0);
          wavesurferRef.current.setMuted(true);

          // Start WaveSurfer for visualization (muted)
          wavesurferRef.current.play().catch((error) => {
            debug.error('Failed to start WaveSurfer visualization:', error);
            setIsPlaying(false);
            setError(`Playback error: ${error instanceof Error ? error.message : String(error)}`);
          });

          return;
        }
      } catch (error) {
        debug.error('Native playback failed, falling back to WaveSurfer:', error);
        // Fall through to WaveSurfer playback
        isUsingNativePlaybackRef.current = false;
      }
    }

    // Standard WaveSurfer playback (or fallback from native playback failure)
    if (wavesurferRef.current.isPlaying()) {
      wavesurferRef.current.pause();
    } else {
      // Ensure WaveSurfer is not muted if not using native playback
      if (!isUsingNativePlaybackRef.current) {
        wavesurferRef.current.setMuted(false);
        wavesurferRef.current.setVolume(volume);
      }

      wavesurferRef.current.play().catch((error) => {
        debug.error('Failed to play:', error);
        setIsPlaying(false);
        setError(`Playback error: ${error instanceof Error ? error.message : String(error)}`);
      });
    }
  };

  const handleSeek = (value: number[]) => {
    if (!wavesurferRef.current || duration === 0) return;
    const progress = value[0] / 100;
    wavesurferRef.current.seekTo(progress);
  };

  const handleVolumeChange = (value: number[]) => {
    setVolume(value[0] / 100);
  };

  const handleClose = () => {
    // Stop any native playback
    if (isUsingNativePlaybackRef.current && platform.metadata.isTauri) {
      try {
        platform.audio.stopPlayback();
      } catch (error) {
        debug.error('Failed to stop native playback:', error);
      }
    }
    // Stop WaveSurfer
    if (wavesurferRef.current) {
      wavesurferRef.current.pause();
      wavesurferRef.current.seekTo(0);
    }
    // Reset player state
    reset();
  };

  // Don't render if no audio
  if (!audioUrl) {
    return null;
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 border-t bg-background/95 backdrop-blur supports-backdrop-filter:bg-background/60 z-50">
      <div className="container mx-auto px-4 py-3 max-w-7xl">
        <div className="flex items-center gap-4">
          {/* Play/Pause Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={handlePlayPause}
            disabled={isLoading || duration === 0}
            className={`shrink-0 -mt-2 ${isPlaying ? 'bg-accent text-accent-foreground' : ''}`}
            title={duration === 0 && !isLoading ? 'Audio not loaded' : ''}
            aria-label={
              duration === 0 && !isLoading ? 'Audio not loaded' : isPlaying ? 'Pause' : 'Play'
            }
          >
            {isPlaying ? (
              <Pause className="h-5 w-5 fill-current" />
            ) : (
              <Play className="h-5 w-5 fill-current" />
            )}
          </Button>

          {/* Waveform */}
          <div className="flex-1 min-w-0 flex flex-col gap-1">
            <div ref={waveformRef} className="w-full min-h-[80px] select-none" />
            <Slider
              value={duration > 0 ? [(currentTime / duration) * 100] : [0]}
              onValueChange={handleSeek}
              max={100}
              step={0.1}
              className="w-full"
              aria-label="Playback position"
              aria-valuetext={`${formatAudioDuration(currentTime)} of ${formatAudioDuration(duration)}`}
            />

            {error && <div className="text-xs text-destructive text-center py-2">{error}</div>}
          </div>

          {/* Time Display */}
          <div className="flex items-center gap-2 text-sm text-muted-foreground shrink-0 min-w-[100px]">
            <span className="font-mono">{formatAudioDuration(currentTime)}</span>
            <span>/</span>
            <span className="font-mono">{formatAudioDuration(duration)}</span>
          </div>

          {/* Loop Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleLoop}
            className={isLooping ? 'bg-accent text-accent-foreground' : ''}
            title="Toggle loop"
            aria-label={isLooping ? 'Stop looping' : 'Loop'}
          >
            <Repeat className="h-4 w-4" />
          </Button>

          {/* Volume Control */}
          <div
            className="flex items-center gap-2 shrink-0 w-[120px]"
            role="group"
            aria-label="Volume"
          >
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setVolume(volume > 0 ? 0 : 1)}
              className="h-8 w-8"
              aria-label={volume > 0 ? 'Mute' : 'Unmute'}
            >
              {volume > 0 ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
            </Button>
            <span id={volumeLabelId} className="sr-only">
              Volume level, {Math.round(volume * 100)}%
            </span>
            <Slider
              value={[volume * 100]}
              onValueChange={handleVolumeChange}
              max={100}
              step={1}
              className="flex-1"
              aria-labelledby={volumeLabelId}
              aria-valuetext={`${Math.round(volume * 100)}%`}
            />
          </div>

          {/* Close Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={handleClose}
            className="shrink-0"
            title="Close player"
            aria-label="Close player"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
  );
}
