import { useQuery } from '@tanstack/react-query';
import { HugeiconsIcon } from '@hugeicons/react';
import { PauseIcon, PlayIcon, RepeatIcon, VolumeHighIcon, VolumeMuteIcon, Cancel01Icon } from '@hugeicons/core-free-icons';
import { useEffect, useMemo, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { apiClient } from '@/lib/api/client';
import { formatAudioDuration } from '@/lib/utils/audio';
import { debug } from '@/lib/utils/debug';
import { usePlayerStore } from '@/stores/playerStore';
import { usePlatform } from '@/platform/PlatformContext';

export function AudioPlayer() {
  const platform = usePlatform();
  const {
    audioUrl,
    audioId,
    profileId,
    title,
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
  }, [profileChannels, channels, profileId]);

  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const loadingRef = useRef(false);
  const previousAudioIdRef = useRef<string | null>(null);
  const hasInitializedRef = useRef(false);
  const isUsingNativePlaybackRef = useRef(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize WaveSurfer (only when audioUrl exists and container is ready)
  useEffect(() => {
    // Don't initialize if no audioUrl or already initialized
    if (!audioUrl) {
      return;
    }

    if (wavesurferRef.current) {
      debug.log('WaveSurfer already initialized, skipping');
      return;
    }

    debug.log('Creating NEW WaveSurfer instance');

    // Wait for container to be properly rendered
    const initWaveSurfer = () => {
      const container = waveformRef.current;
      if (!container) {
        // Container not ready yet, retry
        setTimeout(initWaveSurfer, 50);
        return;
      }

      // Check if container has dimensions and is visible
      const rect = container.getBoundingClientRect();
      const style = window.getComputedStyle(container);
      const isVisible =
        rect.width > 0 &&
        rect.height > 0 &&
        style.display !== 'none' &&
        style.visibility !== 'hidden';

      if (!isVisible) {
        // Retry after a short delay
        setTimeout(initWaveSurfer, 50);
        return;
      }

      debug.log('Initializing WaveSurfer...', {
        container,
        width: rect.width,
        height: rect.height,
      });

      try {
        // Get computed CSS variable values
        const root = document.documentElement;
        const getCSSVar = (varName: string) => {
          const value = getComputedStyle(root).getPropertyValue(varName).trim();
          return value ? `hsl(${value})` : '';
        };

        const waveColor = getCSSVar('--muted');
        const progressColor = getCSSVar('--accent');
        const cursorColor = getCSSVar('--accent');

        const wavesurfer = WaveSurfer.create({
          container: container,
          waveColor: waveColor,
          progressColor: progressColor,
          cursorColor: cursorColor,
          barWidth: 2,
          barRadius: 2,
          height: 80,
          normalize: true,
          backend: 'WebAudio',
          interact: true, // Enable interaction (click to seek)
          mediaControls: false, // Don't show native controls
        });

        wavesurferRef.current = wavesurfer;
        debug.log('WaveSurfer created successfully');
      } catch (error) {
        debug.error('Failed to create WaveSurfer:', error);
        setError(
          `Failed to initialize waveform: ${error instanceof Error ? error.message : String(error)}`,
        );
        return;
      }

      const wavesurfer = wavesurferRef.current;
      if (!wavesurfer) return;

      // Update store when time changes
      wavesurfer.on('timeupdate', (time) => {
        setCurrentTime(time);
      });

      // Update store when duration is loaded
      wavesurfer.on('ready', async () => {
        const dur = wavesurfer.getDuration();
        setDuration(dur);
        loadingRef.current = false;
        setIsLoading(false);
        setError(null);
        debug.log('Audio ready, duration:', dur);
        debug.log('Waveform should be visible now');

        // Ensure volume is set
        const currentVolume = usePlayerStore.getState().volume;
        wavesurfer.setVolume(currentVolume);

        // Get the underlying audio element and ensure it's not muted
        // (unless we're using native playback, which will be set later)
        const mediaElement = wavesurfer.getMediaElement();
        if (mediaElement && !isUsingNativePlaybackRef.current) {
          mediaElement.volume = currentVolume;
          mediaElement.muted = false;
          debug.log('Audio element volume:', mediaElement.volume, 'muted:', mediaElement.muted);
        }

        // Auto-play when ready - check if we should use native playback
        // Get current values from the store and queries at runtime (not captured closure values)
        const currentAudioUrl = usePlayerStore.getState().audioUrl;
        const currentProfileId = usePlayerStore.getState().profileId;

        debug.log('Auto-play check - capturing runtime values...');

        // Fetch profile channels at runtime (not using captured value)
        let runtimeProfileChannels = null;
        let runtimeChannels = null;

        if (platform.metadata.isTauri && currentProfileId) {
          try {
            runtimeProfileChannels = await apiClient.getProfileChannels(currentProfileId);
            debug.log('Runtime profileChannels:', runtimeProfileChannels);

            if (runtimeProfileChannels && runtimeProfileChannels.channel_ids.length > 0) {
              runtimeChannels = await apiClient.listChannels();
              debug.log('Runtime channels:', runtimeChannels);
            }
          } catch (error) {
            debug.error('Failed to fetch runtime channel data:', error);
          }
        }

        debug.log('Auto-play check:', {
          isTauri: platform.metadata.isTauri,
          currentAudioUrl,
          currentProfileId,
          hasProfileChannels: !!runtimeProfileChannels,
          hasChannels: !!runtimeChannels,
        });

        if (
          platform.metadata.isTauri &&
          currentAudioUrl &&
          currentProfileId &&
          runtimeProfileChannels &&
          runtimeChannels
        ) {
          debug.log('Attempting native audio playback...');

          // Stop any existing native playback first
          if (isUsingNativePlaybackRef.current) {
            try {
              platform.audio.stopPlayback();
              debug.log('Stopped existing native playback before starting new one');
            } catch (error) {
              debug.error('Failed to stop existing playback:', error);
            }
          }

          try {
            // Collect all device IDs from assigned channels
            const assignedChannels = runtimeChannels.filter((ch: any) =>
              runtimeProfileChannels.channel_ids.includes(ch.id),
            );
            debug.log('Assigned channels for playback:', assignedChannels);

            // Check if any assigned channel has non-default devices
            const shouldUseNative = assignedChannels.some(
              (ch: any) => ch.device_ids.length > 0 && !ch.is_default,
            );
            debug.log('Should use native playback:', shouldUseNative);

            if (!shouldUseNative) {
              debug.log('No custom devices assigned, falling back to WaveSurfer');
              // Reset native playback flag and unmute WaveSurfer
              isUsingNativePlaybackRef.current = false;
              const mediaElement = wavesurfer.getMediaElement();
              if (mediaElement) {
                const currentVolume = usePlayerStore.getState().volume;
                mediaElement.volume = currentVolume;
                mediaElement.muted = false;
                debug.log(
                  'WaveSurfer unmuted for normal playback - volume:',
                  mediaElement.volume,
                  'muted:',
                  mediaElement.muted,
                );
              }
            } else {
              const deviceIds = assignedChannels.flatMap((ch: any) => ch.device_ids);
              debug.log('Device IDs to play to:', deviceIds);

              if (deviceIds.length > 0) {
                debug.log('Fetching audio data from:', currentAudioUrl);
                // Fetch audio data
                const response = await fetch(currentAudioUrl);
                const audioData = new Uint8Array(await response.arrayBuffer());
                debug.log('Audio data size:', audioData.length);

                // Play via native audio
                debug.log('Invoking play_audio_to_devices...');
                try {
                  await platform.audio.playToDevices(audioData, deviceIds);
                  debug.log('play_audio_to_devices completed successfully');

                  // Mark that we're using native playback
                  isUsingNativePlaybackRef.current = true;

                  // Mute WaveSurfer's audio element to prevent UI audio output
                  // Keep WaveSurfer running for visualization
                  const mediaElement = wavesurfer.getMediaElement();
                  if (mediaElement) {
                    mediaElement.volume = 0;
                    mediaElement.muted = true;
                    debug.log(
                      'WaveSurfer muted for native playback - volume:',
                      mediaElement.volume,
                      'muted:',
                      mediaElement.muted,
                    );
                  }

                  // Start WaveSurfer playback for visualization (muted)
                  wavesurfer.play().catch((error) => {
                    debug.error('Failed to start WaveSurfer visualization:', error);
                  });

                  setIsPlaying(true);
                  debug.log('Auto-playing via native audio routing - SUCCESS');
                  return;
                } catch (invokeError) {
                  debug.error('play_audio_to_devices invoke failed:', invokeError);
                  throw invokeError;
                }
              } else {
                debug.log('No device IDs found, falling back to WaveSurfer');
              }
            }
          } catch (error) {
            debug.error(
              'Native playback failed during auto-play, falling back to WaveSurfer:',
              error,
            );
            // Reset native playback flag and unmute WaveSurfer
            isUsingNativePlaybackRef.current = false;
            const mediaElement = wavesurfer.getMediaElement();
            if (mediaElement) {
              const currentVolume = usePlayerStore.getState().volume;
              mediaElement.volume = currentVolume;
              mediaElement.muted = false;
              debug.log(
                'WaveSurfer unmuted after native playback failure - volume:',
                mediaElement.volume,
                'muted:',
                mediaElement.muted,
              );
            }
            // Fall through to WaveSurfer playback
          }
        } else {
          debug.log('Not using native playback, using WaveSurfer');
          // Reset native playback flag and unmute WaveSurfer
          isUsingNativePlaybackRef.current = false;
          const mediaElement = wavesurfer.getMediaElement();
          if (mediaElement) {
            const currentVolume = usePlayerStore.getState().volume;
            mediaElement.volume = currentVolume;
            mediaElement.muted = false;
            debug.log(
              'WaveSurfer unmuted for normal playback - volume:',
              mediaElement.volume,
              'muted:',
              mediaElement.muted,
            );
          }
        }

        // Only auto-play if shouldAutoPlay flag is set (user explicitly clicked to play)
        const shouldAutoPlayNow = usePlayerStore.getState().shouldAutoPlay;
        if (shouldAutoPlayNow) {
          // Clear the flag first
          usePlayerStore.getState().clearAutoPlayFlag();
          
          // Use a small delay to ensure audio element is fully ready
          setTimeout(() => {
            wavesurfer.play().catch((error) => {
              debug.error('Failed to autoplay:', error);
              // Don't show error for autoplay failures (browser restrictions)
            });
          }, 100);
        } else {
          debug.log('Skipping auto-play - shouldAutoPlay is false');
        }
      });

      // Handle play/pause
      wavesurfer.on('play', () => {
        setIsPlaying(true);
        // Ensure audio element volume is set correctly
        const mediaElement = wavesurfer.getMediaElement();
        if (mediaElement) {
          // Double-check: if using native playback, keep WaveSurfer muted
          // Otherwise, ensure it's unmuted
          if (isUsingNativePlaybackRef.current) {
            mediaElement.volume = 0;
            mediaElement.muted = true;
            debug.log('Playing (native mode) - WaveSurfer muted for visualization only');
          } else {
            // Ensure WaveSurfer is unmuted for normal playback
            const currentVolume = usePlayerStore.getState().volume;
            mediaElement.volume = currentVolume;
            mediaElement.muted = false;
            debug.log(
              'Playing (normal mode) - volume:',
              mediaElement.volume,
              'muted:',
              mediaElement.muted,
            );
          }
        }
      });
      wavesurfer.on('pause', () => setIsPlaying(false));
      wavesurfer.on('finish', () => {
        // Check loop state from store
        const loop = usePlayerStore.getState().isLooping;
        if (loop) {
          wavesurfer.seekTo(0);
          wavesurfer.play();
        } else {
          setIsPlaying(false);
          // Trigger finish callback if set
          const onFinish = usePlayerStore.getState().onFinish;
          if (onFinish) {
            onFinish();
          }
        }
      });

      // Handle errors
      wavesurfer.on('error', (error) => {
        debug.error('WaveSurfer error:', error);
        setIsLoading(false);
        setError(`Audio error: ${error instanceof Error ? error.message : String(error)}`);
      });

      // Handle loading
      wavesurfer.on('loading', (percent) => {
        setIsLoading(true);
        if (percent === 100) {
          setIsLoading(false);
        }
      });

      // Load audio immediately if audioUrl is already set
      if (audioUrl) {
        debug.log('WaveSurfer ready, loading audio:', audioUrl);
        loadingRef.current = true;
        setIsLoading(true);
        // Stop any current playback before loading new audio
        if (wavesurfer.isPlaying()) {
          wavesurfer.pause();
        }
        wavesurfer
          .load(audioUrl)
          .then(() => {
            debug.log('Audio loaded into WaveSurfer');
            loadingRef.current = false;
          })
          .catch((error) => {
            debug.error('Failed to load audio into WaveSurfer:', error);
            loadingRef.current = false;
            setIsLoading(false);
            setError(
              `Failed to load audio: ${error instanceof Error ? error.message : String(error)}`,
            );
          });
      }
    };

    // Use double requestAnimationFrame to ensure DOM is fully rendered
    let rafId1: number;
    let rafId2: number;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    rafId1 = requestAnimationFrame(() => {
      rafId2 = requestAnimationFrame(() => {
        // Add a small delay to ensure container is fully laid out
        timeoutId = setTimeout(() => {
          initWaveSurfer();
        }, 10);
      });
    });

    return () => {
      debug.log('Cleaning up WaveSurfer initialization effect');
      if (rafId1) cancelAnimationFrame(rafId1);
      if (rafId2) cancelAnimationFrame(rafId2);
      if (timeoutId) clearTimeout(timeoutId);
      if (wavesurferRef.current) {
        debug.log('Destroying WaveSurfer instance');
        try {
          const mediaElement = wavesurferRef.current.getMediaElement();
          if (mediaElement) {
            mediaElement.pause();
            mediaElement.src = '';
          }
          wavesurferRef.current.destroy();
        } catch (error) {
          debug.error('Error destroying WaveSurfer:', error);
        }
        wavesurferRef.current = null;
      }
    };
  }, [audioUrl, setIsPlaying, setCurrentTime, setDuration]);

  // Load audio when URL changes (only if WaveSurfer is already initialized)
  useEffect(() => {
    const wavesurfer = wavesurferRef.current;

    if (!audioUrl || !wavesurfer) {
      // Reset state when no audio or WaveSurfer not ready
      if (!audioUrl && wavesurfer) {
        wavesurfer.pause();
        wavesurfer.seekTo(0);
        loadingRef.current = false;
        setIsLoading(false);
        setDuration(0);
        setCurrentTime(0);
        setError(null);
        // Reset native playback flag
        isUsingNativePlaybackRef.current = false;
      }
      return;
    }

    // Stop native playback if it was active
    if (isUsingNativePlaybackRef.current && platform.metadata.isTauri) {
      try {
        platform.audio.stopPlayback();
        debug.log('Stopped native audio playback');
      } catch (error) {
        debug.error('Failed to stop native playback:', error);
      }
    }

    // Reset native playback flag when loading new audio
    // Also unmute WaveSurfer if it was muted
    if (isUsingNativePlaybackRef.current) {
      const mediaElement = wavesurfer.getMediaElement();
      if (mediaElement) {
        mediaElement.muted = false;
        mediaElement.volume = usePlayerStore.getState().volume;
      }
    }
    isUsingNativePlaybackRef.current = false;

    // CRITICAL: Force stop any current playback and cancel any pending loads
    // This must happen BEFORE any early returns
    debug.log('Audio URL changed to:', audioUrl);

    // COMPLETELY stop and destroy the current audio
    try {
      // First pause if playing
      if (wavesurfer.isPlaying()) {
        debug.log('Pausing current playback');
        wavesurfer.pause();
      }

      // Stop the media element explicitly
      const mediaElement = wavesurfer.getMediaElement();
      if (mediaElement) {
        debug.log('Stopping media element');
        mediaElement.pause();
        mediaElement.currentTime = 0;
        mediaElement.src = '';
      }

      // Use empty() to completely destroy the waveform and media element
      debug.log('Calling wavesurfer.empty() to destroy audio');
      wavesurfer.empty();
    } catch (error) {
      debug.error('Error stopping previous audio:', error);
      // Continue anyway to load new audio
    }

    // Reset loading state to allow new load (cancel any pending loads)
    loadingRef.current = false;

    // Now start the new load
    loadingRef.current = true;
    setIsLoading(true);
    setError(null);
    setCurrentTime(0);
    setDuration(0);

    // Load new audio
    debug.log('Starting new audio load for:', audioUrl);
    wavesurfer
      .load(audioUrl)
      .then(() => {
        debug.log('Audio load promise resolved');
        // Don't set loading to false here - wait for 'ready' event
      })
      .catch((error) => {
        debug.error('Failed to load audio:', error);
        debug.error('Audio URL:', audioUrl);
        loadingRef.current = false;
        setIsLoading(false);
        setError(`Failed to load audio: ${error instanceof Error ? error.message : String(error)}`);
      });
  }, [audioUrl, setCurrentTime, setDuration]);

  // Sync play/pause state (only when user clicks play/pause button, not auto-sync)
  // This effect is kept for external state changes but should be minimal
  useEffect(() => {
    if (!wavesurferRef.current || duration === 0) return;

    if (isPlaying && wavesurferRef.current.isPlaying() === false) {
      // Only auto-play if audio is ready
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
      // Also ensure the underlying audio element volume is set
      const mediaElement = wavesurferRef.current.getMediaElement();
      if (mediaElement) {
        // If using native playback, keep WaveSurfer muted regardless of volume setting
        if (isUsingNativePlaybackRef.current) {
          mediaElement.volume = 0;
          mediaElement.muted = true;
          debug.log('Volume sync: Using native playback, keeping WaveSurfer muted');
        } else {
          mediaElement.volume = volume;
          mediaElement.muted = volume === 0;
          debug.log('Volume synced:', volume, 'muted:', mediaElement.muted);
        }
      }
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

    // Reset to beginning and play
    debug.log('Restarting current audio from beginning');
    wavesurfer.seekTo(0);
    wavesurfer.play().catch((error) => {
      debug.error('Failed to play after restart:', error);
      setIsPlaying(false);
      setError(`Playback error: ${error instanceof Error ? error.message : String(error)}`);
    });

    // Clear the restart flag
    clearRestartFlag();
  }, [shouldRestart, duration, setIsPlaying, clearRestartFlag]);

  // Handle shouldAutoPlay flag - for story mode auto-advance
  const shouldAutoPlay = usePlayerStore((state) => state.shouldAutoPlay);
  const clearAutoPlayFlag = usePlayerStore((state) => state.clearAutoPlayFlag);
  
  useEffect(() => {
    const wavesurfer = wavesurferRef.current;
    if (!wavesurfer || !shouldAutoPlay || duration === 0) {
      return;
    }

    // Auto-play the newly loaded audio
    debug.log('Auto-playing next track in story mode');
    wavesurfer.seekTo(0);
    wavesurfer.play().catch((error) => {
      debug.error('Failed to auto-play:', error);
      setIsPlaying(false);
      setError(`Playback error: ${error instanceof Error ? error.message : String(error)}`);
    });

    // Clear the auto-play flag
    clearAutoPlayFlag();
  }, [shouldAutoPlay, duration, setIsPlaying, clearAutoPlayFlag]);

  // Handle loop - WaveSurfer handles this via the 'finish' event

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
          const mediaElement = wavesurferRef.current.getMediaElement();
          if (mediaElement) {
            mediaElement.volume = 0;
            mediaElement.muted = true;
          }

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
        const mediaElement = wavesurferRef.current.getMediaElement();
        if (mediaElement) {
          mediaElement.muted = false;
          mediaElement.volume = volume;
        }
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
            className="shrink-0"
            title={duration === 0 && !isLoading ? 'Audio not loaded' : ''}
          >
            {isPlaying ? <HugeiconsIcon icon={PauseIcon} size={20} className="h-5 w-5" /> : <HugeiconsIcon icon={PlayIcon} size={20} className="h-5 w-5" />}
          </Button>

          {/* Waveform */}
          <div className="flex-1 min-w-0 flex flex-col gap-1">
            <div ref={waveformRef} className="w-full min-h-[80px]" />
            {duration > 0 && (
              <Slider
                value={duration > 0 ? [(currentTime / duration) * 100] : [0]}
                onValueChange={handleSeek}
                max={100}
                step={0.1}
                className="w-full"
              />
            )}
            {isLoading && (
              <div className="text-xs text-muted-foreground text-center py-2">Loading audio...</div>
            )}
            {error && <div className="text-xs text-destructive text-center py-2">{error}</div>}
          </div>

          {/* Time Display */}
          <div className="flex items-center gap-2 text-sm text-muted-foreground shrink-0 min-w-[100px]">
            <span className="font-mono">{formatAudioDuration(currentTime)}</span>
            <span>/</span>
            <span className="font-mono">{formatAudioDuration(duration)}</span>
          </div>

          {/* Title */}
          {title && (
            <div className="text-sm font-medium truncate max-w-[200px] shrink-0">{title}</div>
          )}

          {/* Loop Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleLoop}
            className={isLooping ? 'text-primary' : ''}
            title="Toggle loop"
          >
            <HugeiconsIcon icon={RepeatIcon} size={16} className="h-4 w-4" />
          </Button>

          {/* Volume Control */}
          <div className="flex items-center gap-2 shrink-0 w-[120px]">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setVolume(volume > 0 ? 0 : 1)}
              className="h-8 w-8"
            >
              {volume > 0 ? <HugeiconsIcon icon={VolumeHighIcon} size={16} className="h-4 w-4" /> : <HugeiconsIcon icon={VolumeMuteIcon} size={16} className="h-4 w-4" />}
            </Button>
            <Slider
              value={[volume * 100]}
              onValueChange={handleVolumeChange}
              max={100}
              step={1}
              className="flex-1"
            />
          </div>

          {/* Close Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={handleClose}
            className="shrink-0"
            title="Close player"
          >
            <HugeiconsIcon icon={Cancel01Icon} size={20} className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
  );
}
