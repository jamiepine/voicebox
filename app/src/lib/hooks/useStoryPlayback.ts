import { useCallback, useEffect, useRef } from 'react';
import { apiClient } from '@/lib/api/client';
import type { StoryItemDetail } from '@/lib/api/types';
import { useStoryStore } from '@/stores/storyStore';

interface ActiveSource {
  source: AudioBufferSourceNode;
  itemId: string;
  generationId: string;
  startTimeMs: number;
  endTimeMs: number;
}

/**
 * Hook for managing timecode-based story playback using Web Audio API.
 * Supports multiple simultaneous audio sources for overlapping clips on different tracks.
 * Uses AudioContext for sample-accurate timing synchronization.
 */
export function useStoryPlayback(items: StoryItemDetail[] | undefined) {
  const isPlaying = useStoryStore((state) => state.isPlaying);
  const playbackItems = useStoryStore((state) => state.playbackItems);
  const playbackStartContextTime = useStoryStore((state) => state.playbackStartContextTime);
  const playbackStartStoryTime = useStoryStore((state) => state.playbackStartStoryTime);
  const setPlaybackTiming = useStoryStore((state) => state.setPlaybackTiming);

  // AudioContext instance (created once)
  const audioContextRef = useRef<AudioContext | null>(null);
  // Master gain for volume control
  const masterGainRef = useRef<GainNode | null>(null);
  // Preloaded AudioBuffers by generation_id (audio file is shared between split clips)
  const audioBuffersRef = useRef<Map<string, AudioBuffer>>(new Map());
  // Currently playing AudioBufferSourceNodes by item.id (unique per clip)
  const activeSourcesRef = useRef<Map<string, ActiveSource>>(new Map());
  // Animation frame for syncing visual playhead
  const animationFrameRef = useRef<number | null>(null);

  // Get or create AudioContext and audio graph
  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
      console.log(
        '[StoryPlayback] Created AudioContext, sample rate:',
        audioContextRef.current.sampleRate,
      );

      // Create master gain node for volume control
      masterGainRef.current = audioContextRef.current.createGain();
      masterGainRef.current.gain.value = 1;
      masterGainRef.current.connect(audioContextRef.current.destination);
    }
    // Resume context if suspended (browser autoplay policy)
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume().catch(() => {
        // Ignore resume errors
      });
    }
    return audioContextRef.current;
  }, []);

  // Stop a source by item id
  const stopSource = useCallback((itemId: string) => {
    const activeSource = activeSourcesRef.current.get(itemId);
    if (activeSource) {
      try {
        activeSource.source.stop();
      } catch {
        // Source may have already stopped
      }
      activeSourcesRef.current.delete(itemId);
    }
  }, []);

  // Preload audio files as AudioBuffers
  useEffect(() => {
    if (!items || items.length === 0) {
      // Clear preloaded buffers when no items
      audioBuffersRef.current.clear();
      return;
    }

    const currentIds = new Set(items.map((item) => item.generation_id));
    const audioContext = getAudioContext();

    // Remove buffers for items that no longer exist
    for (const [id] of audioBuffersRef.current) {
      if (!currentIds.has(id)) {
        audioBuffersRef.current.delete(id);
      }
    }

    // Preload audio for new items
    const preloadPromises: Promise<void>[] = [];
    for (const item of items) {
      if (!audioBuffersRef.current.has(item.generation_id)) {
        const audioUrl = apiClient.getAudioUrl(item.generation_id);
        console.log('[StoryPlayback] Preloading audio buffer:', item.generation_id);

        const preloadPromise = fetch(audioUrl)
          .then((response) => response.arrayBuffer())
          .then((arrayBuffer) => audioContext.decodeAudioData(arrayBuffer))
          .then((audioBuffer) => {
            audioBuffersRef.current.set(item.generation_id, audioBuffer);
            console.log(
              '[StoryPlayback] Preloaded buffer:',
              item.generation_id,
              'duration:',
              audioBuffer.duration,
            );
          })
          .catch((err) => {
            console.error('[StoryPlayback] Failed to preload audio:', item.generation_id, err);
          });

        preloadPromises.push(preloadPromise);
      }
    }

    Promise.all(preloadPromises).then(() => {
      console.log('[StoryPlayback] Preloaded', audioBuffersRef.current.size, 'audio buffers');
    });
  }, [items, getAudioContext]);

  // Cleanup AudioContext on unmount
  useEffect(() => {
    return () => {
      // Stop all sources
      for (const [itemId] of activeSourcesRef.current) {
        stopSource(itemId);
      }
      activeSourcesRef.current.clear();

      // Clean up audio graph
      if (masterGainRef.current) {
        masterGainRef.current.disconnect();
        masterGainRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {
          // Ignore errors when closing
        });
        audioContextRef.current = null;
      }

      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [stopSource]);

  // Find ALL items that should be playing at a given story time
  const findActiveItems = useCallback(
    (storyTimeMs: number, itemList: StoryItemDetail[]): StoryItemDetail[] => {
      return itemList.filter((item) => {
        const itemStart = item.start_time_ms;
        // Use effective duration (accounting for trims)
        const trimStartMs = item.trim_start_ms || 0;
        const trimEndMs = item.trim_end_ms || 0;
        const effectiveDurationMs = item.duration * 1000 - trimStartMs - trimEndMs;
        const itemEnd = item.start_time_ms + effectiveDurationMs;
        return storyTimeMs >= itemStart && storyTimeMs < itemEnd;
      });
    },
    [],
  );

  // Convert AudioContext time to story time (ms)
  const contextTimeToStoryTime = useCallback(
    (contextTime: number): number => {
      if (playbackStartContextTime === null || playbackStartStoryTime === null) {
        return 0;
      }
      const elapsedContextTime = contextTime - playbackStartContextTime;
      return playbackStartStoryTime + elapsedContextTime * 1000;
    },
    [playbackStartContextTime, playbackStartStoryTime],
  );

  // Convert story time (ms) to AudioContext time
  const storyTimeToContextTime = useCallback(
    (storyTimeMs: number): number => {
      if (playbackStartContextTime === null || playbackStartStoryTime === null) {
        return 0;
      }
      const elapsedStoryTime = (storyTimeMs - playbackStartStoryTime) / 1000;
      return playbackStartContextTime + elapsedStoryTime;
    },
    [playbackStartContextTime, playbackStartStoryTime],
  );

  // Stop all sources
  const stopAllSources = useCallback(() => {
    console.log('[StoryPlayback] Stopping all sources');
    for (const [itemId] of activeSourcesRef.current) {
      stopSource(itemId);
    }
    activeSourcesRef.current.clear();
  }, [stopSource]);

  // Schedule playback for all items that should be playing
  const schedulePlayback = useCallback(
    (storyTimeMs: number, itemList: StoryItemDetail[]) => {
      const audioContext = getAudioContext();
      const currentContextTime = audioContext.currentTime;

      // Find all items that should be playing
      const shouldBePlaying = findActiveItems(storyTimeMs, itemList);
      const shouldBePlayingIds = new Set(shouldBePlaying.map((item) => item.id));

      // Stop sources that shouldn't be playing anymore
      for (const [itemId] of activeSourcesRef.current) {
        if (!shouldBePlayingIds.has(itemId)) {
          stopSource(itemId);
        }
      }

      // Schedule new sources for items that should be playing
      for (const item of shouldBePlaying) {
        if (!activeSourcesRef.current.has(item.id)) {
          const buffer = audioBuffersRef.current.get(item.generation_id);
          if (!buffer) {
            console.warn('[StoryPlayback] Buffer not loaded for:', item.generation_id);
            continue;
          }

          // Calculate when this item should start in AudioContext time
          const itemStartContextTime = storyTimeToContextTime(item.start_time_ms);
          
          // Calculate effective duration and trim offsets
          const trimStartSec = (item.trim_start_ms || 0) / 1000;
          const trimEndSec = (item.trim_end_ms || 0) / 1000;
          const effectiveDuration = item.duration - trimStartSec - trimEndSec;
          const itemEndStoryTime = item.start_time_ms + effectiveDuration * 1000;

          // Calculate offset into the buffer (if seeking mid-way)
          // Offset is relative to the trimmed start of the clip
          const offsetIntoEffectiveClip = Math.max(0, (storyTimeMs - item.start_time_ms) / 1000);
          const offsetIntoBuffer = trimStartSec + offsetIntoEffectiveClip;
          const duration = effectiveDuration - offsetIntoEffectiveClip;

          // If the item should have already started, schedule it to start immediately
          const startAtContextTime = Math.max(currentContextTime, itemStartContextTime);

          console.log('[StoryPlayback] Scheduling source:', {
            itemId: item.id,
            generationId: item.generation_id,
            storyTimeMs,
            itemStart: item.start_time_ms,
            offsetIntoBuffer,
            startAtContextTime,
            duration,
          });

          const source = audioContext.createBufferSource();
          source.buffer = buffer;
          // TODO: per-voice device routing — each item has a profile_id; look up its
          // channel → device_ids and use AudioContext.setSinkId (or playToDevices via
          // Tauri) so voices route to different speakers. Currently everything goes to
          // the system default output via audioContext.destination.
          source.connect(masterGainRef.current || audioContext.destination);

          const activeSource: ActiveSource = {
            source,
            itemId: item.id,
            generationId: item.generation_id,
            startTimeMs: item.start_time_ms,
            endTimeMs: itemEndStoryTime,
          };

          activeSourcesRef.current.set(item.id, activeSource);

          // Schedule playback
          source.start(startAtContextTime, offsetIntoBuffer, duration);

          // Clean up when source ends
          source.onended = () => {
            console.log('[StoryPlayback] Source ended:', item.id);
            activeSourcesRef.current.delete(item.id);
          };
        }
      }
    },
    [getAudioContext, findActiveItems, storyTimeToContextTime, stopSource],
  );

  // Sync visual playhead from AudioContext time
  useEffect(() => {
    if (!isPlaying || playbackStartContextTime === null || playbackStartStoryTime === null) {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const audioContext = getAudioContext();
    const itemList = playbackItems || [];

    const syncPlayhead = () => {
      if (!useStoryStore.getState().isPlaying) {
        return;
      }

      const currentContextTime = audioContext.currentTime;
      const currentStoryTime = contextTimeToStoryTime(currentContextTime);
      const totalDuration = useStoryStore.getState().totalDurationMs;

      // Update store with current story time
      useStoryStore.setState({ currentTimeMs: Math.min(currentStoryTime, totalDuration) });

      // Schedule any items that should be playing
      schedulePlayback(currentStoryTime, itemList);

      // Check if we've reached the end
      if (currentStoryTime >= totalDuration) {
        // Check if all sources have ended
        if (activeSourcesRef.current.size === 0) {
          console.log('[StoryPlayback] Reached end');
          useStoryStore.getState().stop();
          return;
        }
      }

      // Continue sync loop
      animationFrameRef.current = requestAnimationFrame(syncPlayhead);
    };

    // Initial sync
    const currentContextTime = audioContext.currentTime;
    const currentStoryTime = contextTimeToStoryTime(currentContextTime);
    schedulePlayback(currentStoryTime, itemList);

    // Start sync loop
    animationFrameRef.current = requestAnimationFrame(syncPlayhead);

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [
    isPlaying,
    playbackItems,
    playbackStartContextTime,
    playbackStartStoryTime,
    getAudioContext,
    contextTimeToStoryTime,
    schedulePlayback,
  ]);

  // Handle play/pause changes - stop sources when paused
  useEffect(() => {
    if (!isPlaying) {
      console.log('[StoryPlayback] Stopping playback');
      stopAllSources();
    }
  }, [isPlaying, stopAllSources]);

  // Handle seek - reset timing anchors when they become null (triggered by seek)
  useEffect(() => {
    if (!isPlaying || !playbackItems || playbackItems.length === 0) {
      return;
    }

    // Only run when timing anchors are null (after a seek)
    if (playbackStartContextTime !== null && playbackStartStoryTime !== null) {
      return;
    }

    const audioContext = getAudioContext();
    const currentContextTime = audioContext.currentTime;
    const currentStoryTime = useStoryStore.getState().currentTimeMs;

    console.log('[StoryPlayback] Setting timing anchors after seek:', {
      contextTime: currentContextTime,
      storyTime: currentStoryTime,
    });
    setPlaybackTiming(currentContextTime, currentStoryTime);

    // Stop all existing sources and reschedule from new position
    stopAllSources();
    schedulePlayback(currentStoryTime, playbackItems);
  }, [
    isPlaying,
    playbackItems,
    playbackStartContextTime,
    playbackStartStoryTime,
    getAudioContext,
    stopAllSources,
    schedulePlayback,
    setPlaybackTiming,
  ]);
}
