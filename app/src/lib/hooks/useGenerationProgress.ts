import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { useGenerationSettings } from '@/lib/hooks/useSettings';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';

interface GenerationStatusEvent {
  id: string;
  status: 'loading_model' | 'generating' | 'completed' | 'failed' | 'not_found';
  duration?: number;
  error?: string;
  source?: string;
}

// Agent-initiated generations are played by the floating pill, not the
// main-window AudioPlayer. Skip autoplay here to avoid double-playback.
const AGENT_SOURCES = new Set(['mcp', 'rest']);

// SSE reconnect — cap at 30 s, give up after this many consecutive errors.
const MAX_RECONNECT_ATTEMPTS = 8;
const RECONNECT_BASE_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;

/**
 * Subscribes to SSE for all pending generations. When a generation completes,
 * refetches history, removes it from pending, and auto-plays if the player is idle.
 */
export function useGenerationProgress() {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const pendingIds = useGenerationStore((s) => s.pendingGenerationIds);
  const removePendingGeneration = useGenerationStore((s) => s.removePendingGeneration);
  const removePendingStoryAdd = useGenerationStore((s) => s.removePendingStoryAdd);
  const isPlaying = usePlayerStore((s) => s.isPlaying);
  const setAudioWithAutoPlay = usePlayerStore((s) => s.setAudioWithAutoPlay);
  const { settings: genSettings } = useGenerationSettings();
  const autoplayOnGenerate = genSettings?.autoplay_on_generate ?? true;

  // Keep refs to avoid stale closures in EventSource handlers.
  const isPlayingRef = useRef(isPlaying);
  const autoplayRef = useRef(autoplayOnGenerate);
  isPlayingRef.current = isPlaying;
  autoplayRef.current = autoplayOnGenerate;

  // Track active EventSource instances and reconnect state.
  const eventSourcesRef = useRef<Map<string, EventSource>>(new Map());
  const errorCountRef = useRef<Map<string, number>>(new Map());
  const reconnectTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  // Unmount-only cleanup — close all SSE connections and timers.
  useEffect(() => {
    const sources = eventSourcesRef.current;
    const timers = reconnectTimersRef.current;
    return () => {
      for (const timer of timers.values()) clearTimeout(timer);
      timers.clear();
      for (const source of sources.values()) source.close();
      sources.clear();
      errorCountRef.current.clear();
    };
  }, []);

  useEffect(() => {
    const currentSources = eventSourcesRef.current;
    const reconnectTimers = reconnectTimersRef.current;

    const clearReconnectTimer = (id: string) => {
      const timer = reconnectTimers.get(id);
      if (timer) {
        clearTimeout(timer);
        reconnectTimers.delete(id);
      }
    };

    // Close SSE connections and cancel pending reconnect timers for IDs that
    // are no longer pending.
    for (const [id, source] of currentSources.entries()) {
      if (!pendingIds.has(id)) {
        source.close();
        currentSources.delete(id);
      }
    }
    for (const [id, timer] of reconnectTimers.entries()) {
      if (!pendingIds.has(id)) {
        clearTimeout(timer);
        reconnectTimers.delete(id);
        errorCountRef.current.delete(id);
      }
    }

    const openSource = (id: string) => {
      if (!pendingIds.has(id) || currentSources.has(id)) return;

      const url = apiClient.getGenerationStatusUrl(id);
      const source = new EventSource(url);
      currentSources.set(id, source);

      const cleanup = (removePending = true) => {
        source.close();
        if (currentSources.get(id) === source) {
          currentSources.delete(id);
        }
        clearReconnectTimer(id);
        if (removePending) {
          removePendingGeneration(id);
          errorCountRef.current.delete(id);
        }
      };

      source.onmessage = (event) => {
        // A successful message resets the consecutive-error counter.
        errorCountRef.current.set(id, 0);

        try {
          const data: GenerationStatusEvent = JSON.parse(event.data);

          if (data.status === 'completed') {
            cleanup();

            // Refetch history to pick up the completed generation.
            queryClient.refetchQueries({ queryKey: ['history'] });

            // If this generation was queued for a story, add it now.
            const storyId = removePendingStoryAdd(id);
            if (storyId) {
              apiClient
                .addStoryItem(storyId, { generation_id: id })
                .then(() => {
                  queryClient.invalidateQueries({ queryKey: ['stories'] });
                  queryClient.invalidateQueries({ queryKey: ['stories', storyId] });
                  toast({
                    title: 'Added to story',
                    description: data.duration
                      ? `Audio generated (${data.duration.toFixed(2)}s) and added to story`
                      : 'Audio generated and added to story',
                  });
                })
                .catch(() => {
                  toast({
                    title: 'Generation complete',
                    description: 'Audio generated but failed to add to story',
                    variant: 'destructive',
                  });
                });
            }

            // Auto-play if enabled and nothing is currently playing.
            // Skip agent-initiated sources — the floating pill window plays those itself.
            const isAgentSpeak = data.source ? AGENT_SOURCES.has(data.source) : false;
            if (autoplayRef.current && !isPlayingRef.current && !isAgentSpeak) {
              const genAudioUrl = apiClient.getAudioUrl(id);
              setAudioWithAutoPlay(genAudioUrl, id, '', '');
            }
          } else if (data.status === 'failed' || data.status === 'not_found') {
            cleanup();
            removePendingStoryAdd(id);
            queryClient.refetchQueries({ queryKey: ['history'] });
            toast({
              title: data.status === 'not_found' ? 'Generation not found' : 'Generation failed',
              description: data.error || 'An error occurred during generation',
              variant: 'destructive',
            });
          }
        } catch {
          // Ignore parse errors from SSE heartbeats or malformed frames.
        }
      };

      source.onerror = () => {
        source.close();
        if (currentSources.get(id) === source) {
          currentSources.delete(id);
        }

        const attempts = (errorCountRef.current.get(id) ?? 0) + 1;
        errorCountRef.current.set(id, attempts);

        if (attempts >= MAX_RECONNECT_ATTEMPTS) {
          // Too many consecutive errors — give up and let the user see the
          // current state in the history list instead of hanging forever.
          removePendingGeneration(id);
          errorCountRef.current.delete(id);
          queryClient.refetchQueries({ queryKey: ['history'] });
          return;
        }

        // Exponential back-off reconnect (1 s, 2 s, 4 s … capped at 30 s).
        const delay = Math.min(RECONNECT_BASE_MS * 2 ** (attempts - 1), RECONNECT_MAX_MS);
        clearReconnectTimer(id);
        const timer = setTimeout(() => {
          reconnectTimers.delete(id);
          if (useGenerationStore.getState().pendingGenerationIds.has(id)) {
            openSource(id);
          }
        }, delay);
        reconnectTimers.set(id, timer);
      };
    };

    // Open SSE connections for new pending IDs.
    for (const id of pendingIds) {
      if (currentSources.has(id) || reconnectTimers.has(id)) continue;
      errorCountRef.current.set(id, 0);
      openSource(id);
    }
  }, [
    pendingIds,
    removePendingGeneration,
    removePendingStoryAdd,
    queryClient,
    toast,
    setAudioWithAutoPlay,
  ]);
}
