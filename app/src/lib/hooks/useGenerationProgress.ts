import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { generationEventBus, GenerationStatusEvent } from '@/lib/eventBus';
import { useGenerationSettings } from '@/lib/hooks/useSettings';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';

// Agent-initiated generations are played by the floating pill, not the
// main-window AudioPlayer. Skip autoplay here to avoid double-playback.
const AGENT_SOURCES = new Set(['mcp', 'rest']);

/**
 * Subscribes to the global /events/generations SSE bus and reacts to
 * every generation status update for pending IDs.
 *
 * The bus is shared across the whole app: opening this hook does NOT
 * open a new EventSource per call. That matters because under
 * HTTP/1.1 browsers cap same-origin connections at 6, and the original
 * implementation (one EventSource per pending generation) made the
 * "Send" button unresponsive past the 6th concurrent generation.
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

  // Keep refs to avoid stale closures in the SSE handler.
  const isPlayingRef = useRef(isPlaying);
  const autoplayRef = useRef(autoplayOnGenerate);
  isPlayingRef.current = isPlaying;
  autoplayRef.current = autoplayOnGenerate;

  // Stable refs for actions so the subscribe effect doesn't churn.
  const queryClientRef = useRef(queryClient);
  const toastRef = useRef(toast);
  const removePendingRef = useRef(removePendingGeneration);
  const removeStoryRef = useRef(removePendingStoryAdd);
  const setAudioRef = useRef(setAudioWithAutoPlay);
  queryClientRef.current = queryClient;
  toastRef.current = toast;
  removePendingRef.current = removePendingGeneration;
  removeStoryRef.current = removePendingStoryAdd;
  setAudioRef.current = setAudioWithAutoPlay;

  useEffect(() => {
    const pendingSet = new Set(useGenerationStore.getState().pendingGenerationIds);

    const handle = (data: GenerationStatusEvent) => {
      if (!pendingSet.has(data.id)) {
        // Event for an ID we don't currently track -- still useful to
        // refetch history so completed items appear in the list.
        if (data.status === 'completed' || data.status === 'failed') {
          queryClientRef.current.refetchQueries({ queryKey: ['history'] });
        }
        return;
      }

      if (data.status === 'completed') {
        removePendingRef.current(data.id);
        pendingSet.delete(data.id);

        queryClientRef.current.refetchQueries({ queryKey: ['history'] });

        const storyId = removeStoryRef.current(data.id);
        if (storyId) {
          apiClient
            .addStoryItem(storyId, { generation_id: data.id })
            .then(() => {
              queryClientRef.current.invalidateQueries({ queryKey: ['stories'] });
              queryClientRef.current.invalidateQueries({ queryKey: ['stories', storyId] });
              toastRef.current({
                title: 'Added to story',
                description: data.duration
                  ? `Audio generated (${data.duration.toFixed(2)}s) and added to story`
                  : 'Audio generated and added to story',
              });
            })
            .catch(() => {
              toastRef.current({
                title: 'Generation complete',
                description: 'Audio generated but failed to add to story',
                variant: 'destructive',
              });
            });
        }

        const isAgentSpeak = data.source ? AGENT_SOURCES.has(data.source) : false;
        if (autoplayRef.current && !isPlayingRef.current && !isAgentSpeak) {
          const genAudioUrl = apiClient.getAudioUrl(data.id);
          setAudioRef.current(genAudioUrl, data.id, '', '');
        }
      } else if (data.status === 'failed' || data.status === 'not_found') {
        removePendingRef.current(data.id);
        removeStoryRef.current(data.id);
        pendingSet.delete(data.id);

        queryClientRef.current.refetchQueries({ queryKey: ['history'] });

        toastRef.current({
          title: data.status === 'not_found' ? 'Generation not found' : 'Generation failed',
          description: data.error || 'An error occurred during generation',
          variant: 'destructive',
        });
      }
    };

    const unsubscribe = generationEventBus.subscribe(handle);
    return unsubscribe;
  }, [pendingIds]);
}