import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '@/lib/api/client';
import { useGenerationStore } from '@/stores/generationStore';
import type { ActiveDownloadTask } from '@/lib/api/types';

// Polling interval in milliseconds
const POLL_INTERVAL = 2000;

/**
 * Hook to monitor active tasks (downloads and generations).
 * Polls the server periodically to catch downloads triggered from anywhere
 * (transcription, generation, explicit download, etc.).
 * 
 * Returns the active downloads so components can render download toasts.
 */
export function useRestoreActiveTasks() {
  const [activeDownloads, setActiveDownloads] = useState<ActiveDownloadTask[]>([]);
  const setIsGenerating = useGenerationStore((state) => state.setIsGenerating);
  const setActiveGenerationId = useGenerationStore((state) => state.setActiveGenerationId);
  
  // Track current download names to avoid spurious re-renders on every poll
  const activeDownloadNamesRef = useRef<string>('');

  const fetchActiveTasks = useCallback(async () => {
    try {
      const tasks = await apiClient.getActiveTasks();

      // Update generation state
      if (tasks.generations.length > 0) {
        setIsGenerating(true);
        setActiveGenerationId(tasks.generations[0].task_id);
      } else {
        // Only clear if we were tracking a generation
        const currentId = useGenerationStore.getState().activeGenerationId;
        if (currentId) {
          setIsGenerating(false);
          setActiveGenerationId(null);
        }
      }

      // Only update state (and cause re-renders) when the set of downloading
      // model names actually changes — prevents SSE from reconnecting every 2s.
      const newKey = tasks.downloads.map((d) => d.model_name).sort().join(',');
      if (newKey !== activeDownloadNamesRef.current) {
        activeDownloadNamesRef.current = newKey;
        setActiveDownloads(tasks.downloads);
      }
    } catch (error) {
      // Silently fail - server might be temporarily unavailable
      console.debug('Failed to fetch active tasks:', error);
    }
  }, [setIsGenerating, setActiveGenerationId]);

  useEffect(() => {
    // Fetch immediately on mount
    fetchActiveTasks();

    // Poll for active tasks
    const interval = setInterval(fetchActiveTasks, POLL_INTERVAL);

    return () => clearInterval(interval);
  }, [fetchActiveTasks]);

  return activeDownloads;
}

/**
 * Map model names to display names for download toasts.
 */
export const MODEL_DISPLAY_NAMES: Record<string, string> = {
  'qwen-tts-1.7B': 'Qwen TTS 1.7B',
  'qwen-tts-0.6B': 'Qwen TTS 0.6B',
  'whisper-base': 'Whisper Base',
  'whisper-small': 'Whisper Small',
  'whisper-medium': 'Whisper Medium',
  'whisper-large': 'Whisper Large',
};
