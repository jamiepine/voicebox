import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

/**
 * Returns whether a Whisper (transcription) model is downloaded and ready.
 * Polls /models/status and checks for any whisper-* model with downloaded: true.
 *
 * Also exposes `isDownloading` so the UI can show a progress indicator.
 */
interface UseWhisperModelReadyOptions {
  enabled?: boolean;
}

export function useWhisperModelReady(options: UseWhisperModelReadyOptions = {}) {
  const { enabled = true } = options;
  const { data } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: () => apiClient.getModelStatus(),
    refetchInterval: enabled ? 15_000 : false,
    enabled,
  });

  const whisperModels = data?.models?.filter((m) => m.model_name.startsWith('whisper-')) ?? [];
  const ready = whisperModels.some((m) => m.downloaded);
  const downloading = whisperModels.some((m) => m.downloading);

  return { ready, downloading };
}
