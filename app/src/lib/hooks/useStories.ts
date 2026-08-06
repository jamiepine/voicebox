import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import type {
  ExportAudioFormat,
  StoryCreate,
  StoryItemBatchUpdate,
  StoryItemCreate,
  StoryItemFadeUpdate,
  StoryItemMove,
  StoryItemReorder,
  StoryItemSpeedUpdate,
  StoryItemSplit,
  StoryItemTrim,
  StoryItemVersionUpdate,
  StoryItemVolumeUpdate,
  StoryTrackUpsert,
} from '@/lib/api/types';
import { usePlatform } from '@/platform/PlatformContext';

export function useStories() {
  return useQuery({
    queryKey: ['stories'],
    queryFn: () => apiClient.listStories(),
  });
}

export function useStory(storyId: string | null) {
  return useQuery({
    queryKey: ['stories', storyId],
    queryFn: () => apiClient.getStory(storyId!),
    enabled: !!storyId,
  });
}

export function useCreateStory() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: StoryCreate) => apiClient.createStory(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
    },
  });
}

export function useUpdateStory() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ storyId, data }: { storyId: string; data: StoryCreate }) =>
      apiClient.updateStory(storyId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

export function useDeleteStory() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (storyId: string) => apiClient.deleteStory(storyId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
    },
  });
}

export function useAddStoryItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ storyId, data }: { storyId: string; data: StoryItemCreate }) =>
      apiClient.addStoryItem(storyId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

export function useRemoveStoryItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ storyId, itemId }: { storyId: string; itemId: string }) =>
      apiClient.removeStoryItem(storyId, itemId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

export function useUpdateStoryItemTimes() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ storyId, data }: { storyId: string; data: StoryItemBatchUpdate }) =>
      apiClient.updateStoryItemTimes(storyId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

export function useReorderStoryItems() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ storyId, data }: { storyId: string; data: StoryItemReorder }) =>
      apiClient.reorderStoryItems(storyId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

export function useMoveStoryItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      storyId,
      itemId,
      data,
    }: {
      storyId: string;
      itemId: string;
      data: StoryItemMove;
    }) => apiClient.moveStoryItem(storyId, itemId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

export function useTrimStoryItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      storyId,
      itemId,
      data,
    }: {
      storyId: string;
      itemId: string;
      data: StoryItemTrim;
    }) => apiClient.trimStoryItem(storyId, itemId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

export function useUpdateStoryItemVolume() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      storyId,
      itemId,
      data,
    }: {
      storyId: string;
      itemId: string;
      data: StoryItemVolumeUpdate;
    }) => apiClient.updateStoryItemVolume(storyId, itemId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

export function useUpdateStoryItemFades() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      storyId,
      itemId,
      data,
    }: {
      storyId: string;
      itemId: string;
      data: StoryItemFadeUpdate;
    }) => apiClient.updateStoryItemFades(storyId, itemId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

export function useUpdateStoryItemSpeed() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      storyId,
      itemId,
      data,
    }: {
      storyId: string;
      itemId: string;
      data: StoryItemSpeedUpdate;
    }) => apiClient.updateStoryItemSpeed(storyId, itemId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

// ── Track mixer settings ─────────────────────────────────────────────

export function useStoryTracks(storyId: string | null) {
  return useQuery({
    queryKey: ['stories', storyId, 'tracks'],
    queryFn: () => apiClient.listStoryTracks(storyId as string),
    enabled: !!storyId,
  });
}

export function useUpsertStoryTrack() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      storyId,
      index,
      data,
    }: {
      storyId: string;
      index: number;
      data: StoryTrackUpsert;
    }) => apiClient.upsertStoryTrack(storyId, index, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId, 'tracks'] });
    },
  });
}

export function useDeleteStoryTrack() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ storyId, index }: { storyId: string; index: number }) =>
      apiClient.deleteStoryTrack(storyId, index),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId, 'tracks'] });
    },
  });
}

export function useSplitStoryItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      storyId,
      itemId,
      data,
    }: {
      storyId: string;
      itemId: string;
      data: StoryItemSplit;
    }) => apiClient.splitStoryItem(storyId, itemId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

export function useDuplicateStoryItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ storyId, itemId }: { storyId: string; itemId: string }) =>
      apiClient.duplicateStoryItem(storyId, itemId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

export function useSetStoryItemVersion() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      storyId,
      itemId,
      data,
    }: {
      storyId: string;
      itemId: string;
      data: StoryItemVersionUpdate;
    }) => apiClient.setStoryItemVersion(storyId, itemId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['stories', variables.storyId] });
    },
  });
}

export function useExportStoryAudio() {
  const platform = usePlatform();

  return useMutation({
    mutationFn: async ({
      storyId,
      storyName,
      format = 'wav',
      normalizeLoudness = false,
    }: {
      storyId: string;
      storyName: string;
      format?: ExportAudioFormat;
      normalizeLoudness?: boolean;
    }) => {
      const blob = await apiClient.exportStoryAudio(storyId, { format, normalizeLoudness });

      // Create safe filename
      const safeName = storyName
        .substring(0, 50)
        .replace(/[^a-z0-9]/gi, '-')
        .toLowerCase();
      const filename = `${safeName || 'story'}.${format}`;

      await platform.filesystem.saveFile(filename, blob, [
        {
          name: 'Audio File',
          extensions: [format],
        },
      ]);

      return blob;
    },
  });
}
