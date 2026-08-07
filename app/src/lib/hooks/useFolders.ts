import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import type { FolderKind, FolderUpdate } from '@/lib/api/types';

/**
 * Folders for voices and clips.
 *
 * Both kinds live in one table server-side, so every query is keyed by kind
 * — otherwise the voice panel and the clip panel would evict each other's
 * cache entry on every mutation.
 */

export function useFolders(kind: FolderKind) {
  return useQuery({
    queryKey: ['folders', kind],
    queryFn: () => apiClient.listFolders(kind),
  });
}

export function useCreateFolder(kind: FolderKind) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ name, parentId }: { name: string; parentId?: string | null }) =>
      apiClient.createFolder({ name, kind, parent_id: parentId ?? null }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['folders', kind] });
    },
  });
}

export function useUpdateFolder(kind: FolderKind) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ folderId, data }: { folderId: string; data: FolderUpdate }) =>
      apiClient.updateFolder(folderId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['folders', kind] });
    },
  });
}

export function useDetachFolder(kind: FolderKind) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (folderId: string) => apiClient.detachFolder(folderId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['folders', kind] });
    },
  });
}

export function useDeleteFolder(kind: FolderKind) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (folderId: string) => apiClient.deleteFolder(folderId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['folders', kind] });
      // Deleting a folder releases its members, so whichever list holds them
      // is now stale too.
      queryClient.invalidateQueries({
        queryKey: [kind === 'voice' ? 'profiles' : 'history'],
      });
    },
  });
}

export function useSetProfileFolder() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ profileId, folderId }: { profileId: string; folderId: string | null }) =>
      apiClient.setProfileFolder(profileId, folderId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      // item_count changes on both the old and new folder.
      queryClient.invalidateQueries({ queryKey: ['folders', 'voice'] });
    },
  });
}

export function useSetStoryFolder() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ storyId, folderId }: { storyId: string; folderId: string | null }) =>
      apiClient.setStoryFolder(storyId, folderId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['stories'] });
      queryClient.invalidateQueries({ queryKey: ['folders', 'story'] });
    },
  });
}

export function useSetGenerationFolder() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ generationId, folderId }: { generationId: string; folderId: string | null }) =>
      apiClient.setGenerationFolder(generationId, folderId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['history'] });
      queryClient.invalidateQueries({ queryKey: ['folders', 'generation'] });
    },
  });
}
