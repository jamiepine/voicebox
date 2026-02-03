import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { useServerStore } from '@/stores/serverStore';

export function useSystemFolders() {
  const serverUrl = useServerStore((state) => state.serverUrl);

  return useQuery({
    queryKey: ['system', 'folders', serverUrl],
    queryFn: () => apiClient.getSystemFolders(),
    staleTime: 60000, // Cache for 1 minute - folder paths don't change often
    retry: 1,
  });
}
