import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { useServerStore } from '@/stores/serverStore';

export function useServerHealth() {
  const serverUrl = useServerStore((state) => state.serverUrl);

  return useQuery({
    queryKey: ['server', 'health', serverUrl],
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 5000, // Check every 5 seconds for backend changes
    refetchOnMount: 'always', // Always refetch on mount
    retry: 1,
  });
}
