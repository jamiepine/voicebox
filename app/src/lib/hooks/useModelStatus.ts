import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

export function useModelStatus() {
    return useQuery({
        queryKey: ['models', 'status'],
        queryFn: () => apiClient.getModelStatus(),
        refetchInterval: 10000, // Check every 10 seconds
    });
}
