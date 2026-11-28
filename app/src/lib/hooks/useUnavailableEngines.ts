import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';
import { apiClient } from '@/lib/api/client';

/** Engines the backend reports as unable to run on this machine, mapped to the reason. */
export function useUnavailableEngines(): Map<string, string> {
  const { data } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: () => apiClient.getModelStatus(),
  });

  return useMemo(
    () =>
      new Map(
        (data?.models ?? [])
          .filter((m) => m.available === false && m.engine)
          .map((m) => [m.engine as string, m.unavailable_reason ?? ''] as const),
      ),
    [data],
  );
}
