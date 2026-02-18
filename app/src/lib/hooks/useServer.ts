import { useQuery } from '@tanstack/react-query';
import { apiClient, getLastApiEmissionTimestamp, getPiggybackHealth } from '@/lib/api/client';
import type { HealthResponse } from '@/lib/api/types';
import { useServerStore } from '@/stores/serverStore';

let _lastResolvedHealth: HealthResponse | null = null;

function fromPiggyback(): HealthResponse | null {
  const piggyback = getPiggybackHealth(45_000);
  if (!piggyback) return null;
  return {
    status: 'healthy',
    model_loaded: piggyback.model_loaded,
    model_size: piggyback.model_size ?? undefined,
    gpu_available: piggyback.gpu_type !== null,
    gpu_type: piggyback.gpu_type ?? undefined,
    backend_type: piggyback.backend ?? undefined,
  };
}

export function useServerHealth() {
  const serverUrl = useServerStore((state) => state.serverUrl);

  return useQuery({
    queryKey: ['server', 'health', serverUrl],
    queryFn: async (): Promise<HealthResponse> => {
      const piggybackHealth = fromPiggyback();
      if (piggybackHealth) {
        _lastResolvedHealth = piggybackHealth;
        return piggybackHealth;
      }

      // Only call /health when no API emission occurred in the last 7s.
      const recentApiEmission = Date.now() - getLastApiEmissionTimestamp() < 7_000;
      if (recentApiEmission) {
        if (_lastResolvedHealth) return _lastResolvedHealth;
        const stalePiggyback = getPiggybackHealth(Number.MAX_SAFE_INTEGER);
        if (stalePiggyback) {
          const mapped: HealthResponse = {
            status: 'healthy',
            model_loaded: stalePiggyback.model_loaded,
            model_size: stalePiggyback.model_size ?? undefined,
            gpu_available: stalePiggyback.gpu_type !== null,
            gpu_type: stalePiggyback.gpu_type ?? undefined,
            backend_type: stalePiggyback.backend ?? undefined,
          };
          _lastResolvedHealth = mapped;
          return mapped;
        }
      }

      const freshHealth = await apiClient.getHealth();
      _lastResolvedHealth = freshHealth;
      return freshHealth;
    },
    refetchInterval: 30000,
    retry: 1,
  });
}
