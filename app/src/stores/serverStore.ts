import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { queryClient } from '@/lib/queryClient';

interface ServerStore {
  serverUrl: string;
  setServerUrl: (url: string) => void;

  isConnected: boolean;
  setIsConnected: (connected: boolean) => void;

  mode: 'local' | 'remote';
  setMode: (mode: 'local' | 'remote') => void;

  keepServerRunningOnClose: boolean;
  setKeepServerRunningOnClose: (keepRunning: boolean) => void;

  customModelsDir: string | null;
  setCustomModelsDir: (dir: string | null) => void;
}

/**
 * Invalidate all React Query caches so stale data from the previous
 * server is not shown. Called when the server URL changes.
 */
function invalidateAllServerData() {
  queryClient.invalidateQueries();
}

export function getDefaultServerUrl(): string {
  const fallback = 'http://127.0.0.1:17493';

  if (!import.meta.env.PROD || typeof window === 'undefined') {
    return fallback;
  }

  const { protocol, origin, hostname } = window.location;
  if (
    (protocol === 'http:' || protocol === 'https:') &&
    origin &&
    hostname !== 'tauri.localhost'
  ) {
    return origin;
  }

  return fallback;
}

export function isLoopbackVoiceboxServerUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return (
      parsed.port === '17493' &&
      (parsed.hostname === '127.0.0.1' ||
        parsed.hostname === 'localhost' ||
        parsed.hostname === '[::1]' ||
        parsed.hostname === '::1')
    );
  } catch {
    return false;
  }
}

export const useServerStore = create<ServerStore>()(
  persist(
    (set, get) => ({
      serverUrl: getDefaultServerUrl(),
      setServerUrl: (url) => {
        const prev = get().serverUrl;
        set({ serverUrl: url });
        if (url !== prev) {
          invalidateAllServerData();
        }
      },

      isConnected: false,
      setIsConnected: (connected) => set({ isConnected: connected }),

      mode: 'local',
      setMode: (mode) => set({ mode }),

      keepServerRunningOnClose: false,
      setKeepServerRunningOnClose: (keepRunning) => set({ keepServerRunningOnClose: keepRunning }),

      customModelsDir: null,
      setCustomModelsDir: (dir) => set({ customModelsDir: dir }),
    }),
    {
      name: 'voicebox-server',
    },
  ),
);
