import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { queryClient } from '@/lib/queryClient';

interface ServerStore {
  serverUrl: string;
  serverUrlSource: 'auto' | 'custom';
  setServerUrl: (url: string) => void;
  setDiscoveredServerUrl: (url: string) => void;

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

export const useServerStore = create<ServerStore>()(
  persist(
    (set, get) => ({
      serverUrl: getDefaultServerUrl(),
      serverUrlSource: 'auto',
      setServerUrl: (url) => {
        const prev = get().serverUrl;
        set({ serverUrl: url, serverUrlSource: 'custom' });
        if (url !== prev) {
          invalidateAllServerData();
        }
      },

      setDiscoveredServerUrl: (url) => {
        // Read the latest selection: a user may save while startup is pending.
        if (get().serverUrlSource === 'custom') return;
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
      version: 1,
      migrate: (persistedState) => {
        const state = persistedState as Partial<ServerStore>;
        // Older versions did not record whether a URL was saved by the user.
        // Only exact known defaults are safe to treat as automatic. Preserve
        // all other URLs, including custom loopback ports, paths and protocols.
        const isDefault =
          !state.serverUrl ||
          state.serverUrl === 'http://127.0.0.1:17493' ||
          state.serverUrl === getDefaultServerUrl();
        return { ...state, serverUrlSource: isDefault ? 'auto' : 'custom' };
      },
    },
  ),
);
