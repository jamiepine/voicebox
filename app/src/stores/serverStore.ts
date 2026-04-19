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

  maxChunkChars: number;
  setMaxChunkChars: (value: number) => void;

  crossfadeMs: number;
  setCrossfadeMs: (value: number) => void;

  normalizeAudio: boolean;
  setNormalizeAudio: (value: boolean) => void;

  autoplayOnGenerate: boolean;
  setAutoplayOnGenerate: (value: boolean) => void;

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

export const useServerStore = create<ServerStore>()(
  persist(
    (set, get) => ({
      serverUrl: 'http://127.0.0.1:17493',
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

      maxChunkChars: 800,
      setMaxChunkChars: (value) => set({ maxChunkChars: value }),

      crossfadeMs: 50,
      setCrossfadeMs: (value) => set({ crossfadeMs: value }),

      normalizeAudio: true,
      setNormalizeAudio: (value) => set({ normalizeAudio: value }),

      autoplayOnGenerate: true,
      setAutoplayOnGenerate: (value) => set({ autoplayOnGenerate: value }),

      customModelsDir: null,
      setCustomModelsDir: (dir) => set({ customModelsDir: dir }),
    }),
    {
      name: 'voicebox-server',
    },
  ),
);
