import { create } from 'zustand';
import { persist } from 'zustand/middleware';

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

  customModelsDir: string | null;
  setCustomModelsDir: (dir: string | null) => void;
}

export const useServerStore = create<ServerStore>()(
  persist(
    (set) => ({
      serverUrl: 'http://127.0.0.1:17493',
      setServerUrl: (url) => set({ serverUrl: url }),

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

      customModelsDir: null,
      setCustomModelsDir: (dir) => set({ customModelsDir: dir }),
    }),
    {
      name: 'voicebox-server',
    },
  ),
);
