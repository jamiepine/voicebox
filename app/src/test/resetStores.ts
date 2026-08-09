import { queryClient } from '@/lib/queryClient';
import { useAudioChannelStore } from '@/stores/audioChannelStore';
import { useEffectsStore } from '@/stores/effectsStore';
import { useGenerationStore } from '@/stores/generationStore';
import { useLogStore } from '@/stores/logStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useServerStore } from '@/stores/serverStore';
import { useStoryStore } from '@/stores/storyStore';
import { useUIStore } from '@/stores/uiStore';

const stores = [
  useAudioChannelStore,
  useEffectsStore,
  useGenerationStore,
  useLogStore,
  usePlayerStore,
  useServerStore,
  useStoryStore,
  useUIStore,
] as const;

// Snapshot pristine state at module load, before any test mutates anything.
const snapshots = stores.map((store) => store.getState());

/**
 * Restore every zustand store to its initial state and clear persisted
 * copies so tests can't leak state into each other. Persisted stores write
 * through to localStorage on setState, so localStorage is cleared last.
 */
export function resetAllStores(): void {
  stores.forEach((store, i) => {
    store.setState(snapshots[i] as never, true);
  });
  queryClient.clear();
  localStorage.clear();
}
