import { create } from 'zustand';
import type { EffectConfig } from '@/lib/api/types';

interface EffectsStore {
  selectedPresetId: string | null;
  setSelectedPresetId: (id: string | null) => void;

  // Working chain for the detail panel (editing a preset or building a new one)
  workingChain: EffectConfig[];
  setWorkingChain: (chain: EffectConfig[]) => void;

  // Track if editing an existing preset vs creating new
  isCreatingNew: boolean;
  setIsCreatingNew: (v: boolean) => void;
}

export const useEffectsStore = create<EffectsStore>((set) => ({
  selectedPresetId: null,
  setSelectedPresetId: (id) => set({ selectedPresetId: id, isCreatingNew: false }),

  workingChain: [],
  setWorkingChain: (chain) => set({ workingChain: chain }),

  isCreatingNew: false,
  setIsCreatingNew: (v) => set({ isCreatingNew: v, ...(v && { selectedPresetId: null }) }),
}));
