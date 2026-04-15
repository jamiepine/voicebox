import { create } from 'zustand';
import type { EffectConfig } from '@/lib/api/types';

export interface ReuseParams {
  text: string;
  language?: string;
  engine?: string;
  model_size?: string;
  temperature?: number | null;
  top_k?: number | null;
  top_p?: number | null;
  repetition_penalty?: number | null;
  speed?: number | null;
  effects_chain?: EffectConfig[] | null;
}

interface GenerationState {
  /** IDs of generations currently in progress */
  pendingGenerationIds: Set<string>;
  /** Whether any generation is in progress (derived from pendingGenerationIds) */
  isGenerating: boolean;
  /** Map of generationId → storyId for deferred story additions */
  pendingStoryAdds: Map<string, string>;
  addPendingGeneration: (id: string) => void;
  removePendingGeneration: (id: string) => void;
  addPendingStoryAdd: (generationId: string, storyId: string) => void;
  removePendingStoryAdd: (generationId: string) => string | undefined;
  setActiveGenerationId: (id: string | null) => void;
  activeGenerationId: string | null;
  /** Params to reuse from history — set by HistoryTable, consumed by FloatingGenerateBox */
  reuseParams: ReuseParams | null;
  setReuseParams: (params: ReuseParams | null) => void;
}

export const useGenerationStore = create<GenerationState>((set, get) => ({
  pendingGenerationIds: new Set(),
  isGenerating: false,
  activeGenerationId: null,
  pendingStoryAdds: new Map(),
  reuseParams: null,
  setReuseParams: (params) => set({ reuseParams: params }),

  addPendingGeneration: (id) =>
    set((state) => {
      const next = new Set(state.pendingGenerationIds);
      next.add(id);
      return { pendingGenerationIds: next, isGenerating: true };
    }),

  removePendingGeneration: (id) =>
    set((state) => {
      const next = new Set(state.pendingGenerationIds);
      next.delete(id);
      return { pendingGenerationIds: next, isGenerating: next.size > 0 };
    }),

  addPendingStoryAdd: (generationId, storyId) =>
    set((state) => {
      const next = new Map(state.pendingStoryAdds);
      next.set(generationId, storyId);
      return { pendingStoryAdds: next };
    }),

  removePendingStoryAdd: (generationId) => {
    const storyId = get().pendingStoryAdds.get(generationId);
    if (storyId) {
      set((state) => {
        const next = new Map(state.pendingStoryAdds);
        next.delete(generationId);
        return { pendingStoryAdds: next };
      });
    }
    return storyId;
  },

  setActiveGenerationId: (id) => set({ activeGenerationId: id }),
}));
