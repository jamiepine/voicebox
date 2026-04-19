import { create } from 'zustand';

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
}

export const useGenerationStore = create<GenerationState>((set, get) => ({
  pendingGenerationIds: new Set(),
  isGenerating: false,
  activeGenerationId: null,
  pendingStoryAdds: new Map(),

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
