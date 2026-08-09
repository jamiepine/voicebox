import { create } from 'zustand';

export interface GenerationProgressData {
  progress: number;
  currentChunk?: number;
  totalChunks?: number;
  status: string;
  message?: string;
}

interface GenerationState {
  /** IDs of generations currently in progress */
  pendingGenerationIds: Set<string>;
  /** Whether any generation is in progress (derived from pendingGenerationIds) */
  isGenerating: boolean;
  /** Real-time progress data per generation ID */
  generationProgress: Map<string, GenerationProgressData>;
  /** Map of generationId → storyId for deferred story additions */
  pendingStoryAdds: Map<string, string>;
  addPendingGeneration: (id: string) => void;
  removePendingGeneration: (id: string) => void;
  updateGenerationProgress: (id: string, data: GenerationProgressData) => void;
  removeGenerationProgress: (id: string) => void;
  addPendingStoryAdd: (generationId: string, storyId: string) => void;
  removePendingStoryAdd: (generationId: string) => string | undefined;
  setActiveGenerationId: (id: string | null) => void;
  activeGenerationId: string | null;
}

export const useGenerationStore = create<GenerationState>((set, get) => ({
  pendingGenerationIds: new Set(),
  isGenerating: false,
  activeGenerationId: null,
  generationProgress: new Map(),
  pendingStoryAdds: new Map(),

  addPendingGeneration: (id) =>
    set((state) => {
      const next = new Set(state.pendingGenerationIds);
      next.add(id);
      return { pendingGenerationIds: next, isGenerating: true, activeGenerationId: id };
    }),

  removePendingGeneration: (id) =>
    set((state) => {
      const next = new Set(state.pendingGenerationIds);
      next.delete(id);
      const nextProgress = new Map(state.generationProgress);
      nextProgress.delete(id);
      return {
        pendingGenerationIds: next,
        isGenerating: next.size > 0,
        generationProgress: nextProgress,
        activeGenerationId: state.activeGenerationId === id ? null : state.activeGenerationId,
      };
    }),

  updateGenerationProgress: (id, data) =>
    set((state) => {
      const next = new Map(state.generationProgress);
      next.set(id, data);
      return { generationProgress: next };
    }),

  removeGenerationProgress: (id) =>
    set((state) => {
      const next = new Map(state.generationProgress);
      next.delete(id);
      return { generationProgress: next };
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
