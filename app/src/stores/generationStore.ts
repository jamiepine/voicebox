import { create } from 'zustand';

export interface PendingJob {
  id: string;
  profileId: string;
  profileName: string;
  text: string;
  language: string;
  modelSize?: string;
  backendType?: string;
  requestedByUserId?: string;
  requestedByFirstName?: string;
  requestIp?: string;
  status: 'queued' | 'generating' | 'cancelling';
  progress: number; // 0-100
  createdAt: string;
  startedAt?: string;
}

interface GenerationState {
  // Legacy fields used by upstream voicebox app.
  isGenerating: boolean;
  activeGenerationId: string | null;
  pendingJobs: PendingJob[];
  setIsGenerating: (isGenerating: boolean) => void;
  setActiveGenerationId: (id: string | null) => void;
  addJob: (job: PendingJob) => void;
  mergePendingJobs: (jobs: PendingJob[]) => void;
  removeJob: (id: string) => void;
  setPendingJobs: (jobs: PendingJob[]) => void;
  updateJobProgress: (id: string, progress: number) => void;
  updateJobStatus: (id: string, status: 'queued' | 'generating' | 'cancelling') => void;
  /** True if any job is queued or generating. */
  hasActiveJobs: () => boolean;
}

export const useGenerationStore = create<GenerationState>((set, get) => ({
  isGenerating: false,
  activeGenerationId: null,
  pendingJobs: [],
  setIsGenerating: (isGenerating) => set({ isGenerating }),
  setActiveGenerationId: (id) => set({ activeGenerationId: id }),
  addJob: (job) =>
    set((state) => ({
      isGenerating: true,
      activeGenerationId: state.activeGenerationId || job.id,
      pendingJobs: state.pendingJobs.some((j) => j.id === job.id)
        ? state.pendingJobs.map((j) => (j.id === job.id ? { ...j, ...job } : j))
        : [...state.pendingJobs, job],
    })),
  mergePendingJobs: (jobs) =>
    set((state) => {
      const byId = new Map(state.pendingJobs.map((job) => [job.id, job]));
      for (const job of jobs) {
        const existing = byId.get(job.id);
        byId.set(job.id, existing ? { ...existing, ...job } : job);
      }
      const merged = Array.from(byId.values());
      return {
        pendingJobs: merged,
        isGenerating: merged.length > 0 || state.isGenerating,
        activeGenerationId: state.activeGenerationId || merged[0]?.id || null,
      };
    }),
  removeJob: (id) =>
    set((state) => ({
      pendingJobs: state.pendingJobs.filter((j) => j.id !== id),
      isGenerating: state.pendingJobs.filter((j) => j.id !== id).length > 0,
      activeGenerationId: state.activeGenerationId === id
        ? (state.pendingJobs.filter((j) => j.id !== id)[0]?.id ?? null)
        : state.activeGenerationId,
    })),
  setPendingJobs: (jobs) => set({
    pendingJobs: jobs,
    isGenerating: jobs.length > 0,
    activeGenerationId: jobs[0]?.id ?? null,
  }),
  updateJobProgress: (id, progress) =>
    set((state) => ({
      pendingJobs: state.pendingJobs.map((j) =>
        j.id === id ? { ...j, progress } : j,
      ),
    })),
  updateJobStatus: (id, status) =>
    set((state) => ({
      pendingJobs: state.pendingJobs.map((j) =>
        j.id === id ? { ...j, status } : j,
      ),
    })),
  hasActiveJobs: () => get().pendingJobs.length > 0,
}));
