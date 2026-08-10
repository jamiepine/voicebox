import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { FolderKind } from '@/lib/api/types';

export type Theme = 'light' | 'dark' | 'system';

/** How the Generate tab lists voices. */
export type VoiceViewMode = 'list' | 'card';

function resolveTheme(theme: Theme): 'light' | 'dark' {
  if (theme !== 'system') return theme;
  if (typeof window === 'undefined') return 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme: Theme) {
  if (typeof document === 'undefined') return;
  document.documentElement.classList.toggle('dark', resolveTheme(theme) === 'dark');
}

// Draft state for the create voice profile form
export interface ProfileFormDraft {
  name: string;
  description: string;
  language: string;
  personality: string;
  referenceText: string;
  sampleMode: 'upload' | 'record' | 'system';
  // Note: File objects can't be persisted, so we store metadata
  sampleFileName?: string;
  sampleFileType?: string;
  sampleFileData?: string; // Base64 encoded
}

interface UIStore {
  // Sidebar
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;

  // Modals
  profileDialogOpen: boolean;
  setProfileDialogOpen: (open: boolean) => void;
  editingProfileId: string | null;
  setEditingProfileId: (id: string | null) => void;

  generationDialogOpen: boolean;
  setGenerationDialogOpen: (open: boolean) => void;

  // Selected profile for generation
  selectedProfileId: string | null;
  setSelectedProfileId: (id: string | null) => void;

  // Currently selected engine (synced from generation form)
  selectedEngine: string;
  setSelectedEngine: (engine: string) => void;

  // Selected voice in Voices tab inspector
  selectedVoiceId: string | null;
  setSelectedVoiceId: (id: string | null) => void;

  // Profile form draft (for persisting create voice modal state)
  profileFormDraft: ProfileFormDraft | null;
  setProfileFormDraft: (draft: ProfileFormDraft | null) => void;

  // How the Generate tab renders voices
  voiceViewMode: VoiceViewMode;
  setVoiceViewMode: (mode: VoiceViewMode) => void;

  // Collapsed folder ids, keyed by folder kind. Collapsed rather than
  // expanded ids so a newly created folder starts open without having to
  // touch this set.
  collapsedFolderIds: Partial<Record<FolderKind, string[]>>;
  toggleFolderCollapsed: (kind: FolderKind, folderId: string) => void;

  // Theme
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      setSidebarOpen: (open) => set({ sidebarOpen: open }),

      profileDialogOpen: false,
      setProfileDialogOpen: (open) => set({ profileDialogOpen: open }),
      editingProfileId: null,
      setEditingProfileId: (id) => set({ editingProfileId: id }),

      generationDialogOpen: false,
      setGenerationDialogOpen: (open) => set({ generationDialogOpen: open }),

      selectedProfileId: null,
      setSelectedProfileId: (id) => set({ selectedProfileId: id }),

      selectedEngine: 'qwen',
      setSelectedEngine: (engine) => set({ selectedEngine: engine }),

      selectedVoiceId: null,
      setSelectedVoiceId: (id) => set({ selectedVoiceId: id }),

      profileFormDraft: null,
      setProfileFormDraft: (draft) => set({ profileFormDraft: draft }),

      voiceViewMode: 'list',
      setVoiceViewMode: (mode) => set({ voiceViewMode: mode }),

      collapsedFolderIds: { voice: [], generation: [], story: [] },
      toggleFolderCollapsed: (kind, folderId) =>
        set((state) => {
          const current = state.collapsedFolderIds[kind] ?? [];
          const next = current.includes(folderId)
            ? current.filter((id) => id !== folderId)
            : [...current, folderId];
          return { collapsedFolderIds: { ...state.collapsedFolderIds, [kind]: next } };
        }),

      theme: 'system',
      setTheme: (theme) => {
        set({ theme });
        applyTheme(theme);
      },
    }),
    {
      name: 'voicebox-ui',
      partialize: (state) => ({
        selectedProfileId: state.selectedProfileId,
        theme: state.theme,
        voiceViewMode: state.voiceViewMode,
        collapsedFolderIds: state.collapsedFolderIds,
      }),
      onRehydrateStorage: () => (state) => {
        if (state) applyTheme(state.theme);
        // Persisted before folders existed, so an older store has no map —
        // and a store persisted before story folders has no 'story' key.
        if (state && !state.collapsedFolderIds) {
          state.collapsedFolderIds = { voice: [], generation: [], story: [] };
        }
      },
    },
  ),
);
