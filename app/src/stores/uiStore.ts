import { create } from 'zustand';

// Draft state for the create voice profile form
export interface ProfileFormDraft {
  name: string;
  description: string;
  language: string;
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

  // Theme
  theme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark') => void;
}

export const useUIStore = create<UIStore>((set) => ({
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

  theme: 'light',
  setTheme: (theme) => {
    set({ theme });
    document.documentElement.classList.toggle('dark', theme === 'dark');
  },
}));
