import { create } from 'zustand';
import type { ServerLogEntry } from '@/platform/types';

const MAX_LOG_ENTRIES = 2000;

let nextLogEntryId = 0;

export interface LogEntry extends ServerLogEntry {
  id: number;
  timestamp: number;
}

interface LogStore {
  entries: LogEntry[];
  addEntry: (entry: ServerLogEntry) => void;
  clear: () => void;
}

export const useLogStore = create<LogStore>((set) => ({
  entries: [],
  addEntry: (entry) =>
    set((state) => {
      const newEntry: LogEntry = { ...entry, id: nextLogEntryId++, timestamp: Date.now() };
      const entries = [...state.entries, newEntry];
      if (entries.length > MAX_LOG_ENTRIES) {
        return { entries: entries.slice(entries.length - MAX_LOG_ENTRIES) };
      }
      return { entries };
    }),
  clear: () => set({ entries: [] }),
}));
