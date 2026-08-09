import { describe, expect, it } from 'vitest';
import { useUIStore } from '@/stores/uiStore';

describe('uiStore', () => {
  it('applies the dark class when theme is set to dark', () => {
    useUIStore.getState().setTheme('dark');

    expect(useUIStore.getState().theme).toBe('dark');
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });

  it('removes the dark class when theme is set to light', () => {
    useUIStore.getState().setTheme('dark');
    useUIStore.getState().setTheme('light');

    expect(document.documentElement.classList.contains('dark')).toBe(false);
  });

  it('persists only theme and selectedProfileId', () => {
    useUIStore.getState().setTheme('light');
    useUIStore.getState().setSidebarOpen(false);
    useUIStore.getState().setSelectedEngine('kokoro');

    const persisted = JSON.parse(localStorage.getItem('voicebox-ui') ?? '{}');
    expect(persisted.state).toEqual({ selectedProfileId: null, theme: 'light' });
  });
});
