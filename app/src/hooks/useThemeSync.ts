import { useEffect } from 'react';
import { useUIStore } from '@/stores/uiStore';

export function useThemeSync() {
  const theme = useUIStore((s) => s.theme);

  useEffect(() => {
    if (theme !== 'system') {
      document.documentElement.classList.toggle('dark', theme === 'dark');
      return;
    }

    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const apply = () => {
      document.documentElement.classList.toggle('dark', mq.matches);
    };

    apply();
    mq.addEventListener('change', apply);
    return () => mq.removeEventListener('change', apply);
  }, [theme]);
}
