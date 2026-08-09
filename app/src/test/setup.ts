import '@/i18n';
import { afterEach, vi } from 'vitest';
import { resetAllStores } from './resetStores';

afterEach(() => {
  vi.restoreAllMocks();
  resetAllStores();
});
