import { getVersion } from '@tauri-apps/api/app';
import type { PlatformMetadata } from '@/platform/types';

export const tauriMetadata: PlatformMetadata = {
  async getVersion(): Promise<string> {
    try {
      return await getVersion();
    } catch (error) {
      console.error('Failed to get version:', error);
      return '0.1.0';
    }
  },
  getBuildInfo(): string {
    try {
      const hash = __GIT_HASH__;
      const count = __GIT_COMMIT_COUNT__;
      if (import.meta.env.DEV) {
        return `dev-${hash}`;
      }
      return `${hash} #${count}`;
    } catch {
      return '';
    }
  },
  isTauri: true,
};
