import type { PlatformMetadata } from '@/platform/types';

export const webMetadata: PlatformMetadata = {
  async getVersion(): Promise<string> {
    // Return version from env var or package.json
    return import.meta.env.VITE_APP_VERSION || '0.1.0';
  },
  getBuildInfo(): string {
    // Web builds don't inject git hash by default
    return '';
  },
  isTauri: false,
};
