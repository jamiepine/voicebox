import type { AppFolder, FileFilter, PlatformFilesystem } from '@/platform/types';

export const webFilesystem: PlatformFilesystem = {
  async saveFile(filename: string, blob: Blob, _filters?: FileFilter[]) {
    // Browser: trigger download
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  },

  async openAppFolder(_folder: AppFolder) {
    // No filesystem access in browser
  },

  async pickDirectory(_title: string) {
    return null;
  },
};
