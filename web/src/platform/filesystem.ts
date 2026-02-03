import type { FileFilter, PlatformFilesystem } from '@/platform/types';

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

  async openFolder(_path: string): Promise<boolean> {
    // Browsers cannot open local folders for security reasons
    // The UI will show the path as a read-only text instead
    return false;
  },
};
