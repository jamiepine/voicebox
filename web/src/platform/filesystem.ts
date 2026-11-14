import type { FileFilter, PlatformFilesystem } from '@/platform/types';

export const webFilesystem: PlatformFilesystem = {
  async saveFile(filename, contents, _filters?: FileFilter[]) {
    // Browser: trigger download
    const blob = typeof contents === 'function' ? await contents() : contents;
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    return filename;
  },

  async openPath(_path: string) {
    // No filesystem access in browser
  },

  async pickDirectory(_title: string) {
    return null;
  },
};
