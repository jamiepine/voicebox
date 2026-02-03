import type { FileFilter, PlatformFilesystem } from '@/platform/types';

export const tauriFilesystem: PlatformFilesystem = {
  async saveFile(filename: string, blob: Blob, filters?: FileFilter[]) {
    try {
      const { save } = await import('@tauri-apps/plugin-dialog');
      const filePath = await save({
        defaultPath: filename,
        filters: filters || [],
      });

      if (filePath) {
        let resolvedPath = '';
        if (typeof filePath === 'string') {
          resolvedPath = filePath;
        } else if (filePath && typeof filePath === 'object' && 'path' in filePath) {
          resolvedPath = (filePath as { path: string }).path;
        }

        if (!resolvedPath) {
          throw new Error('Failed to resolve save path');
        }

        const { writeFile } = await import('@tauri-apps/plugin-fs');
        const arrayBuffer = await blob.arrayBuffer();
        await writeFile(resolvedPath, new Uint8Array(arrayBuffer));
      }
    } catch (error) {
      console.error('Failed to use Tauri dialog, falling back to browser download:', error);
      // Fall back to browser download if Tauri dialog fails
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    }
  },

  async openFolder(path: string): Promise<boolean> {
    try {
      const { open } = await import('@tauri-apps/plugin-shell');
      await open(path);
      return true;
    } catch (error) {
      console.error('Failed to open folder:', error);
      return false;
    }
  },
};
