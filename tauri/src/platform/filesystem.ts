import type { FileFilter, PlatformFilesystem } from '@/platform/types';

export const tauriFilesystem: PlatformFilesystem = {
  async saveFile(filename: string, blob: Blob, filters?: FileFilter[]) {
    const { save } = await import('@tauri-apps/plugin-dialog');
    const { writeFile } = await import('@tauri-apps/plugin-fs');

    const filePath = await save({
      defaultPath: filename,
      filters: filters || [],
    });

    if (!filePath) return; // User cancelled the dialog

    const resolvedPath =
      typeof filePath === 'string' ? filePath : (filePath as { path: string }).path;

    if (!resolvedPath) {
      throw new Error('Failed to resolve save path from dialog');
    }

    const arrayBuffer = await blob.arrayBuffer();
    await writeFile(resolvedPath, new Uint8Array(arrayBuffer));
  },

  async openPath(path: string) {
    const { open } = await import('@tauri-apps/plugin-shell');
    await open(path);
  },

  async pickDirectory(title: string) {
    const { open } = await import('@tauri-apps/plugin-dialog');
    const selected = await open({ directory: true, title });
    if (!selected) return null;
    const dir = typeof selected === 'string' ? selected : (selected as { path: string }).path;
    return dir || null;
  },
};
