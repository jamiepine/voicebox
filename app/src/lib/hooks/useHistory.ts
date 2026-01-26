import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import type { HistoryQuery } from '@/lib/api/types';
import { isTauri } from '@/lib/tauri';

export function useHistory(query?: HistoryQuery) {
  return useQuery({
    queryKey: ['history', query],
    queryFn: () => apiClient.listHistory(query),
  });
}

export function useGenerationDetail(generationId: string) {
  return useQuery({
    queryKey: ['history', generationId],
    queryFn: () => apiClient.getGeneration(generationId),
    enabled: !!generationId,
  });
}

export function useDeleteGeneration() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (generationId: string) => apiClient.deleteGeneration(generationId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['history'] });
    },
  });
}

export function useExportGeneration() {
  return useMutation({
    mutationFn: async ({ generationId, text }: { generationId: string; text: string }) => {
      const blob = await apiClient.exportGeneration(generationId);
      
      // Create safe filename from text
      const safeText = text.substring(0, 30).replace(/[^a-z0-9]/gi, '-').toLowerCase();
      const filename = `generation-${safeText}.voicebox.zip`;
      
      if (isTauri()) {
        // Use Tauri's native save dialog
        try {
          const { save } = await import('@tauri-apps/plugin-dialog');
          const filePath = await save({
            defaultPath: filename,
            filters: [
              {
                name: 'Voicebox Generation',
                extensions: ['voicebox.zip', 'zip'],
              },
            ],
          });
          
          if (filePath) {
            // Write file using Tauri's filesystem API
            const { writeBinaryFile } = await import('@tauri-apps/plugin-fs');
            const arrayBuffer = await blob.arrayBuffer();
            await writeBinaryFile(filePath, new Uint8Array(arrayBuffer));
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
      } else {
        // Browser: trigger download
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
      
      return blob;
    },
  });
}

export function useExportGenerationAudio() {
  return useMutation({
    mutationFn: async ({ generationId, text }: { generationId: string; text: string }) => {
      const blob = await apiClient.exportGenerationAudio(generationId);
      
      // Create safe filename from text
      const safeText = text.substring(0, 30).replace(/[^a-z0-9]/gi, '-').toLowerCase();
      const filename = `${safeText}.wav`;
      
      if (isTauri()) {
        // Use Tauri's native save dialog
        try {
          const { save } = await import('@tauri-apps/plugin-dialog');
          const filePath = await save({
            defaultPath: filename,
            filters: [
              {
                name: 'Audio File',
                extensions: ['wav'],
              },
            ],
          });
          
          if (filePath) {
            // Write file using Tauri's filesystem API
            const { writeBinaryFile } = await import('@tauri-apps/plugin-fs');
            const arrayBuffer = await blob.arrayBuffer();
            await writeBinaryFile(filePath, new Uint8Array(arrayBuffer));
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
      } else {
        // Browser: trigger download
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
      
      return blob;
    },
  });
}

export function useImportGeneration() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => apiClient.importGeneration(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['history'] });
    },
  });
}
