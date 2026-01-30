import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import type { VoiceProfileCreate } from '@/lib/api/types';
import { isTauri } from '@/lib/tauri';

export function useProfiles() {
  return useQuery({
    queryKey: ['profiles'],
    queryFn: () => apiClient.listProfiles(),
  });
}

export function useProfile(profileId: string) {
  return useQuery({
    queryKey: ['profiles', profileId],
    queryFn: () => apiClient.getProfile(profileId),
    enabled: !!profileId,
  });
}

export function useCreateProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: VoiceProfileCreate) => apiClient.createProfile(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
    },
  });
}

export function useUpdateProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ profileId, data }: { profileId: string; data: VoiceProfileCreate }) =>
      apiClient.updateProfile(profileId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      queryClient.invalidateQueries({
        queryKey: ['profiles', variables.profileId],
      });
    },
  });
}

export function useDeleteProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (profileId: string) => apiClient.deleteProfile(profileId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
    },
  });
}

export function useProfileSamples(profileId: string) {
  return useQuery({
    queryKey: ['profiles', profileId, 'samples'],
    queryFn: () => apiClient.listProfileSamples(profileId),
    enabled: !!profileId,
  });
}

export function useAddSample() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      profileId,
      file,
      referenceText,
    }: {
      profileId: string;
      file: File;
      referenceText: string;
    }) => apiClient.addProfileSample(profileId, file, referenceText),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ['profiles', variables.profileId, 'samples'],
      });
      queryClient.invalidateQueries({
        queryKey: ['profiles', variables.profileId],
      });
    },
  });
}

export function useDeleteSample() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (sampleId: string) => apiClient.deleteProfileSample(sampleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
    },
  });
}

export function useUpdateSample() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ sampleId, referenceText }: { sampleId: string; referenceText: string }) =>
      apiClient.updateProfileSample(sampleId, referenceText),
    onSuccess: (data) => {
      queryClient.invalidateQueries({
        queryKey: ['profiles', data.profile_id, 'samples'],
      });
      queryClient.invalidateQueries({
        queryKey: ['profiles', data.profile_id],
      });
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
    },
  });
}

export function useExportProfile() {
  return useMutation({
    mutationFn: async (profileId: string) => {
      const blob = await apiClient.exportProfile(profileId);

      // Get profile name for filename
      const profile = await apiClient.getProfile(profileId);
      const safeName = profile.name.replace(/[^a-z0-9]/gi, '-').toLowerCase();
      const filename = `profile-${safeName}.voicebox.zip`;

      if (isTauri()) {
        // Use Tauri's native save dialog
        try {
          const { save } = await import('@tauri-apps/plugin-dialog');
          const filePath = await save({
            defaultPath: filename,
            filters: [
              {
                name: 'Voicebox Profile',
                extensions: ['zip'],
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

export function useImportProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => apiClient.importProfile(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
    },
  });
}

export function useUploadAvatar() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ profileId, file }: { profileId: string; file: File }) =>
      apiClient.uploadAvatar(profileId, file),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      queryClient.invalidateQueries({
        queryKey: ['profiles', variables.profileId],
      });
    },
  });
}

export function useDeleteAvatar() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (profileId: string) => apiClient.deleteAvatar(profileId),
    onSuccess: (_, profileId) => {
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
      queryClient.invalidateQueries({
        queryKey: ['profiles', profileId],
      });
    },
  });
}
