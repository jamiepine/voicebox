import { useEffect, useRef } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { useServerStore } from '@/stores/serverStore';
import type { FinetuneStartRequest, FinetuneStatusResponse } from '@/lib/api/types';

export function useFinetuneSamples(profileId: string) {
  return useQuery({
    queryKey: ['finetune', profileId, 'samples'],
    queryFn: () => apiClient.listFinetuneSamples(profileId),
    enabled: !!profileId,
  });
}

export function useAddFinetuneSample() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      profileId,
      file,
      transcript,
    }: {
      profileId: string;
      file: File;
      transcript?: string;
    }) => apiClient.addFinetuneSample(profileId, file, transcript),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ['finetune', variables.profileId, 'samples'],
      });
      queryClient.invalidateQueries({
        queryKey: ['finetune', variables.profileId, 'status'],
      });
    },
  });
}

export function useDeleteFinetuneSample() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ profileId, sampleId }: { profileId: string; sampleId: string }) =>
      apiClient.deleteFinetuneSample(profileId, sampleId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ['finetune', variables.profileId, 'samples'],
      });
      queryClient.invalidateQueries({
        queryKey: ['finetune', variables.profileId, 'status'],
      });
    },
  });
}

export function useImportProfileSamples() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      profileId,
      sampleIds,
    }: {
      profileId: string;
      sampleIds?: string[];
    }) => apiClient.importProfileSamples(profileId, sampleIds),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ['finetune', variables.profileId, 'samples'],
      });
      queryClient.invalidateQueries({
        queryKey: ['finetune', variables.profileId, 'status'],
      });
    },
  });
}

export function useSetRefAudio() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ profileId, sampleId }: { profileId: string; sampleId: string }) =>
      apiClient.setRefAudio(profileId, sampleId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ['finetune', variables.profileId, 'samples'],
      });
    },
  });
}

export function useFinetuneStatus(profileId: string) {
  const queryClient = useQueryClient();
  const eventSourceRef = useRef<EventSource | null>(null);

  const query = useQuery({
    queryKey: ['finetune', profileId, 'status'],
    queryFn: () => apiClient.getFinetuneStatus(profileId),
    enabled: !!profileId,
    staleTime: 10_000,
  });

  const activeJobId = query.data?.active_job?.id;

  useEffect(() => {
    if (!activeJobId) {
      // No active job — close any existing SSE connection
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      return;
    }

    let isActive = true;

    // Open SSE connection for real-time progress
    const serverUrl = useServerStore.getState().serverUrl;
    const es = new EventSource(`${serverUrl}/finetune/progress/${activeJobId}`);
    eventSourceRef.current = es;

    es.addEventListener('progress', (event) => {
      if (!isActive) return;
      try {
        const progress = JSON.parse(event.data);
        // Push SSE progress data into the React Query cache
        queryClient.setQueryData<FinetuneStatusResponse>(
          ['finetune', profileId, 'status'],
          (prev) => {
            if (!prev?.active_job) return prev;
            return {
              ...prev,
              active_job: {
                ...prev.active_job,
                current_step: progress.current ?? prev.active_job.current_step,
                total_steps: progress.total ?? prev.active_job.total_steps,
                current_loss: progress.current_loss,
                status: 'training',
              },
            };
          },
        );
      } catch {
        // Ignore malformed SSE data
      }
    });

    es.addEventListener('complete', () => {
      if (!isActive) return;
      es.close();
      eventSourceRef.current = null;
      // Refetch status and profiles to get final state
      queryClient.invalidateQueries({ queryKey: ['finetune', profileId, 'status'] });
      queryClient.invalidateQueries({ queryKey: ['finetune', profileId, 'jobs'] });
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
    });

    es.onerror = () => {
      if (!isActive) return;
      es.close();
      eventSourceRef.current = null;
      // Fall back to a one-time refetch after SSE drops
      queryClient.invalidateQueries({ queryKey: ['finetune', profileId, 'status'] });
    };

    return () => {
      isActive = false;
      es.close();
      eventSourceRef.current = null;
    };
  }, [activeJobId, profileId, queryClient]);

  return query;
}

export function useStartFinetune() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      profileId,
      config,
    }: {
      profileId: string;
      config?: FinetuneStartRequest;
    }) => apiClient.startFinetune(profileId, config),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ['finetune', variables.profileId, 'status'],
      });
      queryClient.invalidateQueries({
        queryKey: ['finetune', variables.profileId, 'jobs'],
      });
    },
  });
}

export function useCancelFinetune() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (profileId: string) => apiClient.cancelFinetune(profileId),
    onSuccess: (_, profileId) => {
      queryClient.invalidateQueries({
        queryKey: ['finetune', profileId, 'status'],
      });
      queryClient.invalidateQueries({
        queryKey: ['finetune', profileId, 'jobs'],
      });
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
    },
  });
}

export function useFinetuneJobs(profileId: string) {
  return useQuery({
    queryKey: ['finetune', profileId, 'jobs'],
    queryFn: () => apiClient.listFinetuneJobs(profileId),
    enabled: !!profileId,
  });
}

export function useAdapters(profileId: string) {
  return useQuery({
    queryKey: ['finetune', profileId, 'adapters'],
    queryFn: () => apiClient.listAdapters(profileId),
    enabled: !!profileId,
  });
}

export function useSetActiveAdapter() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ profileId, jobId }: { profileId: string; jobId: string | null }) =>
      apiClient.setActiveAdapter(profileId, jobId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['finetune', variables.profileId, 'adapters'] });
      queryClient.invalidateQueries({ queryKey: ['finetune', variables.profileId, 'status'] });
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
    },
  });
}

export function useUpdateAdapterLabel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      profileId,
      jobId,
      label,
    }: {
      profileId: string;
      jobId: string;
      label: string;
    }) => apiClient.updateAdapterLabel(profileId, jobId, label),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['finetune', variables.profileId, 'adapters'] });
      queryClient.invalidateQueries({ queryKey: ['finetune', variables.profileId, 'jobs'] });
    },
  });
}

export function useDeleteAdapter() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ profileId, jobId }: { profileId: string; jobId: string }) =>
      apiClient.deleteAdapter(profileId, jobId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['finetune', variables.profileId, 'adapters'] });
      queryClient.invalidateQueries({ queryKey: ['finetune', variables.profileId, 'status'] });
      queryClient.invalidateQueries({ queryKey: ['finetune', variables.profileId, 'jobs'] });
      queryClient.invalidateQueries({ queryKey: ['profiles'] });
    },
  });
}
