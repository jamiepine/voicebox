import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

/**
 * Shared hook for fetching model status and splitting models into
 * built-in and custom groups.
 *
 * Used by GenerationForm, FloatingGenerateBox, and ModelManagement
 * so the query key, refetch interval, and filtering logic stay
 * consistent in one place.
 *
 * @author AJ - Kamyab (Ankit Jain) — Extracted from inline useQuery calls
 */
export function useModelStatus() {
    const { data: modelStatus, ...rest } = useQuery({
        queryKey: ['modelStatus'],
        queryFn: () => apiClient.getModelStatus(),
        refetchInterval: 10000,
    });

    const builtInModels = useMemo(
        () => modelStatus?.models.filter((m) => m.model_name.startsWith('qwen-tts')) ?? [],
        [modelStatus],
    );
    const customModels = useMemo(
        () => modelStatus?.models.filter((m) => m.is_custom) ?? [],
        [modelStatus],
    );

    return { modelStatus, builtInModels, customModels, ...rest };
}
