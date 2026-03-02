import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

/**
 * Shared hook for fetching model status and splitting models into
 * built-in and custom groups.
 *
 * Used by GenerationForm and FloatingGenerateBox so the query key,
 * refetch interval, and filtering logic stay consistent in one place.
 *
 * @author AJ - Kamyab (Ankit Jain) — Extracted from inline useQuery calls
 */
export function useModelStatus() {
    const { data: modelStatus, ...rest } = useQuery({
        queryKey: ['modelStatus'],
        queryFn: () => apiClient.getModelStatus(),
        refetchInterval: 10000,
    });

    const builtInModels =
        modelStatus?.models.filter((m) => m.model_name.startsWith('qwen-tts')) || [];
    const customModels =
        modelStatus?.models.filter((m) => m.is_custom) || [];

    return { modelStatus, builtInModels, customModels, ...rest };
}
