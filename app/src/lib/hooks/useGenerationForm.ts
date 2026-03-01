import { zodResolver } from '@hookform/resolvers/zod';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { LANGUAGE_CODES, type LanguageCode } from '@/lib/constants/languages';
import { useGeneration } from '@/lib/hooks/useGeneration';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';

/**
 * Zod schema for the generation form.
 *
 * `modelSize` is a free-form string rather than a strict enum
 * because it can be either a built-in size ("1.7B", "0.6B") or
 * a custom model identifier ("custom:<slug>").
 *
 * @modified AJ - Kamyab (Ankit Jain) — Changed modelSize from enum to string for custom model support
 */
const generationSchema = z.object({
  text: z.string().min(1, 'Text is required').max(5000),
  language: z.enum(LANGUAGE_CODES as [LanguageCode, ...LanguageCode[]]),
  seed: z.number().int().optional(),
  modelSize: z.string().optional(),
  instruct: z.string().max(500).optional(),
});

export type GenerationFormValues = z.infer<typeof generationSchema>;

interface UseGenerationFormOptions {
  onSuccess?: (generationId: string) => void;
  defaultValues?: Partial<GenerationFormValues>;
}

export function useGenerationForm(options: UseGenerationFormOptions = {}) {
  const { toast } = useToast();
  const generation = useGeneration();
  const setAudioWithAutoPlay = usePlayerStore((state) => state.setAudioWithAutoPlay);
  const setIsGenerating = useGenerationStore((state) => state.setIsGenerating);
  const [downloadingModelName, setDownloadingModelName] = useState<string | null>(null);
  const [downloadingDisplayName, setDownloadingDisplayName] = useState<string | null>(null);

  useModelDownloadToast({
    modelName: downloadingModelName || '',
    displayName: downloadingDisplayName || '',
    enabled: !!downloadingModelName,
  });

  const form = useForm<GenerationFormValues>({
    resolver: zodResolver(generationSchema),
    defaultValues: {
      text: '',
      language: 'en',
      seed: undefined,
      modelSize: '1.7B',
      instruct: '',
      ...options.defaultValues,
    },
  });

  async function handleSubmit(
    data: GenerationFormValues,
    selectedProfileId: string | null,
  ): Promise<void> {
    if (!selectedProfileId) {
      toast({
        title: 'No profile selected',
        description: 'Please select a voice profile from the cards above.',
        variant: 'destructive',
      });
      return;
    }

    try {
      setIsGenerating(true);

      const modelSize = data.modelSize || '1.7B';

      // Derive model tracking name and display name.
      // Built-in models use "qwen-tts-<size>" format for tracking.
      // Custom models use the full "custom:<slug>" identifier as-is.
      let modelName: string;
      let displayName: string;

      if (modelSize.startsWith('custom:')) {
        // Custom model: use the full "custom:slug" as the tracking key
        modelName = modelSize;
        displayName = modelSize.replace('custom:', '');
      } else {
        // Built-in model: construct the standard tracking name
        modelName = `qwen-tts-${modelSize}`;
        displayName = modelSize === '1.7B' ? 'Qwen TTS 1.7B' : 'Qwen TTS 0.6B';
      }

      // Pre-flight check: query model status to get the accurate display name
      // and to detect if the model needs downloading first.
      // If the model isn't downloaded yet, enable the SSE download progress toast.
      try {
        const modelStatus = await apiClient.getModelStatus();
        const model = modelStatus.models.find((m) => m.model_name === modelName);

        if (model) {
          displayName = model.display_name;
          if (!model.downloaded) {
            // Not yet downloaded — enable progress tracking UI
            setDownloadingModelName(modelName);
            setDownloadingDisplayName(displayName);
          }
        }
      } catch (error) {
        // Non-fatal: generation will still attempt and may trigger download on the backend
        console.error('Failed to check model status:', error);
      }

      const result = await generation.mutateAsync({
        profile_id: selectedProfileId,
        text: data.text,
        language: data.language,
        seed: data.seed,
        model_size: modelSize,
        instruct: data.instruct || undefined,
      });

      toast({
        title: 'Generation complete!',
        description: `Audio generated (${result.duration.toFixed(2)}s)`,
      });

      const audioUrl = apiClient.getAudioUrl(result.id);
      setAudioWithAutoPlay(audioUrl, result.id, selectedProfileId, data.text.substring(0, 50));

      form.reset();
      options.onSuccess?.(result.id);
    } catch (error) {
      toast({
        title: 'Generation failed',
        description: error instanceof Error ? error.message : 'Failed to generate audio',
        variant: 'destructive',
      });
    } finally {
      setIsGenerating(false);
      setDownloadingModelName(null);
      setDownloadingDisplayName(null);
    }
  }

  return {
    form,
    handleSubmit,
    isPending: generation.isPending,
  };
}
