import { zodResolver } from '@hookform/resolvers/zod';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { EffectConfig } from '@/lib/api/types';
import { LANGUAGE_CODES, type LanguageCode } from '@/lib/constants/languages';
import { useGeneration } from '@/lib/hooks/useGeneration';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';
import { useGenerationStore } from '@/stores/generationStore';
import { useServerStore } from '@/stores/serverStore';
import { useUIStore } from '@/stores/uiStore';

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
  text: z.string().min(1, '').max(50000),
  language: z.enum(LANGUAGE_CODES as [LanguageCode, ...LanguageCode[]]),
  seed: z.number().int().optional(),
  modelSize: z.string().regex(
    /^(1\.7B|0\.6B|1B|3B|custom:[a-z0-9][a-z0-9._-]*[a-z0-9])$/,
    'Must be a built-in size (1.7B, 0.6B, 1B, 3B) or custom model (custom:<slug>)',
  ).optional(),
  instruct: z.string().max(500).optional(),
  engine: z
    .enum([
      'qwen',
      'qwen_custom_voice',
      'luxtts',
      'chatterbox',
      'chatterbox_turbo',
      'tada',
      'kokoro',
    ])
    .optional(),
});

export type GenerationFormValues = z.infer<typeof generationSchema>;

interface UseGenerationFormOptions {
  onSuccess?: (generationId: string) => void;
  defaultValues?: Partial<GenerationFormValues>;
  getEffectsChain?: () => EffectConfig[] | undefined;
}

export function useGenerationForm(options: UseGenerationFormOptions = {}) {
  const { toast } = useToast();
  const generation = useGeneration();
  const addPendingGeneration = useGenerationStore((state) => state.addPendingGeneration);
  const maxChunkChars = useServerStore((state) => state.maxChunkChars);
  const crossfadeMs = useServerStore((state) => state.crossfadeMs);
  const normalizeAudio = useServerStore((state) => state.normalizeAudio);
  const selectedEngine = useUIStore((state) => state.selectedEngine);
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
      engine: (selectedEngine as GenerationFormValues['engine']) || 'qwen',
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
      const engine = data.engine || 'qwen';
      const modelSize = data.modelSize || '1.7B';
      let modelName = '';
      let displayName = '';

      if (modelSize.startsWith('custom:')) {
        modelName = modelSize;
        displayName = modelSize.replace('custom:', '');
      } else {
        modelName =
          engine === 'luxtts'
            ? 'luxtts'
            : engine === 'chatterbox'
              ? 'chatterbox-tts'
              : engine === 'chatterbox_turbo'
                ? 'chatterbox-turbo'
                : engine === 'tada'
                  ? modelSize === '3B'
                    ? 'tada-3b-ml'
                    : 'tada-1b'
                  : engine === 'kokoro'
                    ? 'kokoro'
                    : engine === 'qwen_custom_voice'
                      ? `qwen-custom-voice-${modelSize}`
                      : `qwen-tts-${modelSize}`;
        displayName =
          engine === 'luxtts'
            ? 'LuxTTS'
            : engine === 'chatterbox'
              ? 'Chatterbox TTS'
              : engine === 'chatterbox_turbo'
                ? 'Chatterbox Turbo'
                : engine === 'tada'
                  ? modelSize === '3B'
                    ? 'TADA 3B Multilingual'
                    : 'TADA 1B'
                  : engine === 'kokoro'
                    ? 'Kokoro 82M'
                    : engine === 'qwen_custom_voice'
                      ? modelSize === '1.7B'
                        ? 'Qwen CustomVoice 1.7B'
                        : 'Qwen CustomVoice 0.6B'
                      : modelSize === '1.7B'
                        ? 'Qwen TTS 1.7B'
                        : 'Qwen TTS 0.6B';
      }

      // Check if model needs downloading
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

      const hasModelSizes =
        engine === 'qwen' || engine === 'qwen_custom_voice' || engine === 'tada';
      // Only Qwen CustomVoice actually honors the instruct kwarg at model level.
      // Base Qwen3-TTS accepts the kwarg but ignores it.
      const supportsInstruct = engine === 'qwen_custom_voice';
      const effectsChain = options.getEffectsChain?.();
      // This now returns immediately with status="generating"
      const result = await generation.mutateAsync({
        profile_id: selectedProfileId,
        text: data.text,
        language: data.language,
        seed: data.seed,
        model_size: (hasModelSizes || data.modelSize?.startsWith('custom:')) ? data.modelSize : undefined,
        engine,
        instruct: supportsInstruct ? data.instruct || undefined : undefined,
        max_chunk_chars: maxChunkChars,
        crossfade_ms: crossfadeMs,
        normalize: normalizeAudio,
        effects_chain: effectsChain?.length ? effectsChain : undefined,
      });

      // Track this generation for SSE status updates
      addPendingGeneration(result.id);

      // Reset form immediately — user can start typing again
      form.reset({
        text: '',
        language: data.language,
        seed: undefined,
        modelSize: data.modelSize,
        instruct: '',
        engine: data.engine,
      });
      options.onSuccess?.(result.id);
    } catch (error) {
      toast({
        title: 'Generation failed',
        description: error instanceof Error ? error.message : 'Failed to generate audio',
        variant: 'destructive',
      });
    } finally {
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
