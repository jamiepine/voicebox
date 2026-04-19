import { useEffect } from 'react';
import type { UseFormReturn } from 'react-hook-form';
import { FormControl } from '@/components/ui/form';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { VoiceProfileResponse } from '@/lib/api/types';
import { getLanguageOptionsForEngine } from '@/lib/constants/languages';
import type { GenerationFormValues } from '@/lib/hooks/useGenerationForm';

/**
 * Engine/model options and their display metadata.
 * Adding a new engine means adding one entry here.
 */
const ENGINE_OPTIONS = [
  { value: 'qwen:1.7B', label: 'Qwen3-TTS 1.7B', engine: 'qwen' },
  { value: 'qwen:0.6B', label: 'Qwen3-TTS 0.6B', engine: 'qwen' },
  { value: 'qwen_custom_voice:1.7B', label: 'Qwen CustomVoice 1.7B', engine: 'qwen_custom_voice' },
  { value: 'qwen_custom_voice:0.6B', label: 'Qwen CustomVoice 0.6B', engine: 'qwen_custom_voice' },
  { value: 'luxtts', label: 'LuxTTS', engine: 'luxtts' },
  { value: 'chatterbox', label: 'Chatterbox', engine: 'chatterbox' },
  { value: 'chatterbox_turbo', label: 'Chatterbox Turbo', engine: 'chatterbox_turbo' },
  { value: 'tada:1B', label: 'TADA 1B', engine: 'tada' },
  { value: 'tada:3B', label: 'TADA 3B Multilingual', engine: 'tada' },
  { value: 'kokoro', label: 'Kokoro 82M', engine: 'kokoro' },
] as const;

const ENGINE_DESCRIPTIONS: Record<string, string> = {
  qwen: 'Multi-language, two sizes',
  qwen_custom_voice: '9 preset voices, instruct control',
  luxtts: 'Fast, English-focused',
  chatterbox: '23 languages, incl. Hebrew',
  chatterbox_turbo: 'English, [laugh] [cough] tags',
  tada: 'HumeAI, 700s+ coherent audio',
  kokoro: '82M params, CPU realtime, 8 langs',
};

/** Engines that only support English and should force language to 'en' on select. */
const ENGLISH_ONLY_ENGINES = new Set(['luxtts', 'chatterbox_turbo']);

/** Engines that support cloned (reference audio) profiles. */
const CLONING_ENGINES = new Set(['qwen', 'luxtts', 'chatterbox', 'chatterbox_turbo', 'tada']);

function getAvailableOptions(selectedProfile?: VoiceProfileResponse | null) {
  if (!selectedProfile) return ENGINE_OPTIONS;
  return ENGINE_OPTIONS.filter((opt) => isProfileCompatibleWithEngine(selectedProfile, opt.engine));
}

function getSelectValue(engine: string, modelSize?: string): string {
  if (engine === 'qwen') return `qwen:${modelSize || '1.7B'}`;
  if (engine === 'qwen_custom_voice') return `qwen_custom_voice:${modelSize || '1.7B'}`;
  if (engine === 'tada') return `tada:${modelSize || '1B'}`;
  return engine;
}

export function applyEngineSelection(form: UseFormReturn<GenerationFormValues>, value: string) {
  if (value.startsWith('qwen_custom_voice:')) {
    const [, modelSize] = value.split(':');
    form.setValue('engine', 'qwen_custom_voice');
    form.setValue('modelSize', modelSize as '1.7B' | '0.6B');
    const currentLang = form.getValues('language');
    const available = getLanguageOptionsForEngine('qwen_custom_voice');
    if (!available.some((l) => l.value === currentLang)) {
      form.setValue('language', available[0]?.value ?? 'en');
    }
  } else if (value.startsWith('qwen:')) {
    const [, modelSize] = value.split(':');
    form.setValue('engine', 'qwen');
    form.setValue('modelSize', modelSize as '1.7B' | '0.6B');
    // Validate language is supported by Qwen
    const currentLang = form.getValues('language');
    const available = getLanguageOptionsForEngine('qwen');
    if (!available.some((l) => l.value === currentLang)) {
      form.setValue('language', available[0]?.value ?? 'en');
    }
  } else if (value.startsWith('tada:')) {
    const [, modelSize] = value.split(':');
    form.setValue('engine', 'tada');
    form.setValue('modelSize', modelSize as '1B' | '3B');
    // TADA 1B is English-only; 3B is multilingual
    if (modelSize === '1B') {
      form.setValue('language', 'en');
    } else {
      const currentLang = form.getValues('language');
      const available = getLanguageOptionsForEngine('tada');
      if (!available.some((l) => l.value === currentLang)) {
        form.setValue('language', available[0]?.value ?? 'en');
      }
    }
  } else {
    form.setValue('engine', value as GenerationFormValues['engine']);
    form.setValue('modelSize', undefined as unknown as '1.7B' | '0.6B');
    if (ENGLISH_ONLY_ENGINES.has(value)) {
      form.setValue('language', 'en');
    } else {
      // If current language isn't supported by the new engine, reset to first available
      const currentLang = form.getValues('language');
      const available = getLanguageOptionsForEngine(value);
      if (!available.some((l) => l.value === currentLang)) {
        form.setValue('language', available[0]?.value ?? 'en');
      }
    }
  }
}

interface EngineModelSelectorProps {
  form: UseFormReturn<GenerationFormValues>;
  compact?: boolean;
  selectedProfile?: VoiceProfileResponse | null;
}

export function EngineModelSelector({ form, compact, selectedProfile }: EngineModelSelectorProps) {
  const engine = form.watch('engine') || 'qwen';
  const modelSize = form.watch('modelSize');
  const selectValue = getSelectValue(engine, modelSize);
  const availableOptions = getAvailableOptions(selectedProfile);

  const currentEngineAvailable = availableOptions.some((opt) => opt.value === selectValue);

  useEffect(() => {
    if (!currentEngineAvailable && availableOptions.length > 0) {
      applyEngineSelection(form, availableOptions[0].value);
    }
  }, [availableOptions, currentEngineAvailable, form]);

  const itemClass = compact ? 'text-xs text-muted-foreground' : undefined;
  const triggerClass = compact
    ? 'h-8 text-xs bg-card border-border rounded-full hover:bg-background/50 transition-all'
    : undefined;

  return (
    <Select value={selectValue} onValueChange={(v) => applyEngineSelection(form, v)}>
      <FormControl>
        <SelectTrigger className={triggerClass}>
          <SelectValue />
        </SelectTrigger>
      </FormControl>
      <SelectContent>
        {availableOptions.map((opt) => (
          <SelectItem key={opt.value} value={opt.value} className={itemClass}>
            {opt.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

/** Returns a human-readable description for the currently selected engine. */
export function getEngineDescription(engine: string): string {
  return ENGINE_DESCRIPTIONS[engine] ?? '';
}

/**
 * Check if a profile is compatible with the currently selected engine.
 * Useful for UI hints.
 */
export function isProfileCompatibleWithEngine(
  profile: VoiceProfileResponse,
  engine: string,
): boolean {
  const voiceType = profile.voice_type || 'cloned';
  if (voiceType === 'preset') return profile.preset_engine === engine;
  if (voiceType === 'cloned') return CLONING_ENGINES.has(engine);
  return true; // designed — future
}
