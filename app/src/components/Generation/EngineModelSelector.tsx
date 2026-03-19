import type { UseFormReturn } from 'react-hook-form';
import { FormControl } from '@/components/ui/form';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { getLanguageOptionsForEngine } from '@/lib/constants/languages';
import type { GenerationFormValues } from '@/lib/hooks/useGenerationForm';

/**
 * Engine/model options and their display metadata.
 * Adding a new engine means adding one entry here.
 */
const ENGINE_OPTIONS = [
  { value: 'qwen:1.7B', label: 'Qwen3-TTS 1.7B' },
  { value: 'qwen:0.6B', label: 'Qwen3-TTS 0.6B' },
  { value: 'luxtts', label: 'LuxTTS' },
  { value: 'chatterbox', label: 'Chatterbox' },
  { value: 'chatterbox_turbo', label: 'Chatterbox Turbo' },
  { value: 'tada:1B', label: 'TADA 1B' },
  { value: 'tada:3B', label: 'TADA 3B Multilingual' },
  { value: 'cosyvoice:v2', label: 'CosyVoice2 0.5B' },
  { value: 'cosyvoice:v3', label: 'CosyVoice3 0.5B' },
] as const;

const ENGINE_DESCRIPTIONS: Record<string, string> = {
  qwen: 'Multi-language, two sizes',
  luxtts: 'Fast, English-focused',
  chatterbox: '23 languages, incl. Hebrew',
  chatterbox_turbo: 'English, [laugh] [cough] tags',
  tada: 'HumeAI, 700s+ coherent audio',
  cosyvoice: 'Alibaba, instruct + cloning',
};

/** Engines that only support English and should force language to 'en' on select. */
const ENGLISH_ONLY_ENGINES = new Set(['luxtts', 'chatterbox_turbo']);

function getSelectValue(engine: string, modelSize?: string): string {
  if (engine === 'qwen') return `qwen:${modelSize || '1.7B'}`;
  if (engine === 'tada') return `tada:${modelSize || '1B'}`;
  if (engine === 'cosyvoice') return `cosyvoice:${modelSize || 'v2'}`;
  return engine;
}

function handleEngineChange(form: UseFormReturn<GenerationFormValues>, value: string) {
  if (value.startsWith('qwen:')) {
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
  } else if (value.startsWith('cosyvoice:')) {
    const [, modelSize] = value.split(':');
    form.setValue('engine', 'cosyvoice');
    form.setValue('modelSize', modelSize as 'v2' | 'v3');
    const currentLang = form.getValues('language');
    const available = getLanguageOptionsForEngine('cosyvoice');
    if (!available.some((l) => l.value === currentLang)) {
      form.setValue('language', available[0]?.value ?? 'en');
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
}

export function EngineModelSelector({ form, compact }: EngineModelSelectorProps) {
  const engine = form.watch('engine') || 'qwen';
  const modelSize = form.watch('modelSize');
  const selectValue = getSelectValue(engine, modelSize);

  const itemClass = compact ? 'text-xs text-muted-foreground' : undefined;
  const triggerClass = compact
    ? 'h-8 text-xs bg-card border-border rounded-full hover:bg-background/50 transition-all'
    : undefined;

  return (
    <Select value={selectValue} onValueChange={(v) => handleEngineChange(form, v)}>
      <FormControl>
        <SelectTrigger className={triggerClass}>
          <SelectValue />
        </SelectTrigger>
      </FormControl>
      <SelectContent>
        {ENGINE_OPTIONS.map((opt) => (
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
