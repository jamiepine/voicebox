import { useQueries } from '@tanstack/react-query';
import { Check, ChevronDown, Loader2, Search } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { PresetVoice, VoiceProfileResponse } from '@/lib/api/types';
import type { LanguageCode } from '@/lib/constants/languages';
import { useCreateProfile, useProfiles } from '@/lib/hooks/useProfiles';
import { useVoicePreview, type VoicePreviewState } from '@/lib/hooks/useVoicePreview';
import { cn } from '@/lib/utils/cn';
import { useUIStore } from '@/stores/uiStore';
import { VoicePreviewButton } from './VoicePreviewButton';

/** Engines that ship ready-made voices, in the order they are listed. */
const PRESET_ENGINES = [
  { engine: 'kokoro', label: 'Kokoro' },
  { engine: 'qwen_custom_voice', label: 'Qwen CustomVoice' },
] as const;

/** Preview clips are cached per voice, so a built-in voice and its profile share one. */
const presetKey = (engine: string, voiceId: string) => `preset:${engine}:${voiceId}`;

function profilePreviewKey(profile: VoiceProfileResponse) {
  if (profile.voice_type === 'preset' && profile.preset_engine && profile.preset_voice_id) {
    return presetKey(profile.preset_engine, profile.preset_voice_id);
  }
  // Editing a cloned or designed voice changes how it sounds.
  return `profile:${profile.id}:${profile.updated_at}`;
}

function loadProfilePreview(profile: VoiceProfileResponse) {
  if (profile.voice_type === 'preset' && profile.preset_engine && profile.preset_voice_id) {
    return apiClient.previewPresetVoice(profile.preset_engine, profile.preset_voice_id);
  }
  return apiClient.previewProfileVoice(profile.id);
}

/** Profile names are unique, so a built-in voice added from the picker may need a suffix. */
function uniqueProfileName(base: string, engineLabel: string, profiles: VoiceProfileResponse[]) {
  const taken = new Set(profiles.map((p) => p.name.toLowerCase()));
  if (!taken.has(base.toLowerCase())) return base;
  const withEngine = `${base} (${engineLabel})`;
  if (!taken.has(withEngine.toLowerCase())) return withEngine;
  for (let n = 2; ; n++) {
    const candidate = `${withEngine} ${n}`;
    if (!taken.has(candidate.toLowerCase())) return candidate;
  }
}

interface VoiceRowProps {
  name: string;
  badges: string[];
  selected: boolean;
  busy?: boolean;
  previewState: VoicePreviewState;
  onChoose: () => void;
  onTogglePreview: () => void;
}

function VoiceRow({
  name,
  badges,
  selected,
  busy,
  previewState,
  onChoose,
  onTogglePreview,
}: VoiceRowProps) {
  return (
    <div
      className={cn(
        'flex items-center gap-1 rounded-md transition-colors hover:bg-muted',
        selected && 'bg-accent/10',
      )}
    >
      <button
        type="button"
        onClick={onChoose}
        disabled={busy}
        className="flex min-w-0 flex-1 items-center gap-2 px-2 py-1.5 text-left"
      >
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs font-medium">{name}</span>
          <span className="mt-0.5 flex gap-1">
            {badges.map((badge) => (
              <Badge key={badge} variant="outline" className="h-4 px-1 text-[10px] font-normal">
                {badge}
              </Badge>
            ))}
          </span>
        </span>
        {busy ? (
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-muted-foreground" />
        ) : selected ? (
          <Check className="h-3.5 w-3.5 shrink-0 text-accent" />
        ) : null}
      </button>
      <VoicePreviewButton
        name={name}
        state={previewState}
        onToggle={onTogglePreview}
        className="mr-1"
      />
    </div>
  );
}

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-2 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
      {children}
    </div>
  );
}

/**
 * Chooses the voice to generate with, and lets you hear it first.
 *
 * Lists your own voices plus every built-in voice. Picking a built-in voice
 * adds it to your voices (generation is always tied to a voice profile) and
 * selects it, so there is no separate setup step.
 */
export function VoicePicker() {
  const { t } = useTranslation();
  const { toast } = useToast();
  const selectedProfileId = useUIStore((state) => state.selectedProfileId);
  const setSelectedProfileId = useUIStore((state) => state.setSelectedProfileId);
  const { data: profiles = [] } = useProfiles();
  const createProfile = useCreateProfile();
  const previews = useVoicePreview();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [addingKey, setAddingKey] = useState<string | null>(null);

  const presetQueries = useQueries({
    queries: PRESET_ENGINES.map(({ engine }) => ({
      queryKey: ['presetVoices', engine],
      queryFn: () => apiClient.listPresetVoices(engine),
      staleTime: Number.POSITIVE_INFINITY,
    })),
  });

  const selectedProfile = profiles.find((p) => p.id === selectedProfileId);

  const needle = query.trim().toLowerCase();
  const matches = (...fields: (string | undefined)[]) =>
    !needle || fields.some((field) => field?.toLowerCase().includes(needle));

  const visibleProfiles = profiles.filter((p) => matches(p.name, p.language, p.description));

  function handleOpenChange(next: boolean) {
    setOpen(next);
    if (!next) {
      previews.stop();
      setQuery('');
    }
  }

  function choose(profileId: string) {
    setSelectedProfileId(profileId);
    handleOpenChange(false);
  }

  async function chooseBuiltIn(engine: string, engineLabel: string, voice: PresetVoice) {
    const existing = profiles.find(
      (p) =>
        p.voice_type === 'preset' &&
        p.preset_engine === engine &&
        p.preset_voice_id === voice.voice_id,
    );
    if (existing) {
      choose(existing.id);
      return;
    }

    setAddingKey(presetKey(engine, voice.voice_id));
    try {
      const created = await createProfile.mutateAsync({
        name: uniqueProfileName(voice.name, engineLabel, profiles),
        language: voice.language as LanguageCode,
        voice_type: 'preset',
        preset_engine: engine,
        preset_voice_id: voice.voice_id,
        default_engine: engine,
      });
      choose(created.id);
    } catch (err) {
      toast({
        title: t('generation.voicePicker.addFailed'),
        description: err instanceof Error ? err.message : undefined,
        variant: 'destructive',
      });
    } finally {
      setAddingKey(null);
    }
  }

  const stillLoading = presetQueries.some((q) => q.isLoading);
  const nothingToShow =
    visibleProfiles.length === 0 &&
    presetQueries.every(
      (q) => !(q.data?.voices ?? []).some((v) => matches(v.name, v.gender, v.language)),
    );

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="flex h-8 w-full items-center justify-between gap-2 rounded-full border border-border bg-card px-3 text-xs transition-all hover:bg-background/50"
        >
          <span className={cn('truncate', !selectedProfile && 'text-muted-foreground')}>
            {selectedProfile?.name ?? t('generation.voiceSelector.placeholder')}
          </span>
          <ChevronDown className="h-3.5 w-3.5 shrink-0 opacity-50" />
        </button>
      </PopoverTrigger>
      <PopoverContent side="top" align="start" className="w-[22rem] p-0">
        <div className="relative border-b border-border p-2">
          <Search className="pointer-events-none absolute left-4 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={t('generation.voicePicker.searchPlaceholder')}
            aria-label={t('generation.voicePicker.searchPlaceholder')}
            className="h-8 pl-8 text-xs"
          />
        </div>

        <div className="max-h-[22rem] overflow-y-auto p-1">
          {visibleProfiles.length > 0 && (
            <>
              <SectionHeader>{t('generation.voicePicker.myVoices')}</SectionHeader>
              {visibleProfiles.map((profile) => {
                const key = profilePreviewKey(profile);
                return (
                  <VoiceRow
                    key={profile.id}
                    name={profile.name}
                    badges={[
                      profile.language,
                      t(`generation.voicePicker.type.${profile.voice_type}`),
                    ]}
                    selected={profile.id === selectedProfileId}
                    previewState={previews.stateOf(key)}
                    onChoose={() => choose(profile.id)}
                    onTogglePreview={() => previews.toggle(key, () => loadProfilePreview(profile))}
                  />
                );
              })}
            </>
          )}

          {PRESET_ENGINES.map(({ engine, label }, index) => {
            const voices = (presetQueries[index].data?.voices ?? []).filter((v) =>
              matches(v.name, v.gender, v.language),
            );
            if (voices.length === 0) return null;
            return (
              <div key={engine}>
                <SectionHeader>
                  {t('generation.voicePicker.builtIn', { engine: label })}
                </SectionHeader>
                {voices.map((voice) => {
                  const key = presetKey(engine, voice.voice_id);
                  return (
                    <VoiceRow
                      key={voice.voice_id}
                      name={voice.name}
                      badges={[voice.gender, voice.language]}
                      selected={
                        selectedProfile?.preset_engine === engine &&
                        selectedProfile.preset_voice_id === voice.voice_id
                      }
                      busy={addingKey === key}
                      previewState={previews.stateOf(key)}
                      onChoose={() => chooseBuiltIn(engine, label, voice)}
                      onTogglePreview={() =>
                        previews.toggle(key, () =>
                          apiClient.previewPresetVoice(engine, voice.voice_id),
                        )
                      }
                    />
                  );
                })}
              </div>
            );
          })}

          {stillLoading && (
            <div className="flex items-center justify-center gap-2 py-4 text-xs text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              {t('generation.voicePicker.loading')}
            </div>
          )}
          {!stillLoading && nothingToShow && (
            <div className="py-6 text-center text-xs text-muted-foreground">
              {t('generation.voicePicker.noResults', { query: query.trim() })}
            </div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
