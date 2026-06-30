import { useQuery } from '@tanstack/react-query';
import { Loader2, Plus, Sparkles, Wand2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import {
  ListPane,
  ListPaneActions,
  ListPaneHeader,
  ListPaneScroll,
  ListPaneTitle,
  ListPaneTitleRow,
} from '@/components/ListPane';
import { Button } from '@/components/ui/button';
import { apiClient } from '@/lib/api/client';
import type { EffectPresetResponse } from '@/lib/api/types';
import { cn } from '@/lib/utils/cn';
import { useEffectsStore } from '@/stores/effectsStore';

export function EffectsList() {
  const { t } = useTranslation();
  const selectedPresetId = useEffectsStore((s) => s.selectedPresetId);
  const setSelectedPresetId = useEffectsStore((s) => s.setSelectedPresetId);
  const setWorkingChain = useEffectsStore((s) => s.setWorkingChain);
  const setIsCreatingNew = useEffectsStore((s) => s.setIsCreatingNew);
  const isCreatingNew = useEffectsStore((s) => s.isCreatingNew);

  const { data: presets, isLoading } = useQuery({
    queryKey: ['effect-presets'],
    queryFn: () => apiClient.listEffectPresets(),
    staleTime: 30_000,
  });

  const builtIn = presets?.filter((p) => p.is_builtin) ?? [];
  const userPresets = presets?.filter((p) => !p.is_builtin) ?? [];

  function handleSelect(preset: EffectPresetResponse) {
    setSelectedPresetId(preset.id);
    setWorkingChain(preset.effects_chain);
  }

  function handleCreateNew() {
    setIsCreatingNew(true);
    setWorkingChain([]);
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <ListPane>
      <ListPaneHeader>
        <ListPaneTitleRow>
          <ListPaneTitle>{t('effects.title')}</ListPaneTitle>
          <ListPaneActions>
            <Button onClick={handleCreateNew} size="sm">
              <Plus className="mr-2 h-4 w-4" />
              {t('effects.newPreset')}
            </Button>
          </ListPaneActions>
        </ListPaneTitleRow>
      </ListPaneHeader>

      <ListPaneScroll className="pt-16">
        <div className="px-4 pb-6 space-y-4">
          {builtIn.length > 0 && (
            <div>
              <div className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider mb-2 px-1">
                {t('effects.sections.builtin')}
              </div>
              <div className="space-y-1.5">
                {builtIn.map((preset) => (
                  <PresetCard
                    key={preset.id}
                    preset={preset}
                    isSelected={selectedPresetId === preset.id && !isCreatingNew}
                    onSelect={() => handleSelect(preset)}
                  />
                ))}
              </div>
            </div>
          )}

          {userPresets.length > 0 && (
            <div>
              <div className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider mb-2 px-1">
                {t('effects.sections.custom')}
              </div>
              <div className="space-y-1.5">
                {userPresets.map((preset) => (
                  <PresetCard
                    key={preset.id}
                    preset={preset}
                    isSelected={selectedPresetId === preset.id && !isCreatingNew}
                    onSelect={() => handleSelect(preset)}
                  />
                ))}
              </div>
            </div>
          )}

          {isCreatingNew && (
            <div>
              <div className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider mb-2 px-1">
                {t('effects.sections.new')}
              </div>
              <div className="rounded-xl border-2 border-accent/40 bg-accent/5 p-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-accent" />
                  <span className="text-sm font-medium">{t('effects.unsaved.title')}</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">{t('effects.unsaved.hint')}</p>
              </div>
            </div>
          )}
        </div>
      </ListPaneScroll>
    </ListPane>
  );
}

function PresetCard({
  preset,
  isSelected,
  onSelect,
}: {
  preset: EffectPresetResponse;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const { t } = useTranslation();
  const effectCount = preset.effects_chain.length;
  const name = preset.is_builtin
    ? t(`effects.builtinPresets.${preset.name}.name`, { defaultValue: preset.name })
    : preset.name;
  const description = preset.is_builtin
    ? t(`effects.builtinPresets.${preset.name}.description`, {
        defaultValue: preset.description ?? '',
      })
    : preset.description;

  return (
    <button
      type="button"
      className={cn(
        'w-full text-left rounded-xl border p-3 h-[88px] transition-all duration-150',
        isSelected
          ? 'border-accent/50 bg-accent/10'
          : 'border-border bg-card hover:bg-muted/50 hover:border-border',
      )}
      onClick={onSelect}
    >
      <div className="flex items-center gap-2">
        <Wand2
          className={cn('h-4 w-4 shrink-0', isSelected ? 'text-accent' : 'text-muted-foreground')}
        />
        <span className="text-sm font-medium truncate">{name}</span>
        {preset.is_builtin && (
          <span className="text-[10px] bg-muted text-muted-foreground px-1.5 py-0.5 rounded-full shrink-0">
            {t('effects.badge.builtin')}
          </span>
        )}
      </div>
      <p className="text-xs text-muted-foreground mt-1 line-clamp-1 pl-6">
        {description || t('effects.noDescription')}
      </p>
      <div className="flex items-center gap-2 mt-1.5 pl-6">
        <span className="text-[10px] text-muted-foreground">
          {t('effects.effectCount', { count: effectCount })}
        </span>
        <span className="text-[10px] text-muted-foreground/50">
          {preset.effects_chain
            .filter((e) => e.enabled)
            .map((e) => e.type)
            .join(' → ')}
        </span>
      </div>
    </button>
  );
}
