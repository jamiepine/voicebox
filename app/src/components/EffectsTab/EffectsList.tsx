import { useQuery } from '@tanstack/react-query';
import { Loader2, Plus, Sparkles, Wand2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { apiClient } from '@/lib/api/client';
import type { EffectPresetResponse } from '@/lib/api/types';
import { cn } from '@/lib/utils/cn';
import { useEffectsStore } from '@/stores/effectsStore';

export function EffectsList() {
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
    <div className="flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Effects</h2>
        <Button variant="outline" size="sm" className="h-8 gap-1.5" onClick={handleCreateNew}>
          <Plus className="h-3.5 w-3.5" />
          New Preset
        </Button>
      </div>

      {/* Scrollable list */}
      <div className="flex-1 min-h-0 overflow-y-auto space-y-4">
        {/* Built-in presets */}
        {builtIn.length > 0 && (
          <div>
            <div className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider mb-2 px-1">
              Built-in
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

        {/* User presets */}
        {userPresets.length > 0 && (
          <div>
            <div className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider mb-2 px-1">
              Custom
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

        {/* New preset placeholder */}
        {isCreatingNew && (
          <div>
            <div className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider mb-2 px-1">
              New
            </div>
            <div className="rounded-xl border-2 border-accent/40 bg-accent/5 p-3">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-accent" />
                <span className="text-sm font-medium">Unsaved Preset</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Configure effects in the panel on the right.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
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
  const effectCount = preset.effects_chain.length;

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
        <span className="text-sm font-medium truncate">{preset.name}</span>
        {preset.is_builtin && (
          <span className="text-[10px] bg-muted text-muted-foreground px-1.5 py-0.5 rounded-full shrink-0">
            built-in
          </span>
        )}
      </div>
      <p className="text-xs text-muted-foreground mt-1 line-clamp-1 pl-6">
        {preset.description || 'No description'}
      </p>
      <div className="flex items-center gap-2 mt-1.5 pl-6">
        <span className="text-[10px] text-muted-foreground">
          {effectCount} effect{effectCount !== 1 ? 's' : ''}
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
