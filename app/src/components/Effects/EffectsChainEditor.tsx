import {
  closestCenter,
  DndContext,
  type DragEndEvent,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { useQuery } from '@tanstack/react-query';
import { ChevronDown, ChevronRight, GripVertical, Plus, Power, Trash2 } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { apiClient } from '@/lib/api/client';
import type { AvailableEffect, EffectConfig, EffectPresetResponse } from '@/lib/api/types';
import { cn } from '@/lib/utils/cn';

// Each effect in the chain gets a stable ID for dnd-kit
interface EffectWithId extends EffectConfig {
  _id: string;
}

let nextId = 0;
function makeId() {
  return `fx-${++nextId}`;
}

interface EffectsChainEditorProps {
  value: EffectConfig[];
  onChange: (chain: EffectConfig[]) => void;
  compact?: boolean;
  showPresets?: boolean;
}

export function EffectsChainEditor({
  value,
  onChange,
  compact = false,
  showPresets = true,
}: EffectsChainEditorProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Maintain stable IDs for each effect across renders.
  // We use a ref to map value items to IDs, rebuilding when length changes.
  const idsRef = useRef<string[]>([]);
  const items: EffectWithId[] = useMemo(() => {
    // Grow ID array if effects were added
    while (idsRef.current.length < value.length) {
      idsRef.current.push(makeId());
    }
    // Shrink if effects were removed
    if (idsRef.current.length > value.length) {
      idsRef.current = idsRef.current.slice(0, value.length);
    }
    return value.map((e, i) => ({ ...e, _id: idsRef.current[i] }));
  }, [value]);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  const { data: availableEffects } = useQuery({
    queryKey: ['available-effects'],
    queryFn: () => apiClient.getAvailableEffects(),
    staleTime: Infinity,
  });

  const { data: presets } = useQuery({
    queryKey: ['effect-presets'],
    queryFn: () => apiClient.listEffectPresets(),
    staleTime: 30_000,
  });

  const effectsMap = useMemo(() => {
    const m = new Map<string, AvailableEffect>();
    if (availableEffects) {
      for (const e of availableEffects.effects) {
        m.set(e.type, e);
      }
    }
    return m;
  }, [availableEffects]);

  function addEffect(type: string) {
    const def = effectsMap.get(type);
    if (!def) return;
    const params: Record<string, number> = {};
    for (const [key, p] of Object.entries(def.params)) {
      params[key] = p.default;
    }
    const newEffect: EffectConfig = { type, enabled: true, params };
    const newId = makeId();
    idsRef.current = [...idsRef.current, newId];
    onChange([...value, newEffect]);
    setExpandedId(newId);
  }

  const removeEffect = useCallback(
    (index: number) => {
      const removedId = idsRef.current[index];
      idsRef.current = idsRef.current.filter((_, i) => i !== index);
      onChange(value.filter((_, i) => i !== index));
      if (expandedId === removedId) setExpandedId(null);
    },
    [value, onChange, expandedId],
  );

  const toggleEnabled = useCallback(
    (index: number) => {
      onChange(value.map((e, i) => (i === index ? { ...e, enabled: !e.enabled } : e)));
    },
    [value, onChange],
  );

  const updateParam = useCallback(
    (index: number, paramName: string, paramValue: number) => {
      onChange(
        value.map((e, i) =>
          i === index ? { ...e, params: { ...e.params, [paramName]: paramValue } } : e,
        ),
      );
    },
    [value, onChange],
  );

  function loadPreset(preset: EffectPresetResponse) {
    idsRef.current = preset.effects_chain.map(() => makeId());
    onChange(preset.effects_chain);
    setExpandedId(null);
  }

  function clearAll() {
    idsRef.current = [];
    onChange([]);
    setExpandedId(null);
  }

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over || active.id === over.id) return;

    const oldIndex = idsRef.current.indexOf(active.id as string);
    const newIndex = idsRef.current.indexOf(over.id as string);
    if (oldIndex === -1 || newIndex === -1) return;

    idsRef.current = arrayMove(idsRef.current, oldIndex, newIndex);
    onChange(arrayMove([...value], oldIndex, newIndex));
  }

  return (
    <div className={cn('space-y-2', compact && 'text-sm')}>
      {/* Preset selector row */}
      {showPresets && (
        <div className="flex items-center gap-2">
          <Select
            onValueChange={(id) => {
              const preset = presets?.find((p) => p.id === id);
              if (preset) loadPreset(preset);
            }}
          >
            <SelectTrigger className="h-8 flex-1 text-xs focus:ring-0 focus:ring-offset-0">
              <SelectValue placeholder="Load preset..." />
            </SelectTrigger>
            <SelectContent>
              {presets?.map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.name}
                  {p.description && (
                    <span className="ml-1 text-muted-foreground">- {p.description}</span>
                  )}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {value.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              className="h-8 px-2 text-xs text-muted-foreground"
              onClick={clearAll}
            >
              Clear
            </Button>
          )}
        </div>
      )}

      {/* Sortable effects chain */}
      <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
        <SortableContext items={items.map((i) => i._id)} strategy={verticalListSortingStrategy}>
          {items.map((effect, index) => (
            <SortableEffectItem
              key={effect._id}
              id={effect._id}
              effect={effect}
              index={index}
              effectDef={effectsMap.get(effect.type)}
              isExpanded={expandedId === effect._id}
              onToggleExpand={() => setExpandedId(expandedId === effect._id ? null : effect._id)}
              onRemove={() => removeEffect(index)}
              onToggleEnabled={() => toggleEnabled(index)}
              onUpdateParam={(paramName, paramValue) => updateParam(index, paramName, paramValue)}
            />
          ))}
        </SortableContext>
      </DndContext>

      {/* Add effect */}
      {availableEffects && (
        <Select onValueChange={addEffect}>
          <SelectTrigger className="h-8 border-dashed text-xs text-muted-foreground focus:ring-0 focus:ring-offset-0">
            <Plus className="mr-1 h-3.5 w-3.5" />
            <SelectValue placeholder="Add effect..." />
          </SelectTrigger>
          <SelectContent>
            {availableEffects.effects.map((e) => (
              <SelectItem key={e.type} value={e.type}>
                {e.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sortable effect item
// ---------------------------------------------------------------------------

interface SortableEffectItemProps {
  id: string;
  effect: EffectConfig;
  index: number;
  effectDef?: AvailableEffect;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onRemove: () => void;
  onToggleEnabled: () => void;
  onUpdateParam: (paramName: string, paramValue: number) => void;
}

function SortableEffectItem({
  id,
  effect,
  effectDef,
  isExpanded,
  onToggleExpand,
  onRemove,
  onToggleEnabled,
  onUpdateParam,
}: SortableEffectItemProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id,
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    zIndex: isDragging ? 10 : undefined,
  };

  const label = effectDef?.label ?? effect.type;

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={cn(
        'rounded-md border',
        effect.enabled ? 'border-border bg-card' : 'border-border/50 bg-muted/30',
        isDragging && 'opacity-80 shadow-lg',
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-1 px-2 py-1.5">
        <button
          type="button"
          className="p-0.5 text-muted-foreground hover:text-foreground"
          onClick={onToggleExpand}
        >
          {isExpanded ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" />
          )}
        </button>

        <button
          type="button"
          className="p-0.5 text-muted-foreground/50 hover:text-muted-foreground cursor-grab active:cursor-grabbing touch-none"
          {...attributes}
          {...listeners}
        >
          <GripVertical className="h-3.5 w-3.5" />
        </button>

        <span
          className={cn('flex-1 text-xs font-medium', !effect.enabled && 'text-muted-foreground')}
        >
          {label}
        </span>

        <button
          type="button"
          className={cn(
            'p-0.5 transition-colors',
            effect.enabled ? 'text-primary' : 'text-muted-foreground hover:text-foreground',
          )}
          onClick={onToggleEnabled}
          title={effect.enabled ? 'Disable' : 'Enable'}
        >
          <Power className="h-3.5 w-3.5" />
        </button>

        <button
          type="button"
          className="p-0.5 text-muted-foreground hover:text-destructive"
          onClick={onRemove}
          title="Remove"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Params */}
      {isExpanded && effectDef && (
        <div className="space-y-3 border-t px-3 py-2.5">
          {Object.entries(effectDef.params).map(([paramName, paramDef]) => {
            const currentValue = effect.params[paramName] ?? paramDef.default;
            return (
              <div key={paramName} className="space-y-1">
                <div className="flex items-center justify-between">
                  <Label className="text-[11px] text-muted-foreground">
                    {paramDef.description}
                  </Label>
                  <span className="text-[11px] font-mono tabular-nums text-foreground">
                    {currentValue.toFixed(
                      paramDef.step < 1 ? Math.max(1, -Math.floor(Math.log10(paramDef.step))) : 0,
                    )}
                  </span>
                </div>
                <Slider
                  min={paramDef.min}
                  max={paramDef.max}
                  step={paramDef.step}
                  value={[currentValue]}
                  onValueChange={([v]) => onUpdateParam(paramName, v)}
                />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
