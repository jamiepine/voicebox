import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2, Play, Save, Trash2, Wand2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import { EffectsChainEditor } from '@/components/Effects/EffectsChainEditor';
import { GenerationPicker } from '@/components/Effects/GenerationPicker';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { HistoryResponse } from '@/lib/api/types';
import { useHistory } from '@/lib/hooks/useHistory';
import { useEffectsStore } from '@/stores/effectsStore';
import { usePlayerStore } from '@/stores/playerStore';

export function EffectsDetail() {
  const selectedPresetId = useEffectsStore((s) => s.selectedPresetId);
  const isCreatingNew = useEffectsStore((s) => s.isCreatingNew);
  const workingChain = useEffectsStore((s) => s.workingChain);
  const setWorkingChain = useEffectsStore((s) => s.setWorkingChain);
  const setSelectedPresetId = useEffectsStore((s) => s.setSelectedPresetId);
  const setIsCreatingNew = useEffectsStore((s) => s.setIsCreatingNew);

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // "Save as Custom" dialog state
  const [saveAsDialogOpen, setSaveAsDialogOpen] = useState(false);
  const [saveAsName, setSaveAsName] = useState('');
  const [saveAsDescription, setSaveAsDescription] = useState('');

  // Preview state
  const [previewGenId, setPreviewGenId] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const blobUrlRef = useRef<string | null>(null);
  const setAudioWithAutoPlay = usePlayerStore((s) => s.setAudioWithAutoPlay);

  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Auto-select the most recent generation as preview source
  const { data: historyData } = useHistory({ limit: 1 });
  useEffect(() => {
    if (!previewGenId && historyData?.items?.length) {
      const first = historyData.items.find((g) => g.status === 'completed');
      if (first) setPreviewGenId(first.id);
    }
  }, [historyData, previewGenId]);

  const { data: preset } = useQuery({
    queryKey: ['effect-preset', selectedPresetId],
    queryFn: () =>
      selectedPresetId
        ? apiClient
            .listEffectPresets()
            .then((all) => all.find((p) => p.id === selectedPresetId) ?? null)
        : null,
    enabled: !!selectedPresetId,
    staleTime: 30_000,
  });

  // Sync name/description when selecting a preset
  useEffect(() => {
    if (preset) {
      setName(preset.name);
      setDescription(preset.description ?? '');
    } else if (isCreatingNew) {
      setName('');
      setDescription('');
    }
  }, [preset, isCreatingNew]);

  // Cleanup blob URL on unmount
  useEffect(() => {
    return () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, []);

  const isEditing = !!selectedPresetId || isCreatingNew;
  const isBuiltIn = preset?.is_builtin ?? false;

  async function handlePreview() {
    if (!previewGenId || workingChain.length === 0) return;

    setPreviewLoading(true);
    try {
      const blob = await apiClient.previewEffects(previewGenId, workingChain);

      // Revoke old blob URL
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
      }

      const url = URL.createObjectURL(blob);
      blobUrlRef.current = url;

      // Play through the main audio player
      setAudioWithAutoPlay(url, `preview-${Date.now()}`, null, 'Effects Preview');
    } catch (error) {
      toast({
        title: 'Preview failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setPreviewLoading(false);
    }
  }

  function handleSelectGeneration(gen: HistoryResponse) {
    setPreviewGenId(gen.id);
  }

  async function handleSaveNew() {
    if (!name.trim()) {
      toast({ title: 'Name required', variant: 'destructive' });
      return;
    }
    setSaving(true);
    try {
      const created = await apiClient.createEffectPreset({
        name: name.trim(),
        description: description.trim() || undefined,
        effects_chain: workingChain,
      });
      queryClient.invalidateQueries({ queryKey: ['effect-presets'] });
      setIsCreatingNew(false);
      setSelectedPresetId(created.id);
      toast({ title: 'Preset saved', description: `"${created.name}" has been created.` });
    } catch (error) {
      toast({
        title: 'Failed to save',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  }

  async function handleSaveExisting() {
    if (!selectedPresetId || !name.trim()) return;
    setSaving(true);
    try {
      await apiClient.updateEffectPreset(selectedPresetId, {
        name: name.trim(),
        description: description.trim() || undefined,
        effects_chain: workingChain,
      });
      queryClient.invalidateQueries({ queryKey: ['effect-presets'] });
      queryClient.invalidateQueries({ queryKey: ['effect-preset', selectedPresetId] });
      toast({ title: 'Preset updated' });
    } catch (error) {
      toast({
        title: 'Failed to save',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  }

  function handleSaveAsNew() {
    // Open the dialog with a suggested name based on the current preset
    setSaveAsName(`${name} (Copy)`);
    setSaveAsDescription(description);
    setSaveAsDialogOpen(true);
  }

  async function handleSaveAsConfirm() {
    if (!saveAsName.trim()) {
      toast({ title: 'Name required', variant: 'destructive' });
      return;
    }
    setSaving(true);
    try {
      const created = await apiClient.createEffectPreset({
        name: saveAsName.trim(),
        description: saveAsDescription.trim() || undefined,
        effects_chain: workingChain,
      });
      queryClient.invalidateQueries({ queryKey: ['effect-presets'] });
      setSaveAsDialogOpen(false);
      setSelectedPresetId(created.id);
      toast({ title: 'Preset saved', description: `"${created.name}" has been created.` });
    } catch (error) {
      toast({
        title: 'Failed to save',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!selectedPresetId) return;
    setDeleting(true);
    try {
      await apiClient.deleteEffectPreset(selectedPresetId);
      queryClient.invalidateQueries({ queryKey: ['effect-presets'] });
      setSelectedPresetId(null);
      setWorkingChain([]);
      toast({ title: 'Preset deleted' });
    } catch (error) {
      toast({
        title: 'Failed to delete',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setDeleting(false);
    }
  }

  if (!isEditing) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground">
        <div className="text-center space-y-2">
          <Wand2 className="h-10 w-10 mx-auto opacity-30" />
          <p className="text-sm">Select a preset or create a new one</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">
          {isCreatingNew ? 'New Preset' : isBuiltIn ? preset?.name : 'Edit Preset'}
        </h2>
        <div className="flex items-center gap-2">
          {!isBuiltIn && !isCreatingNew && (
            <>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 text-destructive hover:text-destructive gap-1.5"
                onClick={handleDelete}
                disabled={deleting}
              >
                <Trash2 className="h-3.5 w-3.5" />
                {deleting ? 'Deleting...' : 'Delete'}
              </Button>
              <Button
                size="sm"
                className="h-8 gap-1.5"
                onClick={handleSaveExisting}
                disabled={saving || workingChain.length === 0}
              >
                <Save className="h-3.5 w-3.5" />
                {saving ? 'Saving...' : 'Save'}
              </Button>
            </>
          )}
          {isCreatingNew && (
            <Button
              size="sm"
              className="h-8 gap-1.5"
              onClick={handleSaveNew}
              disabled={saving || workingChain.length === 0}
            >
              <Save className="h-3.5 w-3.5" />
              {saving ? 'Saving...' : 'Save Preset'}
            </Button>
          )}
          {isBuiltIn && (
            <Button
              size="sm"
              variant="outline"
              className="h-8 gap-1.5"
              onClick={handleSaveAsNew}
              disabled={saving}
            >
              <Save className="h-3.5 w-3.5" />
              {saving ? 'Saving...' : 'Save as Custom'}
            </Button>
          )}
        </div>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 min-h-0 overflow-y-auto space-y-5 pr-1">
        {/* Name & description */}
        {(isCreatingNew || !isBuiltIn) && (
          <div className="space-y-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Name</Label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My preset..."
                className="h-9"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Description</Label>
              <Textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe what this preset does..."
                className="min-h-[60px] resize-none"
              />
            </div>
          </div>
        )}

        {/* Built-in description (read-only) */}
        {isBuiltIn && preset?.description && (
          <p className="text-sm text-muted-foreground">{preset.description}</p>
        )}

        {/* Effects chain editor */}
        <EffectsChainEditor value={workingChain} onChange={setWorkingChain} showPresets={false} />

        <Separator />

        {/* Preview section */}
        <div className="space-y-3">
          <Label className="text-xs">Preview</Label>
          <div className="flex items-center gap-2">
            <GenerationPicker
              selectedId={previewGenId}
              onSelect={handleSelectGeneration}
              className="flex-1"
            />
            <Button
              variant="outline"
              size="sm"
              className="h-8 gap-1.5 shrink-0"
              onClick={handlePreview}
              disabled={!previewGenId || workingChain.length === 0 || previewLoading}
            >
              {previewLoading ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5" />
                  Preview
                </>
              )}
            </Button>
          </div>
          <p className="text-[11px] text-muted-foreground">
            Preview applies effects to the clean version without saving.
          </p>
        </div>
      </div>

      {/* Save as Custom dialog */}
      <Dialog open={saveAsDialogOpen} onOpenChange={setSaveAsDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Save as Custom Preset</DialogTitle>
            <DialogDescription>
              Create a new custom preset based on the current effects chain.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div className="space-y-1.5">
              <Label className="text-xs">Name</Label>
              <Input
                value={saveAsName}
                onChange={(e) => setSaveAsName(e.target.value)}
                placeholder="My preset..."
                className="h-9"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && saveAsName.trim()) {
                    handleSaveAsConfirm();
                  }
                }}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Description</Label>
              <Textarea
                value={saveAsDescription}
                onChange={(e) => setSaveAsDescription(e.target.value)}
                placeholder="Describe what this preset does..."
                className="min-h-[60px] resize-none"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setSaveAsDialogOpen(false)} disabled={saving}>
              Cancel
            </Button>
            <Button onClick={handleSaveAsConfirm} disabled={saving || !saveAsName.trim()}>
              <Save className="h-3.5 w-3.5 mr-1.5" />
              {saving ? 'Saving...' : 'Save'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
