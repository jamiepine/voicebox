import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Download, Loader2, Plus, Trash2, X } from 'lucide-react';
import { useCallback, useState } from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
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
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';

/**
 * Model Management panel — displayed in the Settings page.
 *
 * Renders three sections:
 *  1. Built-in Voice Generation models (Qwen TTS 1.7B / 0.6B)
 *  2. Transcription models (Whisper variants)
 *  3. Custom Models — user-added HuggingFace TTS models
 *
 * Custom models use a "custom:<slug>" naming convention throughout the
 * frontend and backend so they can be distinguished from built-in models.
 *
 * @modified AJ - Kamyab (Ankit Jain) — Added Custom Models section, add/remove mutations, and CustomModelItem component
 */
export function ModelManagement() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [downloadingDisplayName, setDownloadingDisplayName] = useState<string | null>(null);

  const { data: modelStatus, isLoading } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: async () => {
      console.log('[Query] Fetching model status');
      const result = await apiClient.getModelStatus();
      console.log('[Query] Model status fetched:', result);
      return result;
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Callbacks for download completion
  const handleDownloadComplete = useCallback(() => {
    console.log('[ModelManagement] Download complete, clearing state');
    setDownloadingModel(null);
    setDownloadingDisplayName(null);
    queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
  }, [queryClient]);

  const handleDownloadError = useCallback(() => {
    console.log('[ModelManagement] Download error, clearing state');
    setDownloadingModel(null);
    setDownloadingDisplayName(null);
  }, []);

  // Use progress toast hook for the downloading model
  useModelDownloadToast({
    modelName: downloadingModel || '',
    displayName: downloadingDisplayName || '',
    enabled: !!downloadingModel && !!downloadingDisplayName,
    onComplete: handleDownloadComplete,
    onError: handleDownloadError,
  });

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [modelToDelete, setModelToDelete] = useState<{
    name: string;
    displayName: string;
    sizeMb?: number;
  } | null>(null);

  // Add Custom Model dialog state
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [newModelRepoId, setNewModelRepoId] = useState('');
  const [newModelDisplayName, setNewModelDisplayName] = useState('');

  const handleDownload = async (modelName: string) => {
    console.log('[Download] Button clicked for:', modelName, 'at', new Date().toISOString());

    // Find display name
    const model = modelStatus?.models.find((m) => m.model_name === modelName);
    const displayName = model?.display_name || modelName;

    try {
      // IMPORTANT: Call the API FIRST before setting state
      // Setting state enables the SSE EventSource in useModelDownloadToast,
      // which can block/delay the download fetch due to HTTP/1.1 connection limits
      console.log('[Download] Calling download API for:', modelName);
      const result = await apiClient.triggerModelDownload(modelName);
      console.log('[Download] Download API responded:', result);

      // NOW set state to enable SSE tracking (after download has started on backend)
      setDownloadingModel(modelName);
      setDownloadingDisplayName(displayName);

      // Download initiated successfully - state will be cleared when SSE reports completion
      // or by the polling interval detecting the model is downloaded
      queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
    } catch (error) {
      console.error('[Download] Download failed:', error);
      setDownloadingModel(null);
      setDownloadingDisplayName(null);
      toast({
        title: 'Download failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    }
  };

  const deleteMutation = useMutation({
    mutationFn: async (modelName: string) => {
      console.log('[Delete] Deleting model:', modelName);
      const result = await apiClient.deleteModel(modelName);
      console.log('[Delete] Model deleted successfully:', modelName);
      return result;
    },
    onSuccess: async (_data, _modelName) => {
      console.log('[Delete] onSuccess - showing toast and invalidating queries');
      toast({
        title: 'Model deleted',
        description: `${modelToDelete?.displayName || 'Model'} has been deleted successfully.`,
      });
      setDeleteDialogOpen(false);
      setModelToDelete(null);
      console.log('[Delete] Invalidating modelStatus query');
      await queryClient.invalidateQueries({
        queryKey: ['modelStatus'],
        refetchType: 'all',
      });
      console.log('[Delete] Explicitly refetching modelStatus query');
      await queryClient.refetchQueries({ queryKey: ['modelStatus'] });
      console.log('[Delete] Query refetched');
    },
    onError: (error: Error) => {
      console.log('[Delete] onError:', error);
      toast({
        title: 'Delete failed',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  // ── Add custom model mutation ───────────────────────────────────────
  // Registers a new HuggingFace model in data/custom_models.json.
  // This does NOT trigger a download — the user must click "Download".
  const addCustomModelMutation = useMutation({
    mutationFn: async (data: { hf_repo_id: string; display_name: string }) => {
      return apiClient.addCustomModel(data);
    },
    onSuccess: async () => {
      toast({
        title: 'Custom model added',
        description: `${newModelDisplayName} has been added successfully.`,
      });
      setAddDialogOpen(false);
      setNewModelRepoId('');
      setNewModelDisplayName('');
      await queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to add model',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  // ── Remove custom model mutation ────────────────────────────────────
  // Removes the model entry from data/custom_models.json.
  // Does NOT delete cached model files from the HuggingFace cache.
  // To delete cache, the user should click the trash icon (onDeleteCache).
  const removeCustomModelMutation = useMutation({
    mutationFn: async (modelId: string) => {
      return apiClient.removeCustomModel(modelId);
    },
    onSuccess: async () => {
      toast({
        title: 'Custom model removed',
        description: 'The custom model has been removed from your list.',
      });
      await queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to remove model',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const formatSize = (sizeMb?: number): string => {
    if (!sizeMb) return 'Unknown';
    if (sizeMb < 1024) return `${sizeMb.toFixed(1)} MB`;
    return `${(sizeMb / 1024).toFixed(2)} GB`;
  };

  const handleAddCustomModel = () => {
    if (!newModelRepoId.trim() || !newModelDisplayName.trim()) return;
    addCustomModelMutation.mutate({
      hf_repo_id: newModelRepoId.trim(),
      display_name: newModelDisplayName.trim(),
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Management</CardTitle>
        <CardDescription>
          Download and manage AI models for voice generation and transcription
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : modelStatus ? (
          <div className="space-y-4">
            {/* TTS Models */}
            <div>
              <h3 className="text-sm font-semibold mb-3 text-muted-foreground">
                Voice Generation Models
              </h3>
              <div className="space-y-2">
                {modelStatus.models
                  .filter((m) => m.model_name.startsWith('qwen-tts'))
                  .map((model) => (
                    <ModelItem
                      key={model.model_name}
                      model={model}
                      onDownload={() => handleDownload(model.model_name)}
                      onDelete={() => {
                        setModelToDelete({
                          name: model.model_name,
                          displayName: model.display_name,
                          sizeMb: model.size_mb,
                        });
                        setDeleteDialogOpen(true);
                      }}
                      isDownloading={downloadingModel === model.model_name}
                      formatSize={formatSize}
                    />
                  ))}
              </div>
            </div>

            {/* Whisper Models */}
            <div>
              <h3 className="text-sm font-semibold mb-3 text-muted-foreground">
                Transcription Models
              </h3>
              <div className="space-y-2">
                {modelStatus.models
                  .filter((m) => m.model_name.startsWith('whisper'))
                  .map((model) => (
                    <ModelItem
                      key={model.model_name}
                      model={model}
                      onDownload={() => handleDownload(model.model_name)}
                      onDelete={() => {
                        setModelToDelete({
                          name: model.model_name,
                          displayName: model.display_name,
                          sizeMb: model.size_mb,
                        });
                        setDeleteDialogOpen(true);
                      }}
                      isDownloading={downloadingModel === model.model_name}
                      formatSize={formatSize}
                    />
                  ))}
              </div>
            </div>

            {/* Custom Models */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-muted-foreground">
                  Custom Models
                </h3>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setAddDialogOpen(true)}
                >
                  <Plus className="h-4 w-4 mr-1" />
                  Add Model
                </Button>
              </div>
              <div className="space-y-2">
                {modelStatus.models
                  .filter((m) => m.is_custom)
                  .map((model) => (
                    <CustomModelItem
                      key={model.model_name}
                      model={model}
                      onDownload={() => handleDownload(model.model_name)}
                      onDeleteCache={() => {
                        setModelToDelete({
                          name: model.model_name,
                          displayName: model.display_name,
                          sizeMb: model.size_mb,
                        });
                        setDeleteDialogOpen(true);
                      }}
                      onRemove={() => {
                        // Extract custom ID from "custom:slug" format
                        const customId = model.model_name.replace('custom:', '');
                        removeCustomModelMutation.mutate(customId);
                      }}
                      isDownloading={downloadingModel === model.model_name}
                      formatSize={formatSize}
                    />
                  ))}
                {modelStatus.models.filter((m) => m.is_custom).length === 0 && (
                  <div className="text-sm text-muted-foreground p-3 border border-dashed rounded-lg text-center">
                    No custom models added yet. Click "Add Model" to add a HuggingFace model.
                  </div>
                )}
              </div>
            </div>

          </div>
        ) : null}
      </CardContent>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Model</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete <strong>{modelToDelete?.displayName}</strong>?
              {modelToDelete?.sizeMb && (
                <>
                  {' '}
                  This will free up {formatSize(modelToDelete.sizeMb)} of disk space. The model will
                  need to be re-downloaded if you want to use it again.
                </>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (modelToDelete) {
                  deleteMutation.mutate(modelToDelete.name);
                }
              }}
              disabled={deleteMutation.isPending}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleteMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                'Delete'
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Add Custom Model Dialog */}
      <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Custom Model</DialogTitle>
            <DialogDescription>
              Add a HuggingFace model to use for voice generation. The model must be compatible
              with the TTS backend.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label htmlFor="hf-repo-id">HuggingFace Repo ID</Label>
              <Input
                id="hf-repo-id"
                placeholder="e.g. AryanNsc/IND-QWENTTS-V1"
                value={newModelRepoId}
                onChange={(e) => setNewModelRepoId(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                The full repo ID from HuggingFace (owner/model-name)
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="display-name">Display Name</Label>
              <Input
                id="display-name"
                placeholder="e.g. IND QwenTTS V1"
                value={newModelDisplayName}
                onChange={(e) => setNewModelDisplayName(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setAddDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleAddCustomModel}
              disabled={
                addCustomModelMutation.isPending ||
                !newModelRepoId.trim() ||
                !newModelDisplayName.trim()
              }
            >
              {addCustomModelMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Adding...
                </>
              ) : (
                'Add Model'
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

interface ModelItemProps {
  model: {
    model_name: string;
    display_name: string;
    downloaded: boolean;
    downloading?: boolean;  // From server - true if download in progress
    size_mb?: number;
    loaded: boolean;
    is_custom?: boolean;
  };
  onDownload: () => void;
  onDelete: () => void;
  isDownloading: boolean;  // Local state - true if user just clicked download
  formatSize: (sizeMb?: number) => string;
}

/**
 * A single row in the built-in model list (Qwen TTS / Whisper).
 * Shows download status, size, and delete/download actions.
 */
function ModelItem({ model, onDownload, onDelete, isDownloading, formatSize }: ModelItemProps) {
  // Use server's downloading state OR local state (for immediate feedback before server updates)
  const showDownloading = model.downloading || isDownloading;

  return (
    <div className="flex items-center justify-between p-3 border rounded-lg">
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm">{model.display_name}</span>
          {model.loaded && (
            <Badge variant="default" className="text-xs">
              Loaded
            </Badge>
          )}
          {/* Only show Downloaded if actually downloaded AND not downloading */}
          {model.downloaded && !model.loaded && !showDownloading && (
            <Badge variant="secondary" className="text-xs">
              Downloaded
            </Badge>
          )}
        </div>
        {model.downloaded && model.size_mb && !showDownloading && (
          <div className="text-xs text-muted-foreground mt-1">
            Size: {formatSize(model.size_mb)}
          </div>
        )}
      </div>
      <div className="flex items-center gap-2">
        {model.downloaded && !showDownloading ? (
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 text-sm text-muted-foreground">
              <span>Ready</span>
            </div>
            <Button
              size="sm"
              onClick={onDelete}
              variant="outline"
              disabled={model.loaded}
              title={model.loaded ? 'Unload model before deleting' : 'Delete model'}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        ) : showDownloading ? (
          <Button size="sm" variant="outline" disabled>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            Downloading...
          </Button>
        ) : (
          <Button size="sm" onClick={onDownload} variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        )}
      </div>
    </div>
  );
}

interface CustomModelItemProps {
  model: {
    model_name: string;
    display_name: string;
    downloaded: boolean;
    downloading?: boolean;
    size_mb?: number;
    loaded: boolean;
  };
  onDownload: () => void;
  onDeleteCache: () => void;
  onRemove: () => void;
  isDownloading: boolean;
  formatSize: (sizeMb?: number) => string;
}

/**
 * A single row in the custom model list.
 * In addition to download/delete-cache, custom models have a "remove" button
 * (X icon) that un-registers the model from the config without deleting cached files.
 */
function CustomModelItem({ model, onDownload, onDeleteCache, onRemove, isDownloading, formatSize }: CustomModelItemProps) {
  const showDownloading = model.downloading || isDownloading;

  return (
    <div className="flex items-center justify-between p-3 border rounded-lg">
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm">{model.display_name}</span>
          <Badge variant="outline" className="text-xs">Custom</Badge>
          {model.loaded && (
            <Badge variant="default" className="text-xs">
              Loaded
            </Badge>
          )}
          {model.downloaded && !model.loaded && !showDownloading && (
            <Badge variant="secondary" className="text-xs">
              Downloaded
            </Badge>
          )}
        </div>
        {model.downloaded && model.size_mb && !showDownloading && (
          <div className="text-xs text-muted-foreground mt-1">
            Size: {formatSize(model.size_mb)}
          </div>
        )}
      </div>
      <div className="flex items-center gap-2">
        {model.downloaded && !showDownloading ? (
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 text-sm text-muted-foreground">
              <span>Ready</span>
            </div>
            <Button
              size="sm"
              onClick={onDeleteCache}
              variant="outline"
              disabled={model.loaded}
              title={model.loaded ? 'Unload model before deleting' : 'Delete cached model files'}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        ) : showDownloading ? (
          <Button size="sm" variant="outline" disabled>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            Downloading...
          </Button>
        ) : (
          <Button size="sm" onClick={onDownload} variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        )}
        <Button
          size="sm"
          onClick={onRemove}
          variant="ghost"
          title="Remove custom model from list"
          disabled={model.loaded}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
