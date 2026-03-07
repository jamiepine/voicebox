import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ChevronDown, ChevronUp, Download, Loader2, RotateCcw, Trash2, X } from 'lucide-react';
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
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { ActiveDownloadTask } from '@/lib/api/types';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';

export function ModelManagement() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [downloadingDisplayName, setDownloadingDisplayName] = useState<string | null>(null);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [dismissedErrors, setDismissedErrors] = useState<Set<string>>(new Set());
  const [localErrors, setLocalErrors] = useState<Map<string, string>>(new Map());

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

  const { data: activeTasks } = useQuery({
    queryKey: ['activeTasks'],
    queryFn: () => apiClient.getActiveTasks(),
    refetchInterval: 5000,
  });

  // Build a map of errored downloads for quick lookup, excluding dismissed ones
  // Merge server errors with locally captured SSE errors
  const erroredDownloads = new Map<string, ActiveDownloadTask>();
  if (activeTasks?.downloads) {
    for (const dl of activeTasks.downloads) {
      if (dl.status === 'error' && !dismissedErrors.has(dl.model_name)) {
        // Prefer locally captured error (from SSE) over server error
        const localErr = localErrors.get(dl.model_name);
        erroredDownloads.set(dl.model_name, localErr ? { ...dl, error: localErr } : dl);
      }
    }
  }
  // Also add locally captured errors that aren't in server response yet
  for (const [modelName, error] of localErrors) {
    if (!erroredDownloads.has(modelName) && !dismissedErrors.has(modelName)) {
      erroredDownloads.set(modelName, {
        model_name: modelName,
        status: 'error',
        started_at: new Date().toISOString(),
        error,
      });
    }
  }

  const errorCount = erroredDownloads.size;

  // Callbacks for download completion
  const handleDownloadComplete = useCallback(() => {
    console.log('[ModelManagement] Download complete, clearing state');
    setDownloadingModel(null);
    setDownloadingDisplayName(null);
    queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
    queryClient.invalidateQueries({ queryKey: ['activeTasks'] });
  }, [queryClient]);

  const handleDownloadError = useCallback((error: string) => {
    console.log('[ModelManagement] Download error, clearing state');
    if (downloadingModel) {
      setLocalErrors((prev) => new Map(prev).set(downloadingModel, error));
      setConsoleOpen(true);
    }
    setDownloadingModel(null);
    setDownloadingDisplayName(null);
    queryClient.invalidateQueries({ queryKey: ['activeTasks'] });
  }, [queryClient, downloadingModel]);

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

  const handleDownload = async (modelName: string) => {
    console.log('[Download] Button clicked for:', modelName, 'at', new Date().toISOString());
    // Clear any previous dismissal so fresh errors can appear
    setDismissedErrors((prev) => {
      const next = new Set(prev);
      next.delete(modelName);
      return next;
    });

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
      queryClient.invalidateQueries({ queryKey: ['activeTasks'] });
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

  const cancelMutation = useMutation({
    mutationFn: (modelName: string) => apiClient.cancelDownload(modelName),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['modelStatus'], refetchType: 'all' });
      await queryClient.invalidateQueries({ queryKey: ['activeTasks'], refetchType: 'all' });
    },
  });

  const handleCancel = (modelName: string) => {
    // Snapshot previous state for rollback
    const prevDismissed = dismissedErrors;
    const prevLocalErrors = localErrors;
    const prevDownloadingModel = downloadingModel;
    const prevDownloadingDisplayName = downloadingDisplayName;

    // Optimistically hide the error and suppress downloading state in UI
    setDismissedErrors((prev) => new Set(prev).add(modelName));
    setLocalErrors((prev) => { const next = new Map(prev); next.delete(modelName); return next; });
    if (downloadingModel === modelName) {
      setDownloadingModel(null);
      setDownloadingDisplayName(null);
    }

    cancelMutation.mutate(modelName, {
      onError: () => {
        // Rollback optimistic updates on failure
        setDismissedErrors(prevDismissed);
        setLocalErrors(prevLocalErrors);
        setDownloadingModel(prevDownloadingModel);
        setDownloadingDisplayName(prevDownloadingDisplayName);
        toast({ title: 'Cancel failed', description: 'Could not cancel the download task.', variant: 'destructive' });
      },
    });
  };

  const clearAllMutation = useMutation({
    mutationFn: () => apiClient.clearAllTasks(),
    onSuccess: async () => {
      setDismissedErrors(new Set());
      setLocalErrors(new Map());
      setDownloadingModel(null);
      setDownloadingDisplayName(null);
      await queryClient.invalidateQueries({ queryKey: ['modelStatus'], refetchType: 'all' });
      await queryClient.invalidateQueries({ queryKey: ['activeTasks'], refetchType: 'all' });
    },
  });

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

  const formatSize = (sizeMb?: number): string => {
    if (!sizeMb) return 'Unknown';
    if (sizeMb < 1024) return `${sizeMb.toFixed(1)} MB`;
    return `${(sizeMb / 1024).toFixed(2)} GB`;
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
                      onCancel={() => handleCancel(model.model_name)}
                      isDownloading={downloadingModel === model.model_name}
                      isCancelling={cancelMutation.isPending && cancelMutation.variables === model.model_name}
                      isDismissed={dismissedErrors.has(model.model_name)}
                      erroredDownload={erroredDownloads.get(model.model_name)}
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
                      onCancel={() => handleCancel(model.model_name)}
                      isDownloading={downloadingModel === model.model_name}
                      isCancelling={cancelMutation.isPending && cancelMutation.variables === model.model_name}
                      isDismissed={dismissedErrors.has(model.model_name)}
                      erroredDownload={erroredDownloads.get(model.model_name)}
                      formatSize={formatSize}
                    />
                  ))}
              </div>
            </div>

            {/* Console Panel */}
            {errorCount > 0 && (
              <div className="border rounded-lg overflow-hidden">
                <div className="flex items-center justify-between px-3 py-1.5 bg-muted/50 text-xs font-medium text-muted-foreground">
                  <button
                    type="button"
                    onClick={() => setConsoleOpen((v) => !v)}
                    className="flex items-center gap-2 hover:text-foreground transition-colors"
                  >
                    {consoleOpen ? (
                      <ChevronUp className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronDown className="h-3.5 w-3.5" />
                    )}
                    <span>Problems</span>
                    <Badge variant="destructive" className="text-[10px] h-4 px-1.5 rounded-full">
                      {errorCount}
                    </Badge>
                  </button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 px-2 text-xs text-muted-foreground hover:text-foreground"
                    onClick={() => clearAllMutation.mutate()}
                    disabled={clearAllMutation.isPending}
                  >
                    <RotateCcw className="h-3 w-3 mr-1" />
                    Clear All
                  </Button>
                </div>
                {consoleOpen && (
                  <div className="bg-[#1e1e1e] text-[#d4d4d4] p-3 max-h-48 overflow-auto font-mono text-xs leading-relaxed">
                    {Array.from(erroredDownloads.entries()).map(([modelName, dl]) => (
                      <div key={modelName} className="mb-2 last:mb-0">
                        <span className="text-[#f44747]">[error]</span>{' '}
                        <span className="text-[#569cd6]">{modelName}</span>
                        {dl.error ? (
                          <>
                            {': '}
                            <span className="text-[#ce9178] whitespace-pre-wrap break-all">{dl.error}</span>
                          </>
                        ) : (
                          <>
                            {': '}
                            <span className="text-[#808080]">No error details available. Try downloading again.</span>
                          </>
                        )}
                        <div className="text-[#6a9955] mt-0.5">
                          started at {new Date(dl.started_at).toLocaleString()}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
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
  };
  onDownload: () => void;
  onDelete: () => void;
  onCancel: () => void;
  isDownloading: boolean;  // Local state - true if user just clicked download
  isCancelling: boolean;
  isDismissed: boolean;
  erroredDownload?: ActiveDownloadTask;
  formatSize: (sizeMb?: number) => string;
}

function ModelItem({ model, onDownload, onDelete, onCancel, isDownloading, isCancelling, isDismissed, erroredDownload, formatSize }: ModelItemProps) {
  // Use server's downloading state OR local state (for immediate feedback before server updates)
  // Suppress downloading if user just dismissed/cancelled this model
  const showDownloading = (model.downloading || isDownloading) && !erroredDownload && !isDismissed;

  return (
    <div className="flex items-center justify-between p-3 border rounded-lg">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm">{model.display_name}</span>
          {model.loaded && (
            <Badge variant="default" className="text-xs">
              Loaded
            </Badge>
          )}
          {model.downloaded && !model.loaded && !showDownloading && !erroredDownload && (
            <Badge variant="secondary" className="text-xs">
              Downloaded
            </Badge>
          )}
          {erroredDownload && (
            <Badge variant="destructive" className="text-xs">
              Error
            </Badge>
          )}
        </div>
        {model.downloaded && model.size_mb && !showDownloading && !erroredDownload && (
          <div className="text-xs text-muted-foreground mt-1">
            Size: {formatSize(model.size_mb)}
          </div>
        )}
      </div>
      <div className="flex items-center gap-2 shrink-0 ml-2">
        {erroredDownload ? (
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={onDownload} variant="outline">
              <Download className="h-4 w-4 mr-2" />
              Retry
            </Button>
            <Button
              size="sm"
              onClick={onCancel}
              variant="ghost"
              disabled={isCancelling}
              title="Dismiss error"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        ) : model.downloaded && !showDownloading ? (
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
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" disabled>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Downloading...
            </Button>
            <Button
              size="sm"
              onClick={onCancel}
              variant="ghost"
              disabled={isCancelling}
              title="Cancel download"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
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
