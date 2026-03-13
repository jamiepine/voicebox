import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ChevronDown,
  ChevronRight,
  ChevronUp,
  CircleCheck,
  CircleX,
  Download,
  ExternalLink,
  HardDrive,
  Heart,
  Loader2,
  RotateCcw,
  Scale,
  Trash2,
  X,
} from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { ActiveDownloadTask, HuggingFaceModelInfo, ModelStatus } from '@/lib/api/types';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';

async function fetchHuggingFaceModelInfo(repoId: string): Promise<HuggingFaceModelInfo> {
  const response = await fetch(`https://huggingface.co/api/models/${repoId}`);
  if (!response.ok) throw new Error(`Failed to fetch model info: ${response.status}`);
  return response.json();
}

function formatDownloads(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return n.toString();
}

function formatLicense(license: string): string {
  const map: Record<string, string> = {
    'apache-2.0': 'Apache 2.0',
    mit: 'MIT',
    'cc-by-4.0': 'CC BY 4.0',
    'cc-by-sa-4.0': 'CC BY-SA 4.0',
    'cc-by-nc-4.0': 'CC BY-NC 4.0',
    'openrail++': 'OpenRAIL++',
    openrail: 'OpenRAIL',
  };
  return map[license] || license;
}

function formatPipelineTag(tag: string): string {
  return tag
    .split('-')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / k ** i).toFixed(1)} ${sizes[i]}`;
}

export function ModelManagement() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);
  const [downloadingDisplayName, setDownloadingDisplayName] = useState<string | null>(null);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [dismissedErrors, setDismissedErrors] = useState<Set<string>>(new Set());
  const [localErrors, setLocalErrors] = useState<Map<string, string>>(new Map());

  // Modal state
  const [selectedModel, setSelectedModel] = useState<ModelStatus | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  const { data: modelStatus, isLoading } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: async () => {
      const result = await apiClient.getModelStatus();
      return result;
    },
    refetchInterval: 5000,
  });

  const { data: activeTasks } = useQuery({
    queryKey: ['activeTasks'],
    queryFn: () => apiClient.getActiveTasks(),
    refetchInterval: (query) => {
      const data = query.state.data;
      const hasActive = data?.downloads.some((d) => d.status === 'downloading');
      return hasActive ? 1000 : 5000;
    },
  });

  // HuggingFace model card query - only fetches when modal is open and model has a repo ID
  const { data: hfModelInfo, isLoading: hfLoading } = useQuery({
    queryKey: ['hfModelInfo', selectedModel?.hf_repo_id],
    queryFn: () => fetchHuggingFaceModelInfo(selectedModel!.hf_repo_id!),
    enabled: detailOpen && !!selectedModel?.hf_repo_id,
    staleTime: 1000 * 60 * 30, // Cache for 30 minutes
    retry: 1,
  });

  // Build a map of errored downloads for quick lookup, excluding dismissed ones
  const erroredDownloads = new Map<string, ActiveDownloadTask>();
  if (activeTasks?.downloads) {
    for (const dl of activeTasks.downloads) {
      if (dl.status === 'error' && !dismissedErrors.has(dl.model_name)) {
        const localErr = localErrors.get(dl.model_name);
        erroredDownloads.set(dl.model_name, localErr ? { ...dl, error: localErr } : dl);
      }
    }
  }
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

  // Build progress map from active tasks for inline display
  const downloadProgressMap = useMemo(() => {
    const map = new Map<string, ActiveDownloadTask>();
    if (activeTasks?.downloads) {
      for (const dl of activeTasks.downloads) {
        if (dl.status === 'downloading') {
          map.set(dl.model_name, dl);
        }
      }
    }
    return map;
  }, [activeTasks]);

  const handleDownloadComplete = useCallback(() => {
    setDownloadingModel(null);
    setDownloadingDisplayName(null);
    queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
    queryClient.invalidateQueries({ queryKey: ['activeTasks'] });
  }, [queryClient]);

  const handleDownloadError = useCallback(
    (error: string) => {
      if (downloadingModel) {
        setLocalErrors((prev) => new Map(prev).set(downloadingModel, error));
        setConsoleOpen(true);
      }
      setDownloadingModel(null);
      setDownloadingDisplayName(null);
      queryClient.invalidateQueries({ queryKey: ['activeTasks'] });
    },
    [queryClient, downloadingModel],
  );

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
    setDismissedErrors((prev) => {
      const next = new Set(prev);
      next.delete(modelName);
      return next;
    });

    const model = modelStatus?.models.find((m) => m.model_name === modelName);
    const displayName = model?.display_name || modelName;

    try {
      await apiClient.triggerModelDownload(modelName);

      setDownloadingModel(modelName);
      setDownloadingDisplayName(displayName);

      queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
      queryClient.invalidateQueries({ queryKey: ['activeTasks'] });
    } catch (error) {
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
    const prevDismissed = dismissedErrors;
    const prevLocalErrors = localErrors;
    const prevDownloadingModel = downloadingModel;
    const prevDownloadingDisplayName = downloadingDisplayName;

    setDismissedErrors((prev) => new Set(prev).add(modelName));
    setLocalErrors((prev) => {
      const next = new Map(prev);
      next.delete(modelName);
      return next;
    });
    if (downloadingModel === modelName) {
      setDownloadingModel(null);
      setDownloadingDisplayName(null);
    }

    cancelMutation.mutate(modelName, {
      onError: () => {
        setDismissedErrors(prevDismissed);
        setLocalErrors(prevLocalErrors);
        setDownloadingModel(prevDownloadingModel);
        setDownloadingDisplayName(prevDownloadingDisplayName);
        toast({
          title: 'Cancel failed',
          description: 'Could not cancel the download task.',
          variant: 'destructive',
        });
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
      const result = await apiClient.deleteModel(modelName);
      return result;
    },
    onSuccess: async () => {
      toast({
        title: 'Model deleted',
        description: `${modelToDelete?.displayName || 'Model'} has been deleted successfully.`,
      });
      setDeleteDialogOpen(false);
      setModelToDelete(null);
      setDetailOpen(false);
      setSelectedModel(null);
      await queryClient.invalidateQueries({ queryKey: ['modelStatus'], refetchType: 'all' });
      await queryClient.refetchQueries({ queryKey: ['modelStatus'] });
    },
    onError: (error: Error) => {
      toast({
        title: 'Delete failed',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const formatSize = (sizeMb?: number): string => {
    if (!sizeMb) return 'Unknown size';
    if (sizeMb < 1024) return `${sizeMb.toFixed(1)} MB`;
    return `${(sizeMb / 1024).toFixed(2)} GB`;
  };

  const getModelState = (model: ModelStatus) => {
    const isDownloading =
      (model.downloading || downloadingModel === model.model_name) &&
      !erroredDownloads.has(model.model_name) &&
      !dismissedErrors.has(model.model_name);
    const hasError = erroredDownloads.has(model.model_name);
    return { isDownloading, hasError };
  };

  const openModelDetail = (model: ModelStatus) => {
    setSelectedModel(model);
    setDetailOpen(true);
  };

  const ttsModels = modelStatus?.models.filter((m) => m.model_name.startsWith('qwen-tts')) ?? [];
  const otherTtsModels =
    modelStatus?.models.filter(
      (m) => m.model_name.startsWith('luxtts') || m.model_name.startsWith('chatterbox'),
    ) ?? [];
  const whisperModels = modelStatus?.models.filter((m) => m.model_name.startsWith('whisper')) ?? [];

  // Build sections
  const sections: { label: string; models: ModelStatus[] }[] = [
    { label: 'Voice Generation', models: ttsModels },
    ...(otherTtsModels.length > 0 ? [{ label: 'Other Voice Models', models: otherTtsModels }] : []),
    { label: 'Transcription', models: whisperModels },
  ];

  // Get detail modal state for selected model
  const selectedState = selectedModel ? getModelState(selectedModel) : null;
  const selectedError = selectedModel ? erroredDownloads.get(selectedModel.model_name) : undefined;

  // Keep selectedModel data fresh from query results
  const freshSelectedModel =
    selectedModel && modelStatus
      ? modelStatus.models.find((m) => m.model_name === selectedModel.model_name) || selectedModel
      : selectedModel;

  // Derive license from HF data
  const license =
    hfModelInfo?.cardData?.license ||
    hfModelInfo?.tags?.find((t) => t.startsWith('license:'))?.replace('license:', '');

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="shrink-0 pb-4">
        <h1 className="text-lg font-semibold">Models</h1>
        <p className="text-sm text-muted-foreground">
          Download and manage AI models for voice generation and transcription
        </p>
      </div>

      {/* Model list */}
      {isLoading ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      ) : modelStatus ? (
        <div className="flex-1 min-h-0 overflow-y-auto space-y-6">
          {sections.map((section) => (
            <div key={section.label}>
              <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1 px-1">
                {section.label}
              </h2>
              <div className="border rounded-lg divide-y overflow-hidden">
                {section.models.map((model) => {
                  const { isDownloading, hasError } = getModelState(model);
                  return (
                    <button
                      key={model.model_name}
                      type="button"
                      onClick={() => openModelDetail(model)}
                      className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-muted/50 transition-colors group"
                    >
                      {/* Status indicator */}
                      <div className="shrink-0">
                        {hasError ? (
                          <CircleX className="h-4 w-4 text-destructive" />
                        ) : isDownloading ? (
                          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                        ) : model.loaded ? (
                          <CircleCheck className="h-4 w-4 text-accent" />
                        ) : model.downloaded ? (
                          <CircleCheck className="h-4 w-4 text-emerald-500" />
                        ) : (
                          <Download className="h-4 w-4 text-muted-foreground/50" />
                        )}
                      </div>

                      {/* Name + inline progress */}
                      <div className="flex-1 min-w-0">
                        <span className="text-sm font-medium">{model.display_name}</span>
                        {isDownloading &&
                          (() => {
                            const dl = downloadProgressMap.get(model.model_name);
                            const pct = dl?.progress ?? 0;
                            const hasProgress = dl && dl.total && dl.total > 0;
                            return (
                              <div className="mt-1 space-y-0.5">
                                <Progress value={hasProgress ? pct : undefined} className="h-1" />
                                <div className="text-[10px] text-muted-foreground truncate">
                                  {hasProgress
                                    ? `${formatBytes(dl.current ?? 0)} / ${formatBytes(dl.total!)} (${pct.toFixed(0)}%)`
                                    : dl?.filename || 'Connecting...'}
                                </div>
                              </div>
                            );
                          })()}
                      </div>

                      {/* Right side info */}
                      <div className="shrink-0 flex items-center gap-2">
                        {hasError && (
                          <Badge variant="destructive" className="text-[10px] h-5">
                            Error
                          </Badge>
                        )}
                        {model.loaded && (
                          <Badge className="text-[10px] h-5 bg-accent/15 text-accent border-accent/30 hover:bg-accent/15">
                            Loaded
                          </Badge>
                        )}
                        {model.downloaded && !isDownloading && !hasError && (
                          <span className="text-xs text-muted-foreground">
                            {formatSize(model.size_mb)}
                          </span>
                        )}
                        {!model.downloaded && !isDownloading && !hasError && (
                          <span className="text-xs text-muted-foreground/60">Not downloaded</span>
                        )}
                        <ChevronRight className="h-4 w-4 text-muted-foreground/40 group-hover:text-muted-foreground transition-colors" />
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}

          {/* Error console */}
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
                          <span className="text-[#ce9178] whitespace-pre-wrap break-all">
                            {dl.error}
                          </span>
                        </>
                      ) : (
                        <>
                          {': '}
                          <span className="text-[#808080]">
                            No error details available. Try downloading again.
                          </span>
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

      {/* Model Detail Modal */}
      <Dialog open={detailOpen} onOpenChange={setDetailOpen}>
        <DialogContent className="sm:max-w-md">
          {freshSelectedModel && (
            <>
              <DialogHeader>
                <DialogTitle>{freshSelectedModel.display_name}</DialogTitle>
                <DialogDescription className="flex items-center gap-1.5">
                  {freshSelectedModel.hf_repo_id ? (
                    <a
                      href={`https://huggingface.co/${freshSelectedModel.hf_repo_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 hover:underline"
                    >
                      {freshSelectedModel.hf_repo_id}
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  ) : (
                    freshSelectedModel.model_name
                  )}
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-4 pt-2">
                {/* Status badges */}
                <div className="flex items-center gap-2 flex-wrap">
                  {freshSelectedModel.loaded && (
                    <Badge className="text-xs bg-accent/15 text-accent border-accent/30 hover:bg-accent/15">
                      <CircleCheck className="h-3 w-3 mr-1" />
                      Loaded
                    </Badge>
                  )}
                  {freshSelectedModel.downloaded && !freshSelectedModel.loaded && (
                    <Badge variant="secondary" className="text-xs">
                      <CircleCheck className="h-3 w-3 mr-1" />
                      Downloaded
                    </Badge>
                  )}
                  {selectedState?.hasError && (
                    <Badge variant="destructive" className="text-xs">
                      <CircleX className="h-3 w-3 mr-1" />
                      Error
                    </Badge>
                  )}
                  {!freshSelectedModel.downloaded &&
                    !selectedState?.isDownloading &&
                    !selectedState?.hasError && (
                      <Badge variant="outline" className="text-xs text-muted-foreground">
                        Not downloaded
                      </Badge>
                    )}
                </div>

                {/* HuggingFace model card info */}
                {hfLoading && freshSelectedModel.hf_repo_id && (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground py-2">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Loading model info...
                  </div>
                )}

                {hfModelInfo && (
                  <div className="space-y-3">
                    {/* Stats row */}
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1" title="Downloads">
                        <Download className="h-3.5 w-3.5" />
                        {formatDownloads(hfModelInfo.downloads)}
                      </span>
                      <span className="flex items-center gap-1" title="Likes">
                        <Heart className="h-3.5 w-3.5" />
                        {formatDownloads(hfModelInfo.likes)}
                      </span>
                      {license && (
                        <span className="flex items-center gap-1" title="License">
                          <Scale className="h-3.5 w-3.5" />
                          {formatLicense(license)}
                        </span>
                      )}
                    </div>

                    {/* Pipeline tag + author */}
                    <div className="flex flex-wrap gap-1.5">
                      {hfModelInfo.pipeline_tag && (
                        <Badge variant="outline" className="text-[10px]">
                          {formatPipelineTag(hfModelInfo.pipeline_tag)}
                        </Badge>
                      )}
                      {hfModelInfo.library_name && (
                        <Badge variant="outline" className="text-[10px]">
                          {hfModelInfo.library_name}
                        </Badge>
                      )}
                      {hfModelInfo.author && (
                        <Badge variant="outline" className="text-[10px]">
                          by {hfModelInfo.author}
                        </Badge>
                      )}
                    </div>

                    {/* Languages */}
                    {hfModelInfo.cardData?.language && hfModelInfo.cardData.language.length > 0 && (
                      <div>
                        <span className="text-xs text-muted-foreground">
                          {hfModelInfo.cardData.language.length > 10
                            ? `${hfModelInfo.cardData.language.length} languages supported`
                            : `Languages: ${hfModelInfo.cardData.language.join(', ')}`}
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* Disk size */}
                {freshSelectedModel.downloaded && freshSelectedModel.size_mb && (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <HardDrive className="h-4 w-4" />
                    <span>{formatSize(freshSelectedModel.size_mb)} on disk</span>
                  </div>
                )}

                {/* Error detail */}
                {selectedError?.error && (
                  <div className="rounded-md bg-destructive/10 border border-destructive/20 p-3 text-xs text-destructive">
                    {selectedError.error}
                  </div>
                )}

                {/* Actions */}
                <div className="flex items-center gap-2 pt-2 border-t">
                  {selectedState?.hasError ? (
                    <>
                      <Button
                        size="sm"
                        onClick={() => handleDownload(freshSelectedModel.model_name)}
                        variant="outline"
                        className="flex-1"
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Retry Download
                      </Button>
                      <Button
                        size="sm"
                        onClick={() => handleCancel(freshSelectedModel.model_name)}
                        variant="ghost"
                        disabled={
                          cancelMutation.isPending &&
                          cancelMutation.variables === freshSelectedModel.model_name
                        }
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </>
                  ) : selectedState?.isDownloading ? (
                    <>
                      <div className="flex-1 space-y-2">
                        {(() => {
                          const dl = freshSelectedModel
                            ? downloadProgressMap.get(freshSelectedModel.model_name)
                            : undefined;
                          const pct = dl?.progress ?? 0;
                          const hasProgress = dl && dl.total && dl.total > 0;
                          return (
                            <>
                              <Progress value={hasProgress ? pct : undefined} className="h-2" />
                              <div className="text-xs text-muted-foreground">
                                {hasProgress
                                  ? `${formatBytes(dl.current ?? 0)} / ${formatBytes(dl.total!)} (${pct.toFixed(1)}%)`
                                  : dl?.filename || 'Connecting to HuggingFace...'}
                              </div>
                            </>
                          );
                        })()}
                      </div>
                      <Button
                        size="sm"
                        onClick={() => handleCancel(freshSelectedModel.model_name)}
                        variant="ghost"
                        disabled={
                          cancelMutation.isPending &&
                          cancelMutation.variables === freshSelectedModel.model_name
                        }
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </>
                  ) : freshSelectedModel.downloaded ? (
                    <Button
                      size="sm"
                      onClick={() => {
                        setModelToDelete({
                          name: freshSelectedModel.model_name,
                          displayName: freshSelectedModel.display_name,
                          sizeMb: freshSelectedModel.size_mb,
                        });
                        setDeleteDialogOpen(true);
                      }}
                      variant="outline"
                      disabled={freshSelectedModel.loaded}
                      title={
                        freshSelectedModel.loaded ? 'Unload model before deleting' : 'Delete model'
                      }
                      className="flex-1"
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      {freshSelectedModel.loaded ? 'Unload to Delete' : 'Delete Model'}
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      onClick={() => handleDownload(freshSelectedModel.model_name)}
                      className="flex-1"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download
                    </Button>
                  )}
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>

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
  isDownloading: boolean;  // Local state - true if user just clicked download
  formatSize: (sizeMb?: number) => string;
}

function ModelItem({ model, onDownload, onDelete, isDownloading, formatSize }: ModelItemProps) {
  // Use server's downloading state OR local state (for immediate feedback before server updates)
  const showDownloading = model.downloading || isDownloading;

  const statusText = model.loaded
    ? 'Loaded'
    : showDownloading
      ? 'Downloading'
      : model.downloaded
        ? 'Downloaded'
        : 'Not downloaded';
  const sizeText =
    model.downloaded && model.size_mb && !showDownloading ? `, ${formatSize(model.size_mb)}` : '';
  const rowLabel = `${model.display_name}, ${statusText}${sizeText}. Use Tab to reach Download or Delete.`;

  return (
    <div
      className="flex items-center justify-between p-3 border rounded-lg"
      role="group"
      tabIndex={0}
      aria-label={rowLabel}
    >
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
              aria-label={
                model.loaded
                  ? 'Unload model before deleting'
                  : `Delete ${model.display_name}`
              }
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        ) : showDownloading ? (
          <Button size="sm" variant="outline" disabled aria-label={`${model.display_name} downloading`}>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            Downloading...
          </Button>
        ) : (
          <Button
            size="sm"
            onClick={onDownload}
            variant="outline"
            aria-label={`Download ${model.display_name}`}
          >
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
        )}
      </div>
    </div>
  );
}
