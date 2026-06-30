import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ChevronDown,
  ChevronRight,
  ChevronUp,
  CircleCheck,
  CircleX,
  Download,
  ExternalLink,
  FolderOpen,
  HardDrive,
  Heart,
  Loader2,
  RotateCcw,
  Scale,
  Trash2,
  Unplug,
  X,
} from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';
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
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';

async function fetchHuggingFaceModelInfo(repoId: string): Promise<HuggingFaceModelInfo> {
  const response = await fetch(`https://huggingface.co/api/models/${repoId}`);
  if (!response.ok) throw new Error(`Failed to fetch model info: ${response.status}`);
  return response.json();
}

const MODEL_DESCRIPTIONS: Record<string, string> = {
  'qwen-tts-1.7B':
    'High-quality multilingual TTS by Alibaba. Supports 10 languages with natural prosody and voice cloning from short reference audio.',
  'qwen-tts-0.6B':
    'Lightweight version of Qwen TTS. Same language support with faster inference, ideal for lower-end hardware.',
  luxtts:
    'Lightweight ZipVoice-based TTS designed for high quality voice cloning and 48kHz speech generation at speeds exceeding 150x realtime.',
  'chatterbox-tts':
    'Production-grade open source TTS by Resemble AI. Supports 23 languages with voice cloning and emotion exaggeration control.',
  'chatterbox-turbo':
    'Streamlined 350M parameter TTS by Resemble AI. High-quality English speech with less compute and VRAM than larger models.',
  'tada-1b':
    'HumeAI TADA 1B — English speech-language model built on Llama 3.2 1B. Generates 700s+ of coherent audio with synchronized text-acoustic alignment.',
  'tada-3b-ml':
    'HumeAI TADA 3B Multilingual — built on Llama 3.2 3B. Supports 10 languages with high-fidelity voice cloning via text-acoustic dual alignment.',
  kokoro:
    'Kokoro 82M by hexgrad. Tiny 82M-parameter TTS that runs at CPU realtime. Supports 8 languages with pre-built voice styles. Apache 2.0 licensed.',
  'qwen-custom-voice-1.7B':
    'Qwen3-TTS CustomVoice 1.7B by Alibaba. 9 premium preset voices with instruct-based style control for tone, emotion, and prosody. Supports 10 languages.',
  'qwen-custom-voice-0.6B':
    'Qwen3-TTS CustomVoice 0.6B by Alibaba. Lightweight version with the same 9 preset voices and instruct control. Faster inference for lower-end hardware.',
  'whisper-base':
    'Smallest Whisper model (74M parameters). Fast transcription with moderate accuracy.',
  'whisper-small':
    'Whisper Small (244M parameters). Good balance of speed and accuracy for transcription.',
  'whisper-medium':
    'Whisper Medium (769M parameters). Higher accuracy transcription at moderate speed.',
  'whisper-large':
    'Whisper Large (1.5B parameters). Best accuracy for speech-to-text across multiple languages.',
  'whisper-turbo':
    'Whisper Large v3 Turbo. Pruned for significantly faster inference while maintaining near-large accuracy.',
  'qwen3-0.6b':
    'Qwen3 0.6B — smallest of the Qwen3 instruct family. Very fast on CPU, runs at ~400 MB quantized on Apple Silicon. Good for dictation refinement and short completions.',
  'qwen3-1.7b':
    'Qwen3 1.7B — balanced size and quality. Handles subtle self-corrections and technical vocabulary better than the 0.6B. Runs at ~1.1 GB quantized on Apple Silicon.',
  'qwen3-4b':
    'Qwen3 4B — highest quality local refinement and longer-form reasoning. ~2.5 GB quantized on Apple Silicon, ~8 GB at full precision on PyTorch.',
};

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
  const { t } = useTranslation();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const platform = usePlatform();
  const customModelsDir = useServerStore((state) => state.customModelsDir);
  const setCustomModelsDir = useServerStore((state) => state.setCustomModelsDir);
  const [migrating, setMigrating] = useState(false);
  const [migrationProgress, setMigrationProgress] = useState<{
    current: number;
    total: number;
    progress: number;
    filename?: string;
    status: string;
  } | null>(null);
  const [pendingMigrateDir, setPendingMigrateDir] = useState<string | null>(null);
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

  const { data: cacheDir } = useQuery({
    queryKey: ['modelsCacheDir'],
    queryFn: () => apiClient.getModelsCacheDir(),
    staleTime: 1000 * 60 * 5,
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
        title: t('models.toast.downloadFailed'),
        description: error instanceof Error ? error.message : t('common.unknownError'),
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
          title: t('models.toast.cancelFailed'),
          description: t('models.toast.cancelFailedDescription'),
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
        title: t('models.toast.deleted'),
        description: t('models.toast.deletedDescription', {
          name: modelToDelete?.displayName || t('models.defaultName'),
        }),
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
        title: t('models.toast.deleteFailed'),
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const unloadMutation = useMutation({
    mutationFn: async (modelName: string) => {
      return await apiClient.unloadModel(modelName);
    },
    onSuccess: async (_data, modelName) => {
      toast({
        title: t('models.toast.unloaded'),
        description: t('models.toast.unloadedDescription', { name: modelName }),
      });
      await queryClient.invalidateQueries({ queryKey: ['modelStatus'], refetchType: 'all' });
      await queryClient.refetchQueries({ queryKey: ['modelStatus'] });
    },
    onError: (error: Error) => {
      toast({
        title: t('models.toast.unloadFailed'),
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const formatSize = (sizeMb?: number): string => {
    if (!sizeMb) return t('models.unknownSize');
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

  const voiceModels =
    modelStatus?.models.filter(
      (m) =>
        m.model_name.startsWith('qwen-tts') ||
        m.model_name.startsWith('qwen-custom-voice') ||
        m.model_name.startsWith('luxtts') ||
        m.model_name.startsWith('chatterbox') ||
        m.model_name.startsWith('tada') ||
        m.model_name.startsWith('kokoro'),
    ) ?? [];
  const whisperModels = modelStatus?.models.filter((m) => m.model_name.startsWith('whisper')) ?? [];
  const llmModels = modelStatus?.models.filter((m) => m.model_name.startsWith('qwen3-')) ?? [];

  // Build sections
  const sections: { label: string; models: ModelStatus[] }[] = [
    { label: t('models.sections.voiceGeneration'), models: voiceModels },
    { label: t('models.sections.transcription'), models: whisperModels },
    { label: t('models.sections.languageModels'), models: llmModels },
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
    hfModelInfo?.tags?.find((tag) => tag.startsWith('license:'))?.replace('license:', '');

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="shrink-0 pb-4">
        <h1 className="text-lg font-semibold">{t('models.title')}</h1>
        <p className="text-sm text-muted-foreground">{t('models.subtitle')}</p>
      </div>

      {/* Model storage location */}
      {platform.metadata.isTauri && cacheDir && (
        <div className="shrink-0 pb-4 border-b mb-4">
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
              <span className="text-xs text-muted-foreground">{t('models.storage.location')}</span>
              <p
                className="text-xs font-mono text-muted-foreground/70 truncate"
                title={cacheDir.path}
              >
                {cacheDir.path}
              </p>
            </div>
            <div className="flex items-center gap-1 shrink-0">
              <Button
                variant="ghost"
                size="sm"
                className="text-xs text-muted-foreground h-7 px-2"
                onClick={async () => {
                  try {
                    await platform.filesystem.openPath(cacheDir.path);
                  } catch {
                    toast({ title: t('models.toast.openFolderFailed'), variant: 'destructive' });
                  }
                }}
              >
                <FolderOpen className="h-3 w-3" />
                {t('models.storage.open')}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-xs text-muted-foreground h-7 px-2"
                onClick={async () => {
                  try {
                    const newDir = await platform.filesystem.pickDirectory(
                      t('models.storage.pickerTitle'),
                    );
                    if (!newDir) return;
                    setPendingMigrateDir(newDir);
                  } catch {
                    toast({ title: t('models.toast.pickerFailed'), variant: 'destructive' });
                  }
                }}
                disabled={migrating}
              >
                {migrating ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <FolderOpen className="h-3 w-3" />
                )}
                {migrating ? t('models.storage.migrating') : t('models.storage.change')}
              </Button>
              {customModelsDir && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-xs text-muted-foreground h-7 px-2"
                  disabled={migrating}
                  onClick={async () => {
                    setCustomModelsDir(null);
                    toast({ title: t('models.toast.resetToDefault') });
                    await platform.lifecycle.restartServer('');
                    queryClient.invalidateQueries();
                  }}
                >
                  <RotateCcw className="h-3 w-3" />
                  {t('models.storage.reset')}
                </Button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Model list */}
      {isLoading ? (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      ) : modelStatus ? (
        <div className="flex-1 min-h-0 overflow-y-auto space-y-6 pb-6">
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
                                    : dl?.filename || t('models.progress.connecting')}
                                </div>
                              </div>
                            );
                          })()}
                      </div>

                      {/* Right side info */}
                      <div className="shrink-0 flex items-center gap-2">
                        {hasError && (
                          <Badge variant="destructive" className="text-[10px] h-5">
                            {t('common.error')}
                          </Badge>
                        )}
                        {model.loaded && (
                          <Badge className="text-[10px] h-5 bg-accent/15 text-accent border-accent/30 hover:bg-accent/15">
                            {t('models.status.loaded')}
                          </Badge>
                        )}
                        {model.downloaded && !isDownloading && !hasError && (
                          <span className="text-xs text-muted-foreground">
                            {formatSize(model.size_mb)}
                          </span>
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
                  <span>{t('models.problems.title')}</span>
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
                  {t('models.problems.clearAll')}
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
                          <span className="text-[#808080]">{t('models.problems.noDetails')}</span>
                        </>
                      )}
                      <div className="text-[#6a9955] mt-0.5">
                        {t('models.problems.startedAt', {
                          time: new Date(dl.started_at).toLocaleString(),
                        })}
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
                      {t('models.status.loaded')}
                    </Badge>
                  )}
                  {selectedState?.hasError && (
                    <Badge variant="destructive" className="text-xs">
                      <CircleX className="h-3 w-3 mr-1" />
                      {t('common.error')}
                    </Badge>
                  )}
                </div>

                {/* HuggingFace model card info */}
                {hfLoading && freshSelectedModel.hf_repo_id && (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground py-2">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    {t('models.detail.loadingInfo')}
                  </div>
                )}

                {/* Description */}
                {MODEL_DESCRIPTIONS[freshSelectedModel.model_name] && (
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {MODEL_DESCRIPTIONS[freshSelectedModel.model_name]}
                  </p>
                )}

                {hfModelInfo && (
                  <div className="space-y-3">
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
                          {t('models.detail.byAuthor', { author: hfModelInfo.author })}
                        </Badge>
                      )}
                    </div>

                    {/* Stats row */}
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span
                        className="flex items-center gap-1"
                        title={t('models.detail.downloads')}
                      >
                        <Download className="h-3.5 w-3.5" />
                        {formatDownloads(hfModelInfo.downloads)}
                      </span>
                      <span className="flex items-center gap-1" title={t('models.detail.likes')}>
                        <Heart className="h-3.5 w-3.5" />
                        {formatDownloads(hfModelInfo.likes)}
                      </span>
                      {license && (
                        <span
                          className="flex items-center gap-1"
                          title={t('models.detail.license')}
                        >
                          <Scale className="h-3.5 w-3.5" />
                          {formatLicense(license)}
                        </span>
                      )}
                    </div>

                    {/* Languages */}
                    {hfModelInfo.cardData?.language && hfModelInfo.cardData.language.length > 0 && (
                      <div>
                        <span className="text-xs text-muted-foreground">
                          {hfModelInfo.cardData.language.length > 10
                            ? t('models.detail.languagesCount', {
                                count: hfModelInfo.cardData.language.length,
                              })
                            : t('models.detail.languagesList', {
                                list: hfModelInfo.cardData.language.join(', '),
                              })}
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* Disk size */}
                {freshSelectedModel.downloaded && freshSelectedModel.size_mb && (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <HardDrive className="h-3.5 w-3.5" />
                    <span>
                      {t('models.detail.onDisk', { size: formatSize(freshSelectedModel.size_mb) })}
                    </span>
                  </div>
                )}

                {/* Error detail */}
                {selectedError?.error && (
                  <div className="rounded-md bg-destructive/10 border border-destructive/20 p-3 text-xs text-destructive">
                    {selectedError.error}
                  </div>
                )}

                {/* Actions */}
                <div className="flex items-center gap-2 pt-2">
                  {selectedState?.hasError ? (
                    <>
                      <Button
                        size="sm"
                        onClick={() => handleDownload(freshSelectedModel.model_name)}
                        variant="outline"
                        className="flex-1"
                      >
                        <Download className="h-4 w-4 mr-2" />
                        {t('models.actions.retry')}
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
                                  : dl?.filename || t('models.progress.connectingHf')}
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
                    <div className="flex gap-2 flex-1">
                      {freshSelectedModel.loaded && (
                        <Button
                          size="sm"
                          onClick={() => unloadMutation.mutate(freshSelectedModel.model_name)}
                          variant="outline"
                          disabled={unloadMutation.isPending}
                          className="flex-1"
                        >
                          {unloadMutation.isPending ? (
                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          ) : (
                            <Unplug className="h-4 w-4 mr-2" />
                          )}
                          {unloadMutation.isPending
                            ? t('models.actions.unloading')
                            : t('models.actions.unload')}
                        </Button>
                      )}
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
                          freshSelectedModel.loaded
                            ? t('models.actions.unloadFirst')
                            : t('models.actions.deleteModel')
                        }
                        className="flex-1"
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        {t('models.actions.deleteModel')}
                      </Button>
                    </div>
                  ) : (
                    <Button
                      size="sm"
                      onClick={() => handleDownload(freshSelectedModel.model_name)}
                      className="flex-1"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      {t('models.actions.download')}
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
            <AlertDialogTitle>{t('models.deleteDialog.title')}</AlertDialogTitle>
            <AlertDialogDescription>
              <Trans
                i18nKey="models.deleteDialog.body"
                values={{ name: modelToDelete?.displayName }}
                components={{ strong: <strong /> }}
              />
              {modelToDelete?.sizeMb && (
                <>
                  {' '}
                  {t('models.deleteDialog.sizeNote', { size: formatSize(modelToDelete.sizeMb) })}
                </>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
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
                  {t('models.deleteDialog.deleting')}
                </>
              ) : (
                t('common.delete')
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Migration confirmation dialog */}
      <AlertDialog
        open={!!pendingMigrateDir}
        onOpenChange={(open) => !open && setPendingMigrateDir(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('models.migrateDialog.title')}</AlertDialogTitle>
            <AlertDialogDescription>{t('models.migrateDialog.description')}</AlertDialogDescription>
          </AlertDialogHeader>
          <div
            className="text-xs font-mono text-muted-foreground bg-muted/50 rounded px-3 py-2 truncate"
            title={pendingMigrateDir ?? ''}
          >
            {pendingMigrateDir}
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
            <AlertDialogAction
              onClick={async () => {
                if (!pendingMigrateDir) return;
                const newDir = pendingMigrateDir;
                setPendingMigrateDir(null);
                setMigrating(true);
                setMigrationProgress({
                  current: 0,
                  total: 0,
                  progress: 0,
                  status: 'downloading',
                  filename: t('models.migrateDialog.preparing'),
                });
                try {
                  // Start the migration (background task)
                  const migrationResult = await apiClient.migrateModels(newDir);

                  // If no models to migrate, warn user and skip the change
                  if (migrationResult.moved === 0) {
                    setMigrating(false);
                    setMigrationProgress(null);
                    toast({
                      title: t('models.toast.noModelsToMigrate'),
                      description: t('models.toast.noModelsToMigrateDescription'),
                    });
                    setPendingMigrateDir(null);
                    return;
                  }

                  // Connect to SSE for progress
                  await new Promise<void>((resolve, reject) => {
                    const es = new EventSource(apiClient.getMigrationProgressUrl());
                    es.onmessage = (event) => {
                      try {
                        const data = JSON.parse(event.data);
                        setMigrationProgress(data);
                        if (data.status === 'complete') {
                          es.close();
                          resolve();
                        } else if (data.status === 'error') {
                          es.close();
                          reject(new Error(data.error || t('models.toast.migrationFailed')));
                        }
                      } catch {
                        /* ignore parse errors */
                      }
                    };
                    es.onerror = () => {
                      es.close();
                      reject(new Error(t('models.toast.migrationConnectionLost')));
                    };
                  });

                  setCustomModelsDir(newDir);
                  setMigrationProgress({
                    current: 1,
                    total: 1,
                    progress: 100,
                    status: 'complete',
                    filename: t('models.migrateDialog.restartingServer'),
                  });
                  await platform.lifecycle.restartServer(newDir);
                  queryClient.invalidateQueries();
                  toast({ title: t('models.toast.migrated') });
                } catch (e) {
                  toast({
                    title: t('models.toast.migrationFailed'),
                    description:
                      e instanceof Error ? e.message : t('models.toast.migrationFailedGeneric'),
                    variant: 'destructive',
                  });
                } finally {
                  setMigrating(false);
                  setMigrationProgress(null);
                }
              }}
            >
              {t('models.migrateDialog.action')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Migration progress overlay */}
      {migrating && migrationProgress && (
        <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-sm flex items-center justify-center">
          <div className="w-full max-w-md px-8 space-y-6 text-center">
            <div className="space-y-2">
              <Loader2 className="h-8 w-8 animate-spin mx-auto text-muted-foreground" />
              <h2 className="text-lg font-semibold">{t('models.migrate.title')}</h2>
              <p className="text-sm text-muted-foreground">
                {migrationProgress.status === 'complete'
                  ? t('models.migrateDialog.restartingServer')
                  : t('models.migrate.offline')}
              </p>
            </div>
            {migrationProgress.total > 0 && (
              <div className="space-y-2">
                <Progress value={migrationProgress.progress} className="h-2" />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span className="truncate max-w-[60%]">{migrationProgress.filename}</span>
                  <span>
                    {formatBytes(migrationProgress.current)} /{' '}
                    {formatBytes(migrationProgress.total)}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
