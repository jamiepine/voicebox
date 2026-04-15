import { useQueryClient } from '@tanstack/react-query';
import { AnimatePresence, motion } from 'framer-motion';
import {
  AlignCenter,
  AudioLines,
  AudioWaveform,
  Download,
  FileArchive,
  Loader2,
  MoreHorizontal,
  Play,
  RotateCcw,
  Star,
  Trash2,
  Wand2,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import { EffectsChainEditor } from '@/components/Effects/EffectsChainEditor';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { EffectConfig, GenerationVersionResponse, HistoryResponse } from '@/lib/api/types';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import {
  useClearFailedGenerations,
  useDeleteGeneration,
  useExportGeneration,
  useExportGenerationAudio,
  useHistory,
  useImportGeneration,
} from '@/lib/hooks/useHistory';
import { cn } from '@/lib/utils/cn';
import { formatDate, formatDuration, formatEngineName } from '@/lib/utils/format';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';

// ─── Audio Bars ─────────────────────────────────────────────────────────────

function AudioBars({ mode }: { mode: 'idle' | 'generating' | 'playing' }) {
  const barColor = mode !== 'idle' ? 'bg-accent' : 'bg-muted-foreground/40';
  return (
    <div className="flex items-center gap-[2px] h-5">
      {[0, 1, 2, 3, 4].map((i) => (
        <motion.div
          key={`${mode}-${i}`}
          className={`w-[3px] rounded-full ${barColor}`}
          animate={
            mode === 'generating'
              ? { height: ['6px', '16px', '6px'] }
              : mode === 'playing'
                ? { height: ['8px', '14px', '4px', '12px', '8px'] }
                : { height: '8px' }
          }
          transition={
            mode === 'generating'
              ? { duration: 0.6, repeat: Infinity, delay: i * 0.08, ease: 'easeInOut' }
              : mode === 'playing'
                ? { duration: 1.2, repeat: Infinity, delay: i * 0.15, ease: 'easeInOut' }
                : { duration: 0.4, ease: 'easeOut' }
          }
        />
      ))}
    </div>
  );
}

// NEW ALTERNATE HISTORY VIEW - FIXED HEIGHT ROWS WITH INFINITE SCROLL
export function HistoryTable() {
  const [page, setPage] = useState(0);
  const [allHistory, setAllHistory] = useState<HistoryResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [isScrolled, setIsScrolled] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const loadMoreRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [generationToDelete, setGenerationToDelete] = useState<{ id: string; name: string } | null>(
    null,
  );
  const [effectsDialogOpen, setEffectsDialogOpen] = useState(false);
  const [effectsTargetId, setEffectsTargetId] = useState<string | null>(null);
  const [effectsTargetVersions, setEffectsTargetVersions] = useState<GenerationVersionResponse[]>(
    [],
  );
  const [effectsSourceVersionId, setEffectsSourceVersionId] = useState<string | null>(null);
  const [effectsChain, setEffectsChain] = useState<EffectConfig[]>([]);
  const [applyingEffects, setApplyingEffects] = useState(false);
  const [expandedVersionsId, setExpandedVersionsId] = useState<string | null>(null);
  const limit = 20;
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const {
    data: historyData,
    isLoading,
    isFetching,
  } = useHistory({
    limit,
    offset: page * limit,
  });

  const deleteGeneration = useDeleteGeneration();
  const clearFailed = useClearFailedGenerations();
  const [clearFailedDialogOpen, setClearFailedDialogOpen] = useState(false);
  const exportGeneration = useExportGeneration();
  const exportGenerationAudio = useExportGenerationAudio();
  const importGeneration = useImportGeneration();
  const addPendingGeneration = useGenerationStore((state) => state.addPendingGeneration);
  const setAudioWithAutoPlay = usePlayerStore((state) => state.setAudioWithAutoPlay);
  const restartCurrentAudio = usePlayerStore((state) => state.restartCurrentAudio);
  const currentAudioId = usePlayerStore((state) => state.audioId);
  const isPlaying = usePlayerStore((state) => state.isPlaying);
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;

  // Update accumulated history when new data arrives
  useEffect(() => {
    if (historyData?.items) {
      setTotal(historyData.total);
      if (page === 0) {
        // Reset to first page
        setAllHistory(historyData.items);
      } else {
        // Append new items, avoiding duplicates
        setAllHistory((prev) => {
          const existingIds = new Set(prev.map((item) => item.id));
          const newItems = historyData.items.filter((item) => !existingIds.has(item.id));
          return [...prev, ...newItems];
        });
      }
    }
  }, [historyData, page]);

  // Reset to page 0 when deletions, imports, or generation completions occur
  const pendingCount = useGenerationStore((state) => state.pendingGenerationIds.size);
  const prevPendingCountRef = useRef(pendingCount);
  useEffect(() => {
    if (deleteGeneration.isSuccess || importGeneration.isSuccess || clearFailed.isSuccess) {
      setPage(0);
      setAllHistory([]);
    }
  }, [deleteGeneration.isSuccess, importGeneration.isSuccess, clearFailed.isSuccess]);

  useEffect(() => {
    // A generation finished (pending count decreased) — scroll back to show it
    if (
      prevPendingCountRef.current > 0 &&
      pendingCount < prevPendingCountRef.current &&
      page !== 0
    ) {
      setPage(0);
      setAllHistory([]);
    }
    prevPendingCountRef.current = pendingCount;
  }, [pendingCount, page]);

  // Intersection Observer for infinite scroll
  useEffect(() => {
    const loadMoreEl = loadMoreRef.current;
    if (!loadMoreEl) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const target = entries[0];
        if (target.isIntersecting && !isFetching && allHistory.length < total) {
          setPage((prev) => prev + 1);
        }
      },
      {
        root: scrollRef.current,
        rootMargin: '100px',
        threshold: 0.1,
      },
    );

    observer.observe(loadMoreEl);
    return () => observer.disconnect();
  }, [isFetching, allHistory.length, total]);

  // Track scroll position for gradient effect
  useEffect(() => {
    const scrollEl = scrollRef.current;
    if (!scrollEl) return;

    const handleScroll = () => {
      setIsScrolled(scrollEl.scrollTop > 0);
    };

    scrollEl.addEventListener('scroll', handleScroll);
    return () => scrollEl.removeEventListener('scroll', handleScroll);
  }, []);

  const handlePlay = (audioId: string, text: string, profileId: string) => {
    // If clicking the same audio, restart it from the beginning
    if (currentAudioId === audioId) {
      restartCurrentAudio();
    } else {
      // Otherwise, load the new audio and auto-play it
      const audioUrl = apiClient.getAudioUrl(audioId);
      setAudioWithAutoPlay(audioUrl, audioId, profileId, text.substring(0, 50));
    }
  };

  const handleDownloadAudio = (generationId: string, text: string) => {
    exportGenerationAudio.mutate(
      { generationId, text },
      {
        onError: (error) => {
          toast({
            title: 'Failed to download audio',
            description: error.message,
            variant: 'destructive',
          });
        },
      },
    );
  };

  const handleExportPackage = (generationId: string, text: string) => {
    exportGeneration.mutate(
      { generationId, text },
      {
        onError: (error) => {
          toast({
            title: 'Failed to export generation',
            description: error.message,
            variant: 'destructive',
          });
        },
      },
    );
  };

  const handleDeleteClick = (generationId: string, profileName: string) => {
    setGenerationToDelete({ id: generationId, name: profileName });
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (generationToDelete) {
      deleteGeneration.mutate(generationToDelete.id);
      setDeleteDialogOpen(false);
      setGenerationToDelete(null);
    }
  };

  const handleRetry = async (generationId: string) => {
    try {
      const result = await apiClient.retryGeneration(generationId);
      addPendingGeneration(result.id);
      queryClient.invalidateQueries({ queryKey: ['history'] });
    } catch (error) {
      toast({
        title: 'Retry failed',
        description: error instanceof Error ? error.message : 'Could not retry generation',
        variant: 'destructive',
      });
    }
  };

  const handleRegenerate = async (generationId: string) => {
    try {
      await apiClient.regenerateGeneration(generationId);
      addPendingGeneration(generationId);
      queryClient.invalidateQueries({ queryKey: ['history'] });
    } catch (error) {
      toast({
        title: 'Regenerate failed',
        description: error instanceof Error ? error.message : 'Could not regenerate',
        variant: 'destructive',
      });
    }
  };

  const handleToggleFavorite = async (generationId: string) => {
    try {
      await apiClient.toggleFavorite(generationId);
      queryClient.invalidateQueries({ queryKey: ['history'] });
    } catch (error) {
      toast({
        title: 'Failed to update favorite',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    }
  };

  const handleApplyEffects = (generationId: string) => {
    const gen = allHistory.find((g) => g.id === generationId);
    const versions = gen?.versions ?? [];
    setEffectsTargetId(generationId);
    setEffectsTargetVersions(versions);
    // Default to clean/original version (no effects chain)
    const cleanVersion = versions.find((v) => !v.effects_chain || v.effects_chain.length === 0);
    setEffectsSourceVersionId(cleanVersion?.id ?? null);
    setEffectsChain([]);
    setEffectsDialogOpen(true);
  };

  const handleApplyEffectsConfirm = async () => {
    if (!effectsTargetId || effectsChain.length === 0) return;
    setApplyingEffects(true);
    try {
      const newVersion = await apiClient.applyEffectsToGeneration(effectsTargetId, {
        effects_chain: effectsChain,
        source_version_id: effectsSourceVersionId ?? undefined,
        set_as_default: true,
      });
      queryClient.invalidateQueries({ queryKey: ['history'] });

      // If the player is currently on this generation, reload with the new version audio
      if (currentAudioId === effectsTargetId) {
        const gen = allHistory.find((g) => g.id === effectsTargetId);
        if (gen) {
          const versionUrl = apiClient.getVersionAudioUrl(newVersion.id);
          setAudioWithAutoPlay(
            versionUrl,
            effectsTargetId,
            gen.profile_id,
            gen.text.substring(0, 50),
          );
        }
      }

      setEffectsDialogOpen(false);
      toast({ title: 'Effects applied', description: 'A new version has been created.' });
    } catch (error) {
      toast({
        title: 'Failed to apply effects',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setApplyingEffects(false);
    }
  };

  const handleSwitchVersion = async (generationId: string, versionId: string) => {
    try {
      await apiClient.setDefaultVersion(generationId, versionId);
      queryClient.invalidateQueries({ queryKey: ['history'] });
    } catch (error) {
      toast({
        title: 'Failed to switch version',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    }
  };

  const handlePlayVersion = (
    generationId: string,
    versionId: string,
    text: string,
    profileId: string,
  ) => {
    const audioUrl = apiClient.getVersionAudioUrl(versionId);
    setAudioWithAutoPlay(audioUrl, generationId, profileId, text.substring(0, 50));
  };

  const handleImportConfirm = () => {
    if (selectedFile) {
      importGeneration.mutate(selectedFile, {
        onSuccess: (data) => {
          setImportDialogOpen(false);
          setSelectedFile(null);
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
          toast({
            title: 'Generation imported',
            description: data.message || 'Generation imported successfully',
          });
        },
        onError: (error) => {
          toast({
            title: 'Failed to import generation',
            description: error.message,
            variant: 'destructive',
          });
        },
      });
    }
  };

  if (isLoading && page === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const history = allHistory;
  const hasMore = allHistory.length < total;
  const failedCount = history.filter((g) => g.status === 'failed').length;

  const handleClearFailedConfirm = () => {
    clearFailed.mutate(undefined, {
      onSuccess: (data) => {
        setClearFailedDialogOpen(false);
        toast({
          title: 'Cleared failed generations',
          description: `${data.deleted} failed ${data.deleted === 1 ? 'generation' : 'generations'} removed.`,
        });
      },
      onError: (error) => {
        setClearFailedDialogOpen(false);
        toast({
          title: 'Failed to clear',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      },
    });
  };

  return (
    <div className="flex flex-col h-full min-h-0 relative">
      {history.length === 0 ? (
        <div className="text-center py-12 px-5 border-2 border-dashed mb-5 border-muted rounded-md text-muted-foreground flex-1 flex items-center justify-center">
          No voice generations, yet...
        </div>
      ) : (
        <>
          {failedCount > 0 && (
            <div className="flex items-center justify-between px-1 pb-2">
              <span className="text-xs text-muted-foreground">
                {failedCount} failed {failedCount === 1 ? 'generation' : 'generations'}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs text-muted-foreground hover:text-destructive"
                onClick={() => setClearFailedDialogOpen(true)}
                disabled={clearFailed.isPending}
              >
                <Trash2 className="h-3 w-3 mr-1.5" />
                {clearFailed.isPending ? 'Clearing...' : 'Clear failed'}
              </Button>
            </div>
          )}
          {isScrolled && (
            <div className="absolute top-0 left-0 right-0 h-16 bg-gradient-to-b from-background to-transparent z-10 pointer-events-none" />
          )}
          <div
            ref={scrollRef}
            className={cn(
              'flex-1 min-h-0 overflow-y-auto space-y-2 pb-4',
              isPlayerVisible && BOTTOM_SAFE_AREA_PADDING,
            )}
          >
            {history.map((gen) => {
              const isCurrentlyPlaying = currentAudioId === gen.id && isPlaying;
              const isInProgress = gen.status === 'loading_model' || gen.status === 'generating';
              const isGenerating = isInProgress;
              const isFailed = gen.status === 'failed';
              const isPlayable = !isGenerating && !isFailed;
              const hasVersions = gen.versions && gen.versions.length > 1;
              const isVersionsExpanded = expandedVersionsId === gen.id;
              return (
                <div
                  key={gen.id}
                  className={cn(
                    'border rounded-md bg-card transition-colors text-left w-full',
                    isCurrentlyPlaying && 'bg-muted/70',
                  )}
                >
                  {/* Main row */}
                  <div
                    role={isPlayable ? 'button' : undefined}
                    tabIndex={isPlayable ? 0 : undefined}
                    className={cn(
                      'flex items-stretch gap-4 h-26 p-3 outline-none',
                      isPlayable && 'hover:bg-muted/70 cursor-pointer rounded-md',
                      isVersionsExpanded && 'rounded-b-none',
                    )}
                    aria-label={
                      isGenerating
                        ? `Generating speech for ${gen.profile_name}...`
                        : isFailed
                          ? `Generation failed for ${gen.profile_name}`
                          : isCurrentlyPlaying
                            ? `Sample from ${gen.profile_name}, ${formatDuration(gen.duration ?? 0)}, ${formatDate(gen.created_at)}. Playing. Press Enter to restart.`
                            : `Sample from ${gen.profile_name}, ${formatDuration(gen.duration ?? 0)}, ${formatDate(gen.created_at)}. Press Enter to play.`
                    }
                    onMouseDown={(e) => {
                      if (!isPlayable) return;
                      const target = e.target as HTMLElement;
                      if (target.closest('textarea') || window.getSelection()?.toString()) {
                        return;
                      }
                      handlePlay(gen.id, gen.text, gen.profile_id);
                    }}
                    onKeyDown={(e) => {
                      if (!isPlayable) return;
                      const target = e.target as HTMLElement;
                      if (target.closest('textarea') || target.closest('button')) return;
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        handlePlay(gen.id, gen.text, gen.profile_id);
                      }
                    }}
                  >
                    {/* Status icon */}
                    <div className="flex items-center shrink-0 w-10 justify-center overflow-hidden">
                      <AudioBars
                        mode={isGenerating ? 'generating' : isCurrentlyPlaying ? 'playing' : 'idle'}
                      />
                    </div>

                    {/* Left side - Meta information */}
                    <div className="flex flex-col gap-1.5 w-48 shrink-0 justify-center">
                      <div className="font-medium text-sm truncate" title={gen.profile_name}>
                        {gen.profile_name}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">{gen.language}</span>
                        <span className="text-xs text-muted-foreground">
                          {formatEngineName(gen.engine, gen.model_size)}
                        </span>
                        {isFailed ? (
                          <span className="text-xs text-destructive">Failed</span>
                        ) : !isGenerating ? (
                          <span className="text-xs text-muted-foreground">
                            {formatDuration(gen.duration ?? 0)}
                          </span>
                        ) : null}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {isInProgress ? (
                          <span className="text-accent">
                            {gen.status === 'loading_model' ? 'Loading model...' : 'Generating...'}
                          </span>
                        ) : (
                          formatDate(gen.created_at)
                        )}
                      </div>
                    </div>

                    {/* Right side - Transcript textarea */}
                    <div className="flex-1 min-w-0 flex">
                      <Textarea
                        value={gen.text}
                        className="flex-1 resize-none text-sm text-muted-foreground select-text"
                        readOnly
                        aria-label={`Transcript for sample from ${gen.profile_name}, ${formatDuration(gen.duration ?? 0)}`}
                      />
                    </div>

                    {/* Far right - Actions */}
                    <div
                      className="shrink-0 flex flex-col justify-center items-center gap-0.5"
                      onMouseDown={(e) => e.stopPropagation()}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <Button
                        variant="ghost"
                        size="icon"
                        className={cn(
                          'h-6 w-6 text-muted-foreground/50 hover:bg-muted-foreground/20 hover:text-muted-foreground',
                          gen.is_favorited && 'text-accent hover:text-accent',
                        )}
                        aria-label={gen.is_favorited ? 'Unfavorite' : 'Favorite'}
                        onClick={() => handleToggleFavorite(gen.id)}
                      >
                        <Star
                          className="h-2 w-2"
                          fill={gen.is_favorited ? 'currentColor' : 'none'}
                        />
                      </Button>
                      {hasVersions && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className={cn(
                            'h-6 w-6 text-muted-foreground/50 hover:bg-muted-foreground/20 hover:text-muted-foreground',
                            isVersionsExpanded && 'text-accent hover:text-accent',
                          )}
                          aria-label="Toggle versions"
                          onClick={() => setExpandedVersionsId(isVersionsExpanded ? null : gen.id)}
                        >
                          <AudioLines className="h-2 w-2" />
                        </Button>
                      )}

                      {isFailed ? (
                        <>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 text-muted-foreground/50 hover:bg-muted-foreground/20 hover:text-muted-foreground"
                            aria-label="Retry generation"
                            onClick={() => handleRetry(gen.id)}
                          >
                            <RotateCcw className="h-2 w-2" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 text-muted-foreground/50 hover:bg-muted-foreground/20 hover:text-muted-foreground"
                            aria-label="Delete generation"
                            disabled={deleteGeneration.isPending}
                            onClick={() => handleDeleteClick(gen.id, gen.profile_name)}
                          >
                            <Trash2 className="h-2 w-2" />
                          </Button>
                        </>
                      ) : (
                        <>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6 text-muted-foreground/50 hover:bg-muted-foreground/20 hover:text-muted-foreground"
                                aria-label="Actions"
                                disabled={isGenerating}
                              >
                                <MoreHorizontal className="h-2 w-2" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem
                                onClick={() => handlePlay(gen.id, gen.text, gen.profile_id)}
                              >
                                <Play className="mr-2 h-4 w-4" />
                                Play
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onClick={() => handleDownloadAudio(gen.id, gen.text)}
                                disabled={exportGenerationAudio.isPending}
                              >
                                <Download className="mr-2 h-4 w-4" />
                                Export Audio
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onClick={() => handleExportPackage(gen.id, gen.text)}
                                disabled={exportGeneration.isPending}
                              >
                                <FileArchive className="mr-2 h-4 w-4" />
                                Export Package
                              </DropdownMenuItem>
                              <DropdownMenuItem onClick={() => handleApplyEffects(gen.id)}>
                                <Wand2 className="mr-2 h-4 w-4" />
                                Apply Effects
                              </DropdownMenuItem>
                              <DropdownMenuItem onClick={() => handleRegenerate(gen.id)}>
                                <RotateCcw className="mr-2 h-4 w-4" />
                                Regenerate
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onClick={() => handleDeleteClick(gen.id, gen.profile_name)}
                                disabled={deleteGeneration.isPending}
                                // className="text-destructive focus:text-destructive"
                              >
                                <Trash2 className="mr-2 h-4 w-4" />
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Expandable versions panel */}
                  <AnimatePresence>
                    {isVersionsExpanded && gen.versions && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2, ease: 'easeOut' }}
                        className="overflow-hidden"
                      >
                        <div className="border-t border-border/50">
                          <div className="divide-y divide-border/40">
                            {gen.versions.map((v) => {
                              // Show source provenance when effects were applied to a non-clean version
                              const sourceVersion = v.source_version_id
                                ? gen.versions?.find((sv) => sv.id === v.source_version_id)
                                : null;
                              const showSource =
                                sourceVersion &&
                                sourceVersion.effects_chain &&
                                sourceVersion.effects_chain.length > 0;

                              return (
                                <button
                                  key={v.id}
                                  type="button"
                                  className="flex items-center gap-2 w-full h-9 px-3 text-left hover:bg-muted/50 transition-colors"
                                  onClick={() => {
                                    handlePlayVersion(gen.id, v.id, gen.text, gen.profile_id);
                                    if (!v.is_default) {
                                      handleSwitchVersion(gen.id, v.id);
                                    }
                                  }}
                                >
                                  <AudioLines className="h-3 w-3 shrink-0 text-muted-foreground" />
                                  <span className="truncate text-xs font-medium">{v.label}</span>
                                  {v.effects_chain && v.effects_chain.length > 0 && (
                                    <span className="text-[10px] text-muted-foreground truncate">
                                      {v.effects_chain.map((e) => e.type).join(' → ')}
                                    </span>
                                  )}
                                  {showSource && (
                                    <span className="text-[10px] text-muted-foreground/60 truncate">
                                      from {sourceVersion.label}
                                    </span>
                                  )}
                                  <span className="flex-1" />
                                  {v.is_default && (
                                    <span className="text-[10px] bg-accent/15 text-accent px-1.5 py-0.5 rounded-full">
                                      active
                                    </span>
                                  )}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })}

            {/* Load more trigger element */}
            {hasMore && (
              <div ref={loadMoreRef} className="flex items-center justify-center py-4">
                {isFetching && <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />}
              </div>
            )}

            {/* End of list indicator */}
            {!hasMore && history.length > 0 && (
              <div className="text-center py-4 text-xs text-muted-foreground">
                You've reached the end
              </div>
            )}
          </div>
        </>
      )}

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Generation</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this generation from "{generationToDelete?.name}"?
              This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setDeleteDialogOpen(false);
                setGenerationToDelete(null);
              }}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteConfirm}
              disabled={deleteGeneration.isPending}
            >
              {deleteGeneration.isPending ? 'Deleting...' : 'Delete'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={clearFailedDialogOpen} onOpenChange={setClearFailedDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Clear failed generations</DialogTitle>
            <DialogDescription>
              This will permanently delete {failedCount} failed{' '}
              {failedCount === 1 ? 'generation' : 'generations'} from your history. This cannot be
              undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setClearFailedDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleClearFailedConfirm}
              disabled={clearFailed.isPending}
            >
              {clearFailed.isPending ? 'Clearing...' : 'Clear all'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={importDialogOpen} onOpenChange={setImportDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Import Generation</DialogTitle>
            <DialogDescription>
              Import the generation from "{selectedFile?.name}". This will add it to your history.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setImportDialogOpen(false);
                setSelectedFile(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={handleImportConfirm}
              disabled={importGeneration.isPending || !selectedFile}
            >
              {importGeneration.isPending ? 'Importing...' : 'Import'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={effectsDialogOpen} onOpenChange={setEffectsDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Apply Effects</DialogTitle>
            <DialogDescription>
              Configure post-processing effects to apply to this generation. A new version will be
              created.
            </DialogDescription>
          </DialogHeader>
          {effectsTargetVersions.length > 1 && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Source</label>
              <Select
                value={effectsSourceVersionId ?? ''}
                onValueChange={(val) => setEffectsSourceVersionId(val || null)}
              >
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue placeholder="Select source version" />
                </SelectTrigger>
                <SelectContent>
                  {effectsTargetVersions.map((v) => (
                    <SelectItem key={v.id} value={v.id} className="text-xs">
                      {v.label}
                      {v.effects_chain && v.effects_chain.length > 0 && (
                        <span className="text-muted-foreground ml-1.5">
                          ({v.effects_chain.map((e) => e.type).join(' + ')})
                        </span>
                      )}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
          <div className="py-2 max-h-80 overflow-y-auto">
            <EffectsChainEditor value={effectsChain} onChange={setEffectsChain} />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setEffectsDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleApplyEffectsConfirm}
              disabled={applyingEffects || effectsChain.length === 0}
            >
              {applyingEffects ? 'Applying...' : 'Apply'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
