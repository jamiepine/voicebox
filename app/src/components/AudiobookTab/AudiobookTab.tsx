import { useQueryClient } from '@tanstack/react-query';
import { useNavigate } from '@tanstack/react-router';
import {
  AlertTriangle,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Download,
  FlaskConical,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  Square,
  Upload,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
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
import { LANGUAGE_OPTIONS, type LanguageCode } from '@/lib/constants/languages';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { useExportStoryAudio } from '@/lib/hooks/useStories';
import { cn } from '@/lib/utils/cn';
import { chunkText, type TextChunk } from '@/lib/utils/textChunking';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useStoryStore } from '@/stores/storyStore';
import { useUIStore } from '@/stores/uiStore';

const HARD_MAX_CHUNK_SIZE = 4500;
const DEFAULT_TARGET_CHUNK_SIZE = 4500;
const MAX_CHUNK_RETRIES = 3;
const LARGE_FILE_WARNING_BYTES = 2 * 1024 * 1024;
const QUICK_PREVIEW_SENTENCE_LIMIT = 5;

type ChunkStatus = 'pending' | 'running' | 'done' | 'failed';
type AudiobookRunStatus =
  | 'running'
  | 'paused'
  | 'stopping'
  | 'stopped'
  | 'completed'
  | 'completed_with_errors';

interface AudiobookChunk extends TextChunk {
  status: ChunkStatus;
  attempts: number;
  generationId?: string;
  error?: string;
}

interface AudiobookRun {
  storyId: string;
  storyName: string;
  profileId: string;
  language: LanguageCode;
  modelSize: '1.7B' | '0.6B';
  instruct?: string;
  chunks: AudiobookChunk[];
  status: AudiobookRunStatus;
  startedAt: string;
  finishedAt?: string;
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return 'Unknown generation error';
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function buildRunChunks(chunks: TextChunk[]): AudiobookChunk[] {
  return chunks.map((chunk) => ({
    ...chunk,
    status: 'pending',
    attempts: 0,
  }));
}

function splitTextIntoSentences(rawText: string): string[] {
  const normalized = rawText.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim();
  if (!normalized) {
    return [];
  }

  const paragraphs = normalized
    .split(/\n+/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);

  const sentencePattern = /[^.!?]+[.!?]+(?:["')\]]+)?|[^.!?]+$/g;
  const sentences: string[] = [];

  for (const paragraph of paragraphs) {
    const matches = paragraph.match(sentencePattern);
    if (!matches || matches.length === 0) {
      sentences.push(paragraph);
      continue;
    }
    sentences.push(...matches.map((sentence) => sentence.trim()).filter(Boolean));
  }

  return sentences;
}

function buildQuickPreviewText(rawText: string): { text: string; sentenceCount: number } {
  const sentences = splitTextIntoSentences(rawText);
  const previewSentences = sentences.slice(0, QUICK_PREVIEW_SENTENCE_LIMIT);
  return {
    text: previewSentences.join(' ').trim(),
    sentenceCount: previewSentences.length,
  };
}

function createStoryName(fileName: string | null): string {
  const dateStamp = new Date().toISOString().slice(0, 10);
  if (!fileName) {
    return `Audiobook ${dateStamp}`;
  }

  const basename = fileName.replace(/\.txt$/i, '').trim();
  if (!basename) {
    return `Audiobook ${dateStamp}`;
  }

  return `${basename} (${dateStamp})`;
}

export function AudiobookTab() {
  const { toast } = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const processingRef = useRef(false);
  const pauseRequestedRef = useRef(false);
  const stopRequestedRef = useRef(false);
  const runRef = useRef<AudiobookRun | null>(null);

  const setIsGenerating = useGenerationStore((state) => state.setIsGenerating);
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const setAudioWithAutoPlay = usePlayerStore((state) => state.setAudioWithAutoPlay);
  const isPlayerVisible = !!audioUrl;

  const selectedProfileId = useUIStore((state) => state.selectedProfileId);
  const setSelectedProfileId = useUIStore((state) => state.setSelectedProfileId);
  const setSelectedStoryId = useStoryStore((state) => state.setSelectedStoryId);
  const { data: profiles } = useProfiles();
  const exportStoryAudio = useExportStoryAudio();

  const [fileName, setFileName] = useState<string | null>(null);
  const [originalText, setOriginalText] = useState('');
  const [text, setText] = useState('');
  const [targetChunkSize, setTargetChunkSize] = useState<number>(DEFAULT_TARGET_CHUNK_SIZE);
  const [language, setLanguage] = useState<LanguageCode>('en');
  const [modelSize, setModelSize] = useState<'1.7B' | '0.6B'>('1.7B');
  const [instruct, setInstruct] = useState('');
  const [run, setRun] = useState<AudiobookRun | null>(null);
  const [isPreviewGenerating, setIsPreviewGenerating] = useState(false);
  const [lastPreviewFingerprint, setLastPreviewFingerprint] = useState<string | null>(null);
  const [isSummaryOpen, setIsSummaryOpen] = useState(true);
  const [isSetupOpen, setIsSetupOpen] = useState(true);
  const [isChunksOpen, setIsChunksOpen] = useState(true);

  useEffect(() => {
    runRef.current = run;
  }, [run]);

  useEffect(() => {
    if (!selectedProfileId && profiles && profiles.length > 0) {
      setSelectedProfileId(profiles[0].id);
    }
  }, [selectedProfileId, profiles, setSelectedProfileId]);

  useEffect(() => {
    if (run?.status === 'running' || run?.status === 'stopping' || run?.status === 'paused') {
      setIsSetupOpen(false);
    }
  }, [run?.status]);

  const preparedChunks = useMemo(
    () => chunkText(text, targetChunkSize, HARD_MAX_CHUNK_SIZE),
    [text, targetChunkSize],
  );
  const oversizedPreparedChunks = useMemo(
    () => preparedChunks.filter((chunk) => chunk.charCount > HARD_MAX_CHUNK_SIZE),
    [preparedChunks],
  );

  const previewChunks = useMemo<AudiobookChunk[]>(() => {
    if (run) {
      return run.chunks;
    }

    return preparedChunks.map(
      (chunk): AudiobookChunk => ({
        ...chunk,
        status: 'pending',
        attempts: 0,
        generationId: undefined,
        error: undefined,
      }),
    );
  }, [preparedChunks, run]);

  const chunkStats = useMemo(() => {
    const chunks = run ? run.chunks : previewChunks;
    const total = chunks.length;
    const completed = chunks.filter((chunk) => chunk.status === 'done').length;
    const failed = chunks.filter((chunk) => chunk.status === 'failed').length;
    const running = chunks.filter((chunk) => chunk.status === 'running').length;
    const pending = chunks.filter((chunk) => chunk.status === 'pending').length;
    const progress = total > 0 ? Math.round((completed / total) * 100) : 0;

    return {
      total,
      completed,
      failed,
      running,
      pending,
      progress,
    };
  }, [previewChunks, run]);

  const charCount = text.length;
  const wordCount = text.split(/\s+/).filter(Boolean).length;
  const lineCount = text ? text.split('\n').length : 0;
  const isDirty = text !== originalText;
  const hasText = text.trim().length > 0;

  const hasPendingRunChunks = useMemo(
    () =>
      !!run && run.chunks.some((chunk) => chunk.status === 'pending' || chunk.status === 'running'),
    [run],
  );
  const quickPreview = useMemo(() => buildQuickPreviewText(text), [text]);
  const previewFingerprint = useMemo(
    () =>
      [selectedProfileId || '', language, modelSize, instruct.trim(), quickPreview.text].join('::'),
    [selectedProfileId, language, modelSize, instruct, quickPreview.text],
  );
  const isPreviewCurrent =
    !!lastPreviewFingerprint && lastPreviewFingerprint === previewFingerprint;

  const processRun = useCallback(async () => {
    if (processingRef.current) {
      return;
    }

    processingRef.current = true;
    setIsGenerating(true);

    try {
      while (true) {
        const current = runRef.current;
        if (!current) {
          break;
        }

        if (stopRequestedRef.current) {
          setRun((prev) => {
            if (!prev) {
              return prev;
            }
            return {
              ...prev,
              status: 'stopped',
              finishedAt: new Date().toISOString(),
            };
          });
          break;
        }

        if (pauseRequestedRef.current) {
          await wait(250);
          continue;
        }

        const nextChunkIndex = current.chunks.findIndex((chunk) => chunk.status === 'pending');
        if (nextChunkIndex === -1) {
          const hasFailures = current.chunks.some((chunk) => chunk.status === 'failed');
          setRun((prev) => {
            if (!prev) {
              return prev;
            }
            return {
              ...prev,
              status: hasFailures ? 'completed_with_errors' : 'completed',
              finishedAt: new Date().toISOString(),
            };
          });
          break;
        }

        for (
          let attempt = current.chunks[nextChunkIndex].attempts + 1;
          attempt <= MAX_CHUNK_RETRIES;
          attempt += 1
        ) {
          setRun((prev) => {
            if (!prev) {
              return prev;
            }

            const chunks = [...prev.chunks];
            const chunk = chunks[nextChunkIndex];
            if (!chunk) {
              return prev;
            }

            chunks[nextChunkIndex] = {
              ...chunk,
              status: 'running',
              attempts: attempt,
              error: undefined,
            };

            return {
              ...prev,
              status: 'running',
              chunks,
            };
          });

          try {
            const latest = runRef.current;
            if (!latest) {
              break;
            }

            const chunk = latest.chunks[nextChunkIndex];
            if (chunk.charCount > HARD_MAX_CHUNK_SIZE) {
              setRun((prev) => {
                if (!prev) {
                  return prev;
                }

                const chunks = [...prev.chunks];
                const target = chunks[nextChunkIndex];
                if (!target) {
                  return prev;
                }

                chunks[nextChunkIndex] = {
                  ...target,
                  status: 'failed',
                  error: `Chunk exceeds ${HARD_MAX_CHUNK_SIZE} characters. Edit text and retry.`,
                };

                return {
                  ...prev,
                  chunks,
                };
              });
              break;
            }

            const generation = await apiClient.generateSpeech({
              profile_id: latest.profileId,
              text: chunk.text,
              language: latest.language,
              model_size: latest.modelSize,
              instruct: latest.instruct || undefined,
            });

            await apiClient.addStoryItem(latest.storyId, {
              generation_id: generation.id,
            });

            setRun((prev) => {
              if (!prev) {
                return prev;
              }

              const chunks = [...prev.chunks];
              const target = chunks[nextChunkIndex];
              if (!target) {
                return prev;
              }

              chunks[nextChunkIndex] = {
                ...target,
                status: 'done',
                generationId: generation.id,
                error: undefined,
              };

              return {
                ...prev,
                chunks,
              };
            });
            break;
          } catch (error) {
            const errorMessage = getErrorMessage(error);
            const isLastAttempt = attempt >= MAX_CHUNK_RETRIES;

            setRun((prev) => {
              if (!prev) {
                return prev;
              }

              const chunks = [...prev.chunks];
              const target = chunks[nextChunkIndex];
              if (!target) {
                return prev;
              }

              chunks[nextChunkIndex] = {
                ...target,
                status: isLastAttempt ? 'failed' : 'pending',
                attempts: attempt,
                error: errorMessage,
              };

              return {
                ...prev,
                chunks,
              };
            });

            if (isLastAttempt) {
              break;
            }

            await wait(1000 * 2 ** (attempt - 1));
          }
        }
      }
    } finally {
      processingRef.current = false;
      setIsGenerating(false);

      const latest = runRef.current;
      if (latest?.storyId) {
        await queryClient.invalidateQueries({ queryKey: ['history'] });
        await queryClient.invalidateQueries({ queryKey: ['stories'] });
        await queryClient.invalidateQueries({ queryKey: ['stories', latest.storyId] });
      }
    }
  }, [queryClient, setIsGenerating]);

  const handlePickTextFile = () => {
    fileInputRef.current?.click();
  };

  const handleTextFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';

    if (!file) {
      return;
    }

    if (!file.name.toLowerCase().endsWith('.txt') && file.type !== 'text/plain') {
      toast({
        title: 'Invalid file type',
        description: 'Please pick a .txt file.',
        variant: 'destructive',
      });
      return;
    }

    if (file.size > LARGE_FILE_WARNING_BYTES) {
      toast({
        title: 'Large text file',
        description: 'Large files are supported, but initial parsing may take a moment.',
      });
    }

    try {
      const loadedText = (await file.text()).replace(/\r\n/g, '\n').replace(/\r/g, '\n');
      if (!loadedText.trim()) {
        toast({
          title: 'Empty text file',
          description: 'The selected file has no readable text.',
          variant: 'destructive',
        });
        return;
      }

      setFileName(file.name);
      setOriginalText(loadedText);
      setText(loadedText);
      setRun(null);
      pauseRequestedRef.current = false;
      stopRequestedRef.current = false;

      toast({
        title: 'Text loaded',
        description: `${file.name} is ready. Review the beginning and end before starting generation.`,
      });
    } catch (error) {
      toast({
        title: 'Failed to read file',
        description: getErrorMessage(error),
        variant: 'destructive',
      });
    }
  };

  const handleResetText = () => {
    setText(originalText);
  };

  const handleStartGeneration = async () => {
    if (!selectedProfileId) {
      toast({
        title: 'No voice selected',
        description: 'Select a voice profile before generating.',
        variant: 'destructive',
      });
      return;
    }

    if (!text.trim()) {
      toast({
        title: 'No text',
        description: 'Load or enter text before starting generation.',
        variant: 'destructive',
      });
      return;
    }

    const chunks = chunkText(text, targetChunkSize, HARD_MAX_CHUNK_SIZE);
    if (chunks.length === 0) {
      toast({
        title: 'No chunks created',
        description: 'Adjust text or chunk size and try again.',
        variant: 'destructive',
      });
      return;
    }

    try {
      const storyName = createStoryName(fileName);
      const story = await apiClient.createStory({
        name: storyName,
        description: `Generated from ${fileName || 'manual text'} in Audiobook tab`,
      });

      const nextRun: AudiobookRun = {
        storyId: story.id,
        storyName: story.name,
        profileId: selectedProfileId,
        language,
        modelSize,
        instruct: instruct.trim() || undefined,
        chunks: buildRunChunks(chunks),
        status: 'running',
        startedAt: new Date().toISOString(),
      };

      stopRequestedRef.current = false;
      pauseRequestedRef.current = false;
      runRef.current = nextRun;
      setRun(nextRun);
      setIsSetupOpen(false);
      setIsSummaryOpen(true);
      setIsChunksOpen(true);

      toast({
        title: 'Generation started',
        description: `Created story "${story.name}" and queued ${chunks.length} chunks.`,
      });

      void processRun();
    } catch (error) {
      toast({
        title: 'Failed to start generation',
        description: getErrorMessage(error),
        variant: 'destructive',
      });
    }
  };

  const handleQuickPreview = async () => {
    if (!selectedProfileId) {
      toast({
        title: 'No voice selected',
        description: 'Select a voice profile before previewing.',
        variant: 'destructive',
      });
      return;
    }

    if (!quickPreview.text) {
      toast({
        title: 'No preview text',
        description: 'Add text first. Preview uses the first 5 sentences.',
        variant: 'destructive',
      });
      return;
    }

    try {
      setIsPreviewGenerating(true);
      setIsGenerating(true);

      const generation = await apiClient.generateSpeech({
        profile_id: selectedProfileId,
        text: quickPreview.text,
        language,
        model_size: modelSize,
        instruct: instruct.trim() || undefined,
      });

      setAudioWithAutoPlay(
        apiClient.getAudioUrl(generation.id),
        generation.id,
        selectedProfileId,
        `Preview (${quickPreview.sentenceCount} sentences)`,
      );
      setLastPreviewFingerprint(previewFingerprint);
      await queryClient.invalidateQueries({ queryKey: ['history'] });

      toast({
        title: 'Preview ready',
        description: `Generated ${quickPreview.sentenceCount} sentence preview. This is not added to a Story.`,
      });
    } catch (error) {
      toast({
        title: 'Preview failed',
        description: getErrorMessage(error),
        variant: 'destructive',
      });
    } finally {
      setIsPreviewGenerating(false);
      setIsGenerating(false);
    }
  };

  const handlePause = () => {
    if (!run || run.status !== 'running') {
      return;
    }
    pauseRequestedRef.current = true;
    setRun((prev) => (prev ? { ...prev, status: 'paused' } : prev));
    setIsGenerating(false);
  };

  const handleResume = () => {
    if (!run || (run.status !== 'paused' && run.status !== 'stopped')) {
      return;
    }
    pauseRequestedRef.current = false;
    stopRequestedRef.current = false;
    setRun((prev) => (prev ? { ...prev, status: 'running' } : prev));
    setIsGenerating(true);
    void processRun();
  };

  const handleStopAfterCurrent = () => {
    if (!run || (run.status !== 'running' && run.status !== 'paused')) {
      return;
    }
    pauseRequestedRef.current = false;
    stopRequestedRef.current = true;
    setRun((prev) => (prev ? { ...prev, status: 'stopping' } : prev));
  };

  const handleRetryFailed = () => {
    if (!run) {
      return;
    }

    setRun((prev) => {
      if (!prev) {
        return prev;
      }

      const chunks = prev.chunks.map((chunk) =>
        chunk.status === 'failed'
          ? {
              ...chunk,
              status: 'pending' as ChunkStatus,
              attempts: 0,
              error: undefined,
            }
          : chunk,
      );

      return {
        ...prev,
        chunks,
        status: 'stopped',
      };
    });
  };

  const handleOpenStory = () => {
    if (!run?.storyId) {
      return;
    }
    setSelectedStoryId(run.storyId);
    navigate({ to: '/stories' });
  };

  const handleExportStory = () => {
    if (!run?.storyId) {
      return;
    }
    exportStoryAudio.mutate({
      storyId: run.storyId,
      storyName: run.storyName,
    });
  };

  const canStart =
    !run ||
    run.status === 'completed' ||
    run.status === 'completed_with_errors' ||
    (run.status === 'stopped' && !hasPendingRunChunks);
  const canResume =
    !!run && (run.status === 'paused' || run.status === 'stopped') && hasPendingRunChunks;
  const canRetryFailed =
    !!run && run.chunks.some((chunk) => chunk.status === 'failed') && run.status !== 'running';
  const isRunActive = run?.status === 'running' || run?.status === 'stopping';

  return (
    <div
      className={cn(
        'flex flex-col h-full min-h-0 overflow-hidden',
        isPlayerVisible ? BOTTOM_SAFE_AREA_PADDING : 'pb-4',
      )}
    >
      <div className="flex items-center justify-between mb-4 px-1">
        <div>
          <h2 className="text-2xl font-bold">Audiobook</h2>
          <p className="text-sm text-muted-foreground">
            Load a long TXT, edit it, then generate chunk-by-chunk into a Story.
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Generation is automatically saved as a Story while chunks are processed.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,text/plain"
            className="hidden"
            onChange={handleTextFileChange}
          />
          <Button
            variant="outline"
            onClick={handlePickTextFile}
            disabled={run?.status === 'running' || run?.status === 'stopping'}
          >
            <Upload className="mr-2 h-4 w-4" />
            Pick TXT
          </Button>
          <Button variant="outline" onClick={handleResetText} disabled={!isDirty || !originalText}>
            <RotateCcw className="mr-2 h-4 w-4" />
            Reset Text
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_340px] gap-6 flex-1 min-h-0 overflow-hidden">
        <Card className="min-h-0 flex flex-col">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Source Text</CardTitle>
            <div className="text-xs text-muted-foreground flex items-center gap-2">
              <span>{fileName || 'No file selected'}</span>
              {isDirty && <span className="text-accent font-medium">Edited</span>}
            </div>
            <div className="text-xs text-muted-foreground">
              {charCount.toLocaleString()} chars • {wordCount.toLocaleString()} words •{' '}
              {lineCount.toLocaleString()} lines
            </div>
          </CardHeader>
          <CardContent className="pt-0 flex-1 min-h-0 flex flex-col overflow-hidden">
            {hasText && (
              <div className="mb-3 rounded-md border border-yellow-500/40 bg-yellow-500/10 p-3 text-sm flex items-start gap-2 shrink-0">
                <AlertTriangle className="h-4 w-4 mt-0.5 text-yellow-400 shrink-0" />
                <div>
                  <p className="font-medium text-foreground">
                    Everything visible in this editor will be converted to audio.
                  </p>
                  <p className="text-muted-foreground">
                    Cleanup tip: check the beginning and end for title pages, legal notes,
                    appendices, or trailing metadata you do not want narrated.
                  </p>
                </div>
              </div>
            )}
            <Textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Pick a .txt file or paste your book text here."
              className="flex-1 min-h-0 resize-none font-mono text-sm overflow-y-auto scrollbar-visible"
              disabled={run?.status === 'running' || run?.status === 'stopping'}
            />
          </CardContent>
        </Card>

        <div className="flex flex-col gap-4 min-h-0 h-full overflow-y-auto scrollbar-visible pr-1">
          <Card className="shrink-0">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between gap-2">
                <CardTitle className="text-lg">Chunk Summary</CardTitle>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={() => setIsSummaryOpen((prev) => !prev)}
                  aria-label={isSummaryOpen ? 'Collapse chunk summary' : 'Expand chunk summary'}
                >
                  {isSummaryOpen ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </CardHeader>
            {isSummaryOpen && (
              <CardContent className="pt-0 space-y-3">
                <div className="rounded-md border p-3 text-sm space-y-1">
                  <div className="text-muted-foreground">
                    {chunkStats.total.toLocaleString()} chunks •{' '}
                    {chunkStats.completed.toLocaleString()} done •{' '}
                    {chunkStats.failed.toLocaleString()} failed •{' '}
                    {chunkStats.pending.toLocaleString()} pending
                  </div>
                  <div className="text-muted-foreground">
                    Progress: {chunkStats.progress}%{chunkStats.running > 0 ? ' (running)' : ''}
                  </div>
                  {run?.status && (
                    <div className="text-muted-foreground capitalize">Run status: {run.status}</div>
                  )}
                </div>

                {oversizedPreparedChunks.length > 0 && !run && (
                  <div className="rounded-md border border-yellow-500/40 bg-yellow-500/10 p-3 text-sm flex items-start gap-2">
                    <AlertTriangle className="h-4 w-4 mt-0.5 text-yellow-400 shrink-0" />
                    <div>
                      {oversizedPreparedChunks.length} chunk
                      {oversizedPreparedChunks.length > 1 ? 's are' : ' is'} over{' '}
                      {HARD_MAX_CHUNK_SIZE} chars because a sentence is too long. Edit text before
                      starting generation.
                    </div>
                  </div>
                )}

                {run?.status === 'completed_with_errors' && (
                  <div className="rounded-md border border-yellow-500/40 bg-yellow-500/10 p-3 text-sm flex items-start gap-2">
                    <AlertTriangle className="h-4 w-4 mt-0.5 text-yellow-400 shrink-0" />
                    <div>
                      Generation finished with failed chunks. Retry failed chunks or inspect errors
                      below.
                    </div>
                  </div>
                )}

                <div className="flex flex-wrap gap-2">
                  <Button
                    variant={isPreviewCurrent ? 'outline' : 'default'}
                    onClick={handleQuickPreview}
                    disabled={
                      !text.trim() || !selectedProfileId || isRunActive || isPreviewGenerating
                    }
                    title="Generate a quick audio test from the first 5 sentences only"
                  >
                    {isPreviewGenerating ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <FlaskConical className="mr-2 h-4 w-4" />
                    )}
                    Quick Preview (5 sentences)
                  </Button>

                  {canStart && (
                    <Button
                      variant={isPreviewCurrent ? 'default' : 'outline'}
                      onClick={handleStartGeneration}
                      disabled={
                        !text.trim() || !selectedProfileId || oversizedPreparedChunks.length > 0
                      }
                    >
                      <Play className="mr-2 h-4 w-4" />
                      Start Generation
                    </Button>
                  )}

                  {run?.status === 'running' && (
                    <>
                      <Button variant="outline" onClick={handlePause}>
                        <Pause className="mr-2 h-4 w-4" />
                        Pause
                      </Button>
                      <Button variant="outline" onClick={handleStopAfterCurrent}>
                        <Square className="mr-2 h-4 w-4" />
                        Stop After Current
                      </Button>
                    </>
                  )}

                  {run?.status === 'stopping' && (
                    <Button variant="outline" disabled>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Stopping...
                    </Button>
                  )}

                  {canResume && (
                    <Button variant="outline" onClick={handleResume}>
                      <Play className="mr-2 h-4 w-4" />
                      Resume
                    </Button>
                  )}

                  {canRetryFailed && (
                    <Button variant="outline" onClick={handleRetryFailed}>
                      <RotateCcw className="mr-2 h-4 w-4" />
                      Retry Failed
                    </Button>
                  )}

                  {run?.storyId && (
                    <>
                      <Button variant="outline" onClick={handleOpenStory}>
                        <BookOpen className="mr-2 h-4 w-4" />
                        Open Story
                      </Button>
                      <Button
                        variant="outline"
                        onClick={handleExportStory}
                        disabled={exportStoryAudio.isPending}
                      >
                        {exportStoryAudio.isPending ? (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : (
                          <Download className="mr-2 h-4 w-4" />
                        )}
                        Export Audio
                      </Button>
                    </>
                  )}
                </div>

                <p className="text-xs text-muted-foreground">
                  {run?.storyId
                    ? `Auto-saving to Story: "${run.storyName}".`
                    : 'When you press Start Generation, a new Story is created automatically and saved chunk-by-chunk.'}
                </p>
                <p className="text-xs text-muted-foreground">
                  Quick Preview uses only the first {QUICK_PREVIEW_SENTENCE_LIMIT} sentences and
                  does not save to a Story.
                </p>
                <p className="text-xs text-muted-foreground">
                  {isPreviewCurrent
                    ? 'Preview is up to date. Start Generation is now the primary action.'
                    : 'Run Quick Preview first. After a successful preview, Start Generation becomes the primary action.'}
                </p>
              </CardContent>
            )}
          </Card>

          <Card className="shrink-0">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between gap-2">
                <CardTitle className="text-lg">Generation Setup</CardTitle>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={() => setIsSetupOpen((prev) => !prev)}
                  aria-label={isSetupOpen ? 'Collapse generation setup' : 'Expand generation setup'}
                >
                  {isSetupOpen ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </CardHeader>
            {isSetupOpen && (
              <CardContent className="pt-0 space-y-4">
                <div className="grid grid-cols-1 gap-3">
                  <div className="space-y-2">
                    <Label>Voice</Label>
                    <Select
                      value={selectedProfileId || ''}
                      onValueChange={(value) => setSelectedProfileId(value || null)}
                      disabled={!profiles || profiles.length === 0}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select a voice" />
                      </SelectTrigger>
                      <SelectContent>
                        {profiles?.map((profile) => (
                          <SelectItem key={profile.id} value={profile.id}>
                            {profile.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Language</Label>
                    <Select
                      value={language}
                      onValueChange={(value) => setLanguage(value as LanguageCode)}
                      disabled={run?.status === 'running' || run?.status === 'stopping'}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {LANGUAGE_OPTIONS.map((option) => (
                          <SelectItem key={option.value} value={option.value}>
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Model Size</Label>
                    <Select
                      value={modelSize}
                      onValueChange={(value) => setModelSize(value as '1.7B' | '0.6B')}
                      disabled={run?.status === 'running' || run?.status === 'stopping'}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1.7B">Qwen3-TTS 1.7B</SelectItem>
                        <SelectItem value="0.6B">Qwen3-TTS 0.6B</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Target Chunk Size</Label>
                    <Input
                      type="number"
                      min={500}
                      max={HARD_MAX_CHUNK_SIZE}
                      step={100}
                      value={targetChunkSize}
                      onChange={(event) => {
                        const value = Number(event.target.value);
                        if (Number.isNaN(value)) {
                          return;
                        }
                        setTargetChunkSize(value);
                      }}
                      disabled={run?.status === 'running' || run?.status === 'stopping'}
                    />
                    <p className="text-xs text-muted-foreground">
                      Hard max: {HARD_MAX_CHUNK_SIZE} chars per chunk.
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Delivery Instructions (optional)</Label>
                  <Textarea
                    value={instruct}
                    onChange={(event) => setInstruct(event.target.value)}
                    maxLength={500}
                    className="min-h-[70px]"
                    placeholder="e.g. calm, clear narration with steady pacing"
                    disabled={run?.status === 'running' || run?.status === 'stopping'}
                  />
                </div>
              </CardContent>
            )}
          </Card>

          <Card className="shrink-0">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between gap-2">
                <CardTitle className="text-lg">Chunks</CardTitle>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={() => setIsChunksOpen((prev) => !prev)}
                  aria-label={isChunksOpen ? 'Collapse chunks' : 'Expand chunks'}
                >
                  {isChunksOpen ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </CardHeader>
            {isChunksOpen && (
              <CardContent className="pt-0 max-h-[46vh] overflow-y-auto scrollbar-visible">
                <div className="space-y-2">
                  {previewChunks.length === 0 && (
                    <div className="text-sm text-muted-foreground py-8 text-center">
                      Pick a TXT file or paste text to preview chunks.
                    </div>
                  )}

                  {previewChunks.map((chunk, index) => {
                    const isOversized = chunk.charCount > HARD_MAX_CHUNK_SIZE;
                    const statusColor =
                      chunk.status === 'done'
                        ? 'text-green-500'
                        : chunk.status === 'failed'
                          ? 'text-red-500'
                          : chunk.status === 'running'
                            ? 'text-accent'
                            : isOversized
                              ? 'text-yellow-500'
                              : 'text-muted-foreground';

                    return (
                      <div key={chunk.id} className="rounded-md border p-3">
                        <div className="flex items-center justify-between gap-2 text-xs mb-1">
                          <div className="font-medium">
                            Chunk {index + 1} • {chunk.charCount.toLocaleString()} chars
                          </div>
                          <div className={cn('font-medium capitalize', statusColor)}>
                            {chunk.status}
                            {chunk.status === 'running' && (
                              <Loader2 className="inline-block ml-1 h-3 w-3 animate-spin" />
                            )}
                          </div>
                        </div>
                        <p className="text-sm text-muted-foreground line-clamp-2">{chunk.text}</p>
                        {chunk.error && (
                          <p className="text-xs text-red-500 mt-1">
                            Attempt {chunk.attempts}/{MAX_CHUNK_RETRIES}: {chunk.error}
                          </p>
                        )}
                        {!chunk.error && isOversized && (
                          <p className="text-xs text-yellow-500 mt-1">
                            Oversized sentence chunk. Edit source text before generation.
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
