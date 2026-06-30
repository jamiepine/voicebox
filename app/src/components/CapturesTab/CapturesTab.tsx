import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from '@tanstack/react-router';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { save } from '@tauri-apps/plugin-dialog';
import { writeFile, writeTextFile } from '@tauri-apps/plugin-fs';
import {
  Captions,
  Check,
  ChevronDown,
  CircleDot,
  Copy,
  Download,
  FileAudio,
  FileText,
  Loader2,
  Mic,
  Settings2,
  Sparkles,
  Square,
  Trash2,
  Upload,
  Volume2,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AudioBars } from '@/components/AudioBars';
import { CapturePill } from '@/components/CapturePill/CapturePill';
import { CaptureInlinePlayer } from '@/components/CapturesTab/CaptureInlinePlayer';
import { DictationReadinessChecklist } from '@/components/CapturesTab/DictationReadinessChecklist';
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Textarea } from '@/components/ui/textarea';
import {
  ListPane,
  ListPaneHeader,
  ListPaneScroll,
  ListPaneSearch,
  ListPaneTitle,
  ListPaneTitleRow,
} from '@/components/ListPane';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type {
  CaptureListResponse,
  CaptureResponse,
  CaptureSource,
  VoiceProfileResponse,
} from '@/lib/api/types';
import type { LanguageCode } from '@/lib/constants/languages';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { useCaptureRecordingSession } from '@/lib/hooks/useCaptureRecordingSession';
import { useDictationReadiness } from '@/lib/hooks/useDictationReadiness';
import { useCaptureSettings } from '@/lib/hooks/useSettings';
import { cn } from '@/lib/utils/cn';
import { formatAbsoluteDate, formatDate } from '@/lib/utils/format';
import { displayLabelForKey, modifierSideHint } from '@/lib/utils/keyCodes';
import { useGenerationStore } from '@/stores/generationStore';
import { usePlayerStore } from '@/stores/playerStore';

const CAPTURE_AUDIO_MIME = 'audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm';

function formatDuration(ms?: number | null): string {
  if (!ms || ms < 0) return '0:00';
  const total = Math.round(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function ChordKeys({ keys }: { keys: string[] }) {
  if (keys.length === 0) return null;
  return (
    <div className="flex items-center gap-1">
      {keys.map((k) => {
        const side = modifierSideHint(k);
        return (
          <span
            key={k}
            className="relative inline-flex items-center justify-center h-6 min-w-[1.5rem] px-1.5 rounded-md border border-border bg-muted/60 font-mono text-[11px] font-medium shadow-sm text-foreground"
          >
            {displayLabelForKey(k)}
            {side ? (
              <span className="absolute -top-1 -right-1 h-3 min-w-[0.75rem] px-0.5 rounded-sm bg-accent text-[7px] font-bold leading-none flex items-center justify-center text-accent-foreground">
                {side}
              </span>
            ) : null}
          </span>
        );
      })}
    </div>
  );
}

function SourceBadge({ source }: { source: CaptureSource }) {
  const { t } = useTranslation();
  const Icon = source === 'dictation' ? Mic : source === 'recording' ? CircleDot : FileAudio;
  const label =
    source === 'dictation'
      ? t('captures.source.dictation')
      : source === 'recording'
        ? t('captures.source.recording')
        : t('captures.source.file');
  return (
    <Badge
      variant="secondary"
      className="h-5 px-1.5 text-[10px] gap-1 font-medium bg-muted/60 text-muted-foreground"
    >
      <Icon className="h-2.5 w-2.5" />
      {label}
    </Badge>
  );
}

type PlaybackState = 'idle' | 'generating' | 'playing';

export function CapturesTab() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);

  const snippetOf = (capture: CaptureResponse): string => {
    const source = capture.transcript_refined || capture.transcript_raw || '';
    return source.trim() || t('captures.snippetEmpty');
  };

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [showRefined, setShowRefined] = useState(true);
  const [launchedPlayAsId, setLaunchedPlayAsId] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

  const audioUrl = usePlayerStore((s) => s.audioUrl);
  const playerAudioId = usePlayerStore((s) => s.audioId);
  const playerIsPlaying = usePlayerStore((s) => s.isPlaying);
  const isPlayerVisible = !!audioUrl;

  const setIsPlaying = usePlayerStore((s) => s.setIsPlaying);

  const addPendingGeneration = useGenerationStore((s) => s.addPendingGeneration);
  const pendingGenerationIds = useGenerationStore((s) => s.pendingGenerationIds);

  const { settings: captureSettings, update: updateCaptureSettings } = useCaptureSettings();
  const sttModel = captureSettings?.stt_model ?? 'turbo';
  const llmModel = captureSettings?.llm_model ?? '0.6B';
  const hotkeyEnabled = captureSettings?.hotkey_enabled ?? false;
  const pushToTalkKeys = captureSettings?.chord_push_to_talk_keys ?? [];
  const toggleToTalkKeys = captureSettings?.chord_toggle_to_talk_keys ?? [];
  const readiness = useDictationReadiness();

  const session = useCaptureRecordingSession({
    onCaptureCreated: (capture) => setSelectedId(capture.id),
  });

  const { data: capturesData, isLoading: capturesLoading } = useQuery({
    queryKey: ['captures'],
    queryFn: () => apiClient.listCaptures(200, 0),
  });

  const { data: profiles } = useQuery({
    queryKey: ['profiles'],
    queryFn: () => apiClient.listProfiles(),
  });

  const captures = capturesData?.items ?? [];

  // Keep a selection. If the current selection disappears (e.g. deletion),
  // fall through to the first capture, then to null.
  useEffect(() => {
    if (!captures.length) {
      if (selectedId !== null) setSelectedId(null);
      return;
    }
    if (!selectedId || !captures.find((c) => c.id === selectedId)) {
      setSelectedId(captures[0].id);
    }
  }, [captures, selectedId]);

  // Live sync from sibling Tauri webviews (the floating dictate window).
  // ``capture:created`` carries the full row so we can seed the cache before
  // the refetch lands and focus the new capture in one shot — without the
  // seed, the selection-guard effect would snap back to ``captures[0]`` in
  // the race window between ``setSelectedId(new)`` and the refetched list
  // actually containing the new row.
  useEffect(() => {
    const unlistens: Promise<UnlistenFn>[] = [];
    unlistens.push(
      listen<{ capture: CaptureResponse }>('capture:created', (event) => {
        const capture = event.payload?.capture;
        if (capture) {
          queryClient.setQueryData<CaptureListResponse>(['captures'], (prev) => {
            if (!prev) return prev;
            if (prev.items.some((c) => c.id === capture.id)) return prev;
            return { ...prev, items: [capture, ...prev.items], total: prev.total + 1 };
          });
          setSelectedId(capture.id);
        }
        queryClient.invalidateQueries({ queryKey: ['captures'] });
      }),
    );
    unlistens.push(
      listen('capture:updated', () => {
        queryClient.invalidateQueries({ queryKey: ['captures'] });
      }),
    );
    return () => {
      for (const p of unlistens) p.then((fn) => fn()).catch(() => {});
    };
  }, [queryClient]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return captures;
    return captures.filter((c) => {
      const raw = (c.transcript_raw || '').toLowerCase();
      const refined = (c.transcript_refined || '').toLowerCase();
      return raw.includes(q) || refined.includes(q);
    });
  }, [search, captures]);

  const selected = captures.find((c) => c.id === selectedId) ?? null;
  // Source of truth is capture_settings.default_playback_voice_id, shared
  // with Settings → Captures and the MCP global default. Stale ids (e.g.
  // referenced profile was deleted) fall through to the first profile.
  const storedVoiceId = captureSettings?.default_playback_voice_id ?? null;
  const playAsVoice =
    (storedVoiceId && profiles?.find((p) => p.id === storedVoiceId)) ||
    profiles?.[0] ||
    null;
  const playAsVoiceId = playAsVoice?.id ?? null;

  const deleteMutation = useMutation({
    mutationFn: async (captureId: string) => apiClient.deleteCapture(captureId),
    onSuccess: () => {
      setDeleteDialogOpen(false);
      queryClient.invalidateQueries({ queryKey: ['captures'] });
    },
    onError: (err: Error) => {
      toast({ title: t('captures.toast.deleteFailed'), description: err.message, variant: 'destructive' });
    },
  });

  const playAsMutation = useMutation({
    mutationFn: async ({ capture, voice }: { capture: CaptureResponse; voice: VoiceProfileResponse }) => {
      const text = capture.transcript_refined || capture.transcript_raw;
      if (!text.trim()) throw new Error(t('captures.noTranscriptError'));
      const language = (capture.language || voice.language) as LanguageCode;
      // Preset profiles (Kokoro etc.) reject the qwen default — honor the
      // profile's stored engine preference. Cloned profiles without an
      // override fall through to whatever the backend picks.
      const engine = voice.default_engine as
        | 'qwen' | 'qwen_custom_voice' | 'luxtts' | 'chatterbox'
        | 'chatterbox_turbo' | 'tada' | 'kokoro'
        | undefined;
      return apiClient.generateSpeech({
        profile_id: voice.id,
        text,
        language,
        engine,
      });
    },
    onSuccess: (result) => {
      // /generate is queue-based — it returns a generating row with an empty
      // audio_path. Hand the id to the global SSE handler which polls
      // /generation/{id}/status and triggers autoplay on completion.
      setLaunchedPlayAsId(result.id);
      addPendingGeneration(result.id);
    },
    onError: (err: Error) => {
      toast({ title: t('captures.toast.playAsFailed'), description: err.message, variant: 'destructive' });
    },
  });

  const playbackState: PlaybackState = playAsMutation.isPending
    ? 'generating'
    : launchedPlayAsId && pendingGenerationIds.has(launchedPlayAsId)
      ? 'generating'
      : launchedPlayAsId && playerAudioId === launchedPlayAsId && playerIsPlaying
        ? 'playing'
        : 'idle';

  const handleUploadClick = () => uploadInputRef.current?.click();

  const handleUploadFile = (e: React.ChangeEvent<HTMLInputElement>, source: CaptureSource) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    session.uploadFile(file, source);
  };

  const handleCopy = async () => {
    if (!selected) return;
    const text = showRefined
      ? selected.transcript_refined || selected.transcript_raw
      : selected.transcript_raw;
    try {
      await navigator.clipboard.writeText(text || '');
      toast({ title: t('captures.toast.transcriptCopied') });
    } catch {
      toast({ title: t('captures.toast.copyFailed'), variant: 'destructive' });
    }
  };

  const exportToastSuccess = (path: string) => {
    const name = path.split(/[\\/]/).pop() ?? path;
    toast({ title: t('captures.toast.exportSuccess', { path: name }) });
  };

  const exportToastError = (err: unknown) => {
    toast({
      title: t('captures.toast.exportFailed'),
      description: err instanceof Error ? err.message : String(err),
      variant: 'destructive',
    });
  };

  const handleExportAudio = async () => {
    if (!selected) return;
    try {
      const dest = await save({
        defaultPath: `capture_${selected.id.slice(0, 8)}.wav`,
        filters: [{ name: 'Audio', extensions: ['wav'] }],
      });
      if (!dest) return;
      const res = await fetch(apiClient.getCaptureAudioUrl(selected.id));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const buf = new Uint8Array(await res.arrayBuffer());
      await writeFile(dest, buf);
      exportToastSuccess(dest);
    } catch (err) {
      exportToastError(err);
    }
  };

  const handleExportTranscript = async () => {
    if (!selected) return;
    const text = (selected.transcript_refined || selected.transcript_raw || '').trim();
    if (!text) {
      toast({ title: t('captures.toast.exportEmpty'), variant: 'destructive' });
      return;
    }
    try {
      const dest = await save({
        defaultPath: `capture_${selected.id.slice(0, 8)}.txt`,
        filters: [{ name: 'Text', extensions: ['txt'] }],
      });
      if (!dest) return;
      await writeTextFile(dest, text);
      exportToastSuccess(dest);
    } catch (err) {
      exportToastError(err);
    }
  };

  const buildCaptureMarkdown = (capture: CaptureResponse): string => {
    const lines: string[] = [];
    lines.push(`# Capture ${capture.id}`, '');
    lines.push(`- **Source:** ${capture.source}`);
    lines.push(`- **Created:** ${capture.created_at}`);
    if (capture.duration_ms != null) lines.push(`- **Duration:** ${formatDuration(capture.duration_ms)}`);
    if (capture.language) lines.push(`- **Language:** ${capture.language}`);
    if (capture.stt_model) lines.push(`- **STT model:** ${capture.stt_model}`);
    if (capture.llm_model) lines.push(`- **LLM model:** ${capture.llm_model}`);
    lines.push('');
    if (capture.transcript_refined?.trim()) {
      lines.push('## Refined transcript', '', capture.transcript_refined.trim(), '');
    }
    if (capture.transcript_raw?.trim()) {
      lines.push('## Raw transcript', '', capture.transcript_raw.trim(), '');
    }
    return lines.join('\n');
  };

  const handleExportMarkdown = async () => {
    if (!selected) return;
    const hasContent = (selected.transcript_refined || selected.transcript_raw || '').trim();
    if (!hasContent) {
      toast({ title: t('captures.toast.exportEmpty'), variant: 'destructive' });
      return;
    }
    try {
      const dest = await save({
        defaultPath: `capture_${selected.id.slice(0, 8)}.md`,
        filters: [{ name: 'Markdown', extensions: ['md'] }],
      });
      if (!dest) return;
      await writeTextFile(dest, buildCaptureMarkdown(selected));
      exportToastSuccess(dest);
    } catch (err) {
      exportToastError(err);
    }
  };

  const handlePlayAs = (voice?: VoiceProfileResponse) => {
    if (!selected) return;
    // Stop the current playback when the button is in its 'playing' state
    // and the user clicked the main button without picking a new voice.
    if (!voice && playbackState === 'playing') {
      setIsPlaying(false);
      return;
    }
    const target = voice ?? playAsVoice;
    if (!target) {
      toast({
        title: t('captures.toast.noVoice'),
        description: t('captures.toast.noVoiceDescription'),
        variant: 'destructive',
      });
      return;
    }
    if (voice && voice.id !== playAsVoiceId) {
      updateCaptureSettings({ default_playback_voice_id: voice.id });
    }
    playAsMutation.mutate({ capture: selected, voice: target });
  };

  return (
    <div className="h-full flex gap-0 overflow-hidden -mx-8">
      <input
        ref={uploadInputRef}
        type="file"
        accept={CAPTURE_AUDIO_MIME}
        onChange={(e) => handleUploadFile(e, 'file')}
        className="hidden"
      />
      <input
        ref={fileInputRef}
        type="file"
        accept={CAPTURE_AUDIO_MIME}
        onChange={(e) => handleUploadFile(e, 'file')}
        className="hidden"
      />

      {/* Left: capture list */}
      <div className="w-[340px] shrink-0">
        <ListPane>
          <ListPaneHeader>
            <ListPaneTitleRow>
              <ListPaneTitle>{t('captures.title')}</ListPaneTitle>
              <Badge
                variant="secondary"
                className="h-5 px-1.5 -ml-2 text-[10px] font-medium text-accent bg-accent/10 border border-accent/20"
              >
                {t('captures.beta')}
              </Badge>
            </ListPaneTitleRow>
            <ListPaneSearch
              value={search}
              onChange={setSearch}
              placeholder={t('captures.searchPlaceholder')}
            />
          </ListPaneHeader>

          <ListPaneScroll className={cn(isPlayerVisible && BOTTOM_SAFE_AREA_PADDING)}>
            <div className="px-4 pb-6 space-y-1">
              {capturesLoading ? (
                <div className="px-4 py-12 flex items-center justify-center text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                </div>
              ) : filtered.length === 0 ? (
                <div className="px-4 py-12 text-center text-sm text-muted-foreground">
                  {search ? (
                    <p>{t('captures.empty.noMatches', { query: search })}</p>
                  ) : (
                    <p>{t('captures.empty.none')}</p>
                  )}
                </div>
              ) : (
                filtered.map((capture) => {
                const isActive = selectedId === capture.id;
                const refined = !!capture.transcript_refined;
                return (
                  <button
                    type="button"
                    key={capture.id}
                    onClick={() => setSelectedId(capture.id)}
                    className={cn(
                      'w-full text-left p-3 rounded-lg transition-colors block',
                      isActive
                        ? 'bg-muted/70 border border-border'
                        : 'border border-transparent hover:bg-muted/30',
                    )}
                  >
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="text-[11px] text-muted-foreground font-medium">
                        {formatDate(capture.created_at)}
                      </span>
                      <div className="flex-1" />
                      <span className="text-[10px] text-muted-foreground/70 tabular-nums">
                        {formatDuration(capture.duration_ms)}
                      </span>
                    </div>
                    <div className="text-[13px] text-foreground/90 line-clamp-2 leading-snug mb-2">
                      {snippetOf(capture)}
                    </div>
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <SourceBadge source={capture.source} />
                      {refined && (
                        <Badge
                          variant="secondary"
                          className="h-5 px-1.5 text-[10px] gap-1 font-medium bg-accent/10 text-accent border border-accent/20"
                        >
                          <Sparkles className="h-2.5 w-2.5" />
                          {t('captures.transcript.refined')}
                        </Badge>
                      )}
                    </div>
                  </button>
                );
              })
            )}
            </div>
          </ListPaneScroll>
        </ListPane>
      </div>

      {/* Right: capture detail */}
      <div className="flex-1 flex flex-col relative overflow-hidden min-w-0">
        <div className="absolute top-0 left-0 right-0 h-20 bg-gradient-to-b from-background to-transparent z-10 pointer-events-none" />

        {/* Top action bar */}
        <div className="absolute top-0 left-0 right-0 z-20 px-8">
          <div className="flex items-center gap-3 py-4">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="w-1.5 h-1.5 rounded-full bg-accent" />
              <span>
                {t('captures.header.modelSummary', {
                  stt: sttModel.charAt(0).toUpperCase() + sttModel.slice(1),
                  llm: llmModel,
                })}
              </span>
            </div>
            <div className="flex-1" />
            {session.pillState !== 'hidden' && (
              <CapturePill
                state={session.pillState}
                elapsedMs={session.pillElapsedMs}
                errorMessage={session.errorMessage}
                onDismiss={session.dismissError}
                onStop={session.isRecording ? session.stopRecording : undefined}
              />
            )}
            {session.pillState === 'hidden' && (
              <>
                <Button variant="outline" asChild>
                  <Link to="/settings/captures">
                    <Settings2 className="mr-2 h-4 w-4" />
                    {t('captures.actions.configure')}
                  </Link>
                </Button>
                {readiness.canRecord && (
                  <Button
                    variant="outline"
                    onClick={handleUploadClick}
                    disabled={session.isUploading}
                  >
                    {session.isUploading ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Upload className="h-4 w-4 mr-2" />
                    )}
                    {session.isUploading ? t('captures.actions.importing') : t('captures.actions.import')}
                  </Button>
                )}
              </>
            )}
            {/* Hide Dictate when recording readiness fails so the user can't kick off
                a capture that has nowhere to land. Stop stays visible if a
                recording is somehow already in flight (e.g. a model was
                uninstalled mid-record) so the user can always cancel. */}
            {(readiness.canRecord || session.isRecording) && (
              <Button
                onClick={session.toggleRecording}
                disabled={session.isUploading && !session.isRecording}
                className="relative overflow-hidden transition-all bg-accent text-accent-foreground hover:bg-accent/90"
              >
                {session.isRecording ? (
                  <>
                    <Square className="h-4 w-4 mr-2 fill-current" />
                    {t('captures.actions.stop')}
                  </>
                ) : (
                  <>
                    <Mic className="h-4 w-4 mr-2" />
                    {t('captures.actions.dictate')}
                  </>
                )}
              </Button>
            )}
          </div>
        </div>

        {selected ? (
          <div
            className={cn(
              'flex-1 overflow-y-auto pt-20 px-8 pb-8',
              isPlayerVisible && BOTTOM_SAFE_AREA_PADDING,
            )}
          >
            {/* Meta row */}
            <div className="flex items-center gap-3 mb-4 text-xs text-muted-foreground">
              <span>{formatAbsoluteDate(selected.created_at)}</span>
              {selected.language && (
                <>
                  <span className="text-muted-foreground/40">·</span>
                  <span>{selected.language.toUpperCase()}</span>
                </>
              )}
              <span className="text-muted-foreground/40">·</span>
              <SourceBadge source={selected.source} />
            </div>

            {/* Audio player card */}
            <div className="rounded-xl border border-border bg-muted/20 p-4 mb-6">
              <CaptureInlinePlayer
                audioUrl={apiClient.getCaptureAudioUrl(selected.id)}
                fallbackDurationMs={selected.duration_ms}
              />
            </div>

            {/* Transcript header */}
            <div className="flex items-center gap-3 mb-3">
              <div className="inline-flex rounded-md bg-muted/40 p-0.5 border border-border">
                <button
                  type="button"
                  onClick={() => setShowRefined(true)}
                  disabled={!selected.transcript_refined}
                  className={cn(
                    'px-3 py-1 text-xs font-medium rounded transition-colors',
                    showRefined && selected.transcript_refined
                      ? 'bg-background shadow-sm text-foreground'
                      : 'text-muted-foreground hover:text-foreground disabled:opacity-40',
                  )}
                >
                  <Sparkles className="h-3 w-3 inline-block mr-1 -translate-y-px" />
                  {t('captures.transcript.refined')}
                </button>
                <button
                  type="button"
                  onClick={() => setShowRefined(false)}
                  className={cn(
                    'px-3 py-1 text-xs font-medium rounded transition-colors',
                    !showRefined || !selected.transcript_refined
                      ? 'bg-background shadow-sm text-foreground'
                      : 'text-muted-foreground hover:text-foreground',
                  )}
                >
                  <Captions className="h-3 w-3 inline-block mr-1 -translate-y-px" />
                  {t('captures.transcript.raw')}
                </button>
              </div>
              <div className="flex-1" />
              <span className="text-xs text-muted-foreground">
                {showRefined && selected.transcript_refined
                  ? t('captures.transcript.refinedHint', { model: selected.llm_model ?? llmModel })
                  : selected.stt_model
                    ? t('captures.transcript.rawHint', { model: selected.stt_model })
                    : null}
              </span>
            </div>

            {/* Transcript body */}
            <div className="rounded-xl border border-border bg-muted/10">
              <Textarea
                key={`${selected.id}-${showRefined}`}
                defaultValue={
                  showRefined && selected.transcript_refined
                    ? selected.transcript_refined
                    : selected.transcript_raw
                }
                readOnly
                className="text-[15px] leading-relaxed min-h-[260px] border-0 bg-transparent resize-none focus-visible:ring-0 focus-visible:ring-offset-0 p-6"
              />
            </div>

            {/* Bottom actions */}
            <div className="flex items-center gap-2 mt-4 flex-wrap">
              <div className="inline-flex">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handlePlayAs()}
                  disabled={!playAsVoice || playAsMutation.isPending}
                  className={cn(
                    'gap-2 rounded-r-none border-r-0 pr-3 pl-2 transition-colors',
                    playbackState !== 'idle' &&
                      'border-accent/50 text-foreground bg-accent/10 hover:bg-accent/15 hover:text-foreground hover:border-accent/50',
                  )}
                >
                  {playbackState === 'generating' ? (
                    <>
                      <AudioBars mode="generating" className="h-3.5" />
                      {t('captures.actions.playAsGenerating')}
                    </>
                  ) : playbackState === 'playing' ? (
                    <>
                      <Square className="h-3 w-3 fill-current" />
                      {playAsVoice
                        ? t('captures.actions.playAsStop', { name: playAsVoice.name })
                        : t('captures.actions.playAsStopFallback')}
                    </>
                  ) : (
                    <>
                      <Volume2 className="h-3.5 w-3.5" />
                      {playAsVoice
                        ? t('captures.actions.playAs', { name: playAsVoice.name })
                        : t('captures.actions.playAsFallback')}
                    </>
                  )}
                </Button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      className={cn(
                        'rounded-l-none px-2 transition-colors',
                        playbackState !== 'idle' &&
                          'border-accent/50 bg-accent/10 hover:bg-accent/15 hover:text-foreground hover:border-accent/50',
                      )}
                      disabled={!profiles || !profiles.length}
                    >
                      <ChevronDown className="h-3.5 w-3.5 opacity-70" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start" className="w-64">
                    <DropdownMenuLabel className="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">
                      {t('captures.actions.playAsDropdownLabel')}
                    </DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    {profiles?.map((v) => (
                      <DropdownMenuItem
                        key={v.id}
                        onClick={() => handlePlayAs(v)}
                        className="py-2"
                      >
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium truncate">{v.name}</div>
                          <div className="text-[11px] text-muted-foreground truncate">
                            {v.description || v.language.toUpperCase()}
                          </div>
                        </div>
                        {v.id === playAsVoiceId && (
                          <Check className="h-3.5 w-3.5 text-accent shrink-0" />
                        )}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
              <Button variant="outline" size="sm" onClick={handleCopy}>
                <Copy className="h-3.5 w-3.5 mr-1.5" />
                {t('captures.actions.copy')}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => session.refine(selected.id)}
                disabled={session.isRefining}
              >
                {session.isRefining ? (
                  <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                ) : (
                  <Sparkles className="h-3.5 w-3.5 mr-1.5" />
                )}
                {selected.transcript_refined
                  ? t('captures.actions.reRefine')
                  : t('captures.actions.refine')}
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm">
                    <Download className="h-3.5 w-3.5 mr-1.5" />
                    {t('captures.actions.export')}
                    <ChevronDown className="h-3.5 w-3.5 ml-1 opacity-70" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-56">
                  <DropdownMenuLabel className="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">
                    {t('captures.actions.exportDropdownLabel')}
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={handleExportAudio}>
                    <FileAudio className="h-3.5 w-3.5 mr-2 text-muted-foreground" />
                    {t('captures.actions.exportAudio')}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={handleExportTranscript}>
                    <Captions className="h-3.5 w-3.5 mr-2 text-muted-foreground" />
                    {t('captures.actions.exportTranscript')}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={handleExportMarkdown}>
                    <FileText className="h-3.5 w-3.5 mr-2 text-muted-foreground" />
                    {t('captures.actions.exportMarkdown')}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
              <div className="flex-1" />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setDeleteDialogOpen(true)}
                disabled={deleteMutation.isPending}
                className="text-muted-foreground "
              >
                {deleteMutation.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                ) : (
                  <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                )}
                {t('captures.actions.delete')}
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground pt-20">
            {capturesLoading ? (
              <div className="text-center space-y-3">
                <Captions className="h-10 w-10 mx-auto opacity-40" />
                <p className="text-sm">{t('captures.empty.loading')}</p>
              </div>
            ) : captures.length ? (
              <div className="text-center space-y-3">
                <Captions className="h-10 w-10 mx-auto opacity-40" />
                <p className="text-sm">{t('captures.empty.pickOne')}</p>
              </div>
            ) : hotkeyEnabled && !readiness.canRecord ? (
              <DictationReadinessChecklist readiness={readiness} />
            ) : hotkeyEnabled && (pushToTalkKeys.length || toggleToTalkKeys.length) ? (
              <div className="max-w-sm mx-auto text-center space-y-5">
                <div className="space-y-2">
                  {pushToTalkKeys.length ? (
                    <div className="flex items-center justify-center gap-3">
                      <ChordKeys keys={pushToTalkKeys} />
                      <span className="text-[11px] uppercase tracking-wider text-muted-foreground">
                        {t('captures.empty.holdToRecord')}
                      </span>
                    </div>
                  ) : null}
                  {toggleToTalkKeys.length ? (
                    <div className="flex items-center justify-center gap-3">
                      <ChordKeys keys={toggleToTalkKeys} />
                      <span className="text-[11px] uppercase tracking-wider text-muted-foreground">
                        {t('captures.empty.toggleHandsFree')}
                      </span>
                    </div>
                  ) : null}
                </div>
                <p className="text-sm">
                  {t('captures.empty.pressShortcut')}
                </p>
              </div>
            ) : (
              <div className="max-w-sm mx-auto text-center space-y-3">
                <Captions className="h-10 w-10 mx-auto opacity-40" />
                <p className="text-sm">{t('captures.empty.none')}</p>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {t('captures.empty.turnOnShortcut')}
                </p>
                <Button asChild variant="outline" size="sm">
                  <Link to="/settings/captures">{t('captures.empty.openSettings')}</Link>
                </Button>
              </div>
            )}
          </div>
        )}
      </div>

      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('captures.deleteDialog.title')}</AlertDialogTitle>
            <AlertDialogDescription>{t('captures.deleteDialog.description')}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
            <AlertDialogAction asChild>
              <Button
                onClick={() => selected && deleteMutation.mutate(selected.id)}
                disabled={deleteMutation.isPending}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                {deleteMutation.isPending ? t('captures.deleteDialog.deleting') : t('common.delete')}
              </Button>
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
