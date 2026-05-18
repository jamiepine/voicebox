import {
  Ban,
  Download,
  FileArchive,
  Loader2,
  MoreHorizontal,
  Pencil,
  Play,
  Plus,
  RotateCcw,
  Scissors,
  TimerReset,
  Trash2,
  Wand2,
} from 'lucide-react';
import type { ChangeEvent } from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { AudioTrackClip } from '@/components/AudioTimeline/AudioTrackEditor';
import { AudioTrackEditor } from '@/components/AudioTimeline/AudioTrackEditor';
import {
  ListPane,
  ListPaneActions,
  ListPaneHeader,
  ListPaneScroll,
  ListPaneSearch,
  ListPaneTitle,
  ListPaneTitleRow,
} from '@/components/ListPane';
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
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import {
  getLanguageOptionsForEngine,
  LANGUAGE_CODES,
  type LanguageCode,
} from '@/lib/constants/languages';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import type {
  DubbingProjectListItemResponse,
  DubbingProjectResponse,
  DubbingSegmentResponse,
  DubbingAutoCutClipResponse,
  DubbingTempoSuggestionResponse,
} from '@/lib/api/types';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';
import { formatDate } from '@/lib/utils/format';
import { usePlatform } from '@/platform/PlatformContext';
import type { FileFilter } from '@/platform/types';
import { usePlayerStore } from '@/stores/playerStore';

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const millis = ms % 1000;
  return `${seconds}.${millis.toString().padStart(3, '0')} s`;
}

function formatDelta(ms?: number | null): string {
  if (ms == null) return '--';
  const sign = ms > 0 ? '+' : '';
  return `${sign}${ms} ms`;
}

const TARGET_CPS = 15;
const TARGET_WORDS_PER_SECOND = 2.2;

function normalizeReadableText(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

function countReadableWords(text: string): number {
  const normalized = normalizeReadableText(text)
    .toLocaleLowerCase('fr-FR')
    .replace(/['’`´]/g, ' ')
    .replace(/[^\p{L}\p{N}\s-]/gu, ' ');
  return normalized.split(/\s+/).filter(Boolean).length;
}

function getSegmentReadability(segment: DubbingSegmentResponse) {
  const durationSeconds = Math.max(0.001, segment.target_duration_ms / 1000);
  const visibleText = normalizeReadableText(segment.text);
  const characterCount = visibleText.length;
  const wordCount = countReadableWords(visibleText);
  const cps = characterCount / durationSeconds;
  const wordsPerSecond = wordCount / durationSeconds;
  return {
    characterCount,
    wordCount,
    cps,
    wordsPerSecond,
    cpsWarning: cps > TARGET_CPS,
    wordsWarning: wordsPerSecond > TARGET_WORDS_PER_SECOND,
  };
}

function readabilityBadgeClasses(isWarning: boolean): string {
  return isWarning
    ? 'border-rose-500/25 bg-rose-500/10 text-rose-300'
    : 'border-emerald-500/25 bg-emerald-500/10 text-emerald-300';
}

function formatSrtTimecode(ms: number): string {
  const safeMs = Math.max(0, Math.round(ms));
  const hours = Math.floor(safeMs / 3_600_000);
  const minutes = Math.floor((safeMs % 3_600_000) / 60_000);
  const seconds = Math.floor((safeMs % 60_000) / 1000);
  const millis = safeMs % 1000;
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds
    .toString()
    .padStart(2, '0')},${millis.toString().padStart(3, '0')}`;
}

function parseSrtTimecode(value: string): number | null {
  const match = value.trim().match(/^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})$/);
  if (!match) return null;
  const [, hours, minutes, seconds, millis] = match;
  const ms = Number(millis.padEnd(3, '0'));
  return Number(hours) * 3_600_000 + Number(minutes) * 60_000 + Number(seconds) * 1000 + ms;
}

function fitBadgeClasses(fitStatus: string): string {
  switch (fitStatus) {
    case 'exact':
      return 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20';
    case 'acceptable':
      return 'bg-sky-500/10 text-sky-300 border-sky-500/20';
    case 'warning':
      return 'bg-amber-500/10 text-amber-300 border-amber-500/20';
    case 'failed':
      return 'bg-rose-500/10 text-rose-300 border-rose-500/20';
    default:
      return 'bg-muted text-muted-foreground border-border';
  }
}

function summarizeSegmentFailure(segment: DubbingSegmentResponse): string | null {
  if (segment.generation_error) {
    return segment.generation_error;
  }
  if (segment.fit_status === 'warning' && (segment.delta_ms ?? 0) > 0) {
    return `Exceeded subtitle end by ${segment.delta_ms} ms.`;
  }
  return null;
}

async function saveBlob(
  blob: Blob,
  filename: string,
  saveFile?: (filename: string, blob: Blob, filters?: FileFilter[]) => Promise<void>,
) {
  if (saveFile) {
    await saveFile(filename, blob, [
      {
        name: 'WAV Audio',
        extensions: ['wav'],
      },
      {
        name: 'Voicebox Package',
        extensions: ['zip'],
      },
    ]);
    return;
  }

  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

type TimelinePlaybackSource = 'auto' | 'full' | 'cuts';
type Srt2VoiceEngine =
  | 'qwen'
  | 'qwen_custom_voice'
  | 'qwen_voice_design'
  | 'luxtts'
  | 'chatterbox'
  | 'chatterbox_turbo'
  | 'tada'
  | 'kokoro';
type Srt2VoiceEngineOption = {
  value: string;
  engine: Srt2VoiceEngine;
  label: string;
  modelSize?: '1B' | '3B';
};
const FULL_NARRATION_CLIP_PREFIX = 'full-narration-clip';
const AUTO_RESTART_SERVER_FOR_VRAM_RELEASE = false;
const QWEN_DEFAULT_TEMPERATURE = 0.9;
const SRT2VOICE_DEFAULT_LANGUAGE: LanguageCode = 'fr';

const SRT2VOICE_ENGINE_OPTIONS: Srt2VoiceEngineOption[] = [
  { value: 'qwen', engine: 'qwen', label: 'Qwen3-TTS 1.7B' },
  { value: 'qwen_custom_voice', engine: 'qwen_custom_voice', label: 'Qwen CustomVoice 1.7B' },
  { value: 'qwen_voice_design', engine: 'qwen_voice_design', label: 'Qwen VoiceDesign 1.7B' },
  { value: 'luxtts', engine: 'luxtts', label: 'LuxTTS' },
  { value: 'chatterbox', engine: 'chatterbox', label: 'Chatterbox' },
  { value: 'chatterbox_turbo', engine: 'chatterbox_turbo', label: 'Chatterbox Turbo' },
  { value: 'tada:1B', engine: 'tada', modelSize: '1B', label: 'TADA 1B' },
  { value: 'tada:3B', engine: 'tada', modelSize: '3B', label: 'TADA 3B Multilingual' },
  { value: 'kokoro', engine: 'kokoro', label: 'Kokoro 82M' },
];

function isSrt2VoiceEngine(value?: string | null): value is Srt2VoiceEngine {
  return (
    value === 'qwen' ||
    value === 'qwen_custom_voice' ||
    value === 'qwen_voice_design' ||
    value === 'luxtts' ||
    value === 'chatterbox' ||
    value === 'chatterbox_turbo' ||
    value === 'tada' ||
    value === 'kokoro'
  );
}

function isLanguageCode(value?: string | null): value is LanguageCode {
  return !!value && LANGUAGE_CODES.includes(value as LanguageCode);
}

function isProfileCompatibleWithSrt2VoiceEngine(
  profile: { voice_type?: string | null; preset_engine?: string | null; default_engine?: string | null },
  engine: Srt2VoiceEngine,
): boolean {
  const voiceType = profile.voice_type || 'cloned';
  if (voiceType === 'designed') return engine === 'qwen_voice_design';
  if (voiceType === 'preset') {
    const presetEngine = profile.preset_engine ?? profile.default_engine;
    if (presetEngine === 'qwen_custom_voice') return engine === 'qwen_custom_voice';
    if (presetEngine === 'qwen_voice_design') return engine === 'qwen_voice_design';
    return presetEngine === engine;
  }
  if (voiceType === 'cloned') {
    return (
      engine === 'qwen' ||
      engine === 'luxtts' ||
      engine === 'chatterbox' ||
      engine === 'chatterbox_turbo' ||
      engine === 'tada'
    );
  }
  return false;
}

function formatSeconds(ms?: number | null): string {
  if (ms == null) return '--';
  return `${(ms / 1000).toFixed(1)} s`;
}

function formatSecondsWords(ms?: number | null): string {
  if (ms == null) return '-- seconds';
  return `${(ms / 1000).toFixed(1)} seconds`;
}

function isPlausibleGenerationElapsed(durationMs?: number | null, elapsedMs?: number | null): elapsedMs is number {
  if (!durationMs || !elapsedMs || elapsedMs <= 0) return false;
  // Guard against stale pre-sidecar values computed from project age/file mtimes.
  return elapsedMs <= Math.max(30 * 60 * 1000, durationMs * 80);
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

interface DubbingFullNarrationClip {
  id: string;
  generationId: string;
  audioRevisionMs?: number | null;
  startMs: number;
  durationMs: number;
  trimStartMs: number;
  trimEndMs: number;
  track: number;
  volume: number;
}

interface PersistedDubbingTimeline {
  sourceGenerationId: string;
  sourceRevisionMs?: number | null;
  sourceDurationMs?: number | null;
  clips: DubbingFullNarrationClip[];
}

function isFullNarrationClipId(value?: string | null) {
  return !!value && value.startsWith(FULL_NARRATION_CLIP_PREFIX);
}

function getFullNarrationAudioUrl(clip: DubbingFullNarrationClip) {
  return apiClient.getAudioUrl(clip.generationId, clip.audioRevisionMs);
}

function getFullClipEffectiveDurationMs(clip: DubbingFullNarrationClip) {
  return Math.max(0, clip.durationMs - clip.trimStartMs - clip.trimEndMs);
}

function getFullClipEndMs(clip: DubbingFullNarrationClip) {
  return clip.startMs + getFullClipEffectiveDurationMs(clip);
}

function isClipAudible(clip: Pick<DubbingFullNarrationClip, 'volume'>) {
  return (clip.volume ?? 1) > 0.001;
}

function findFirstAudibleOverlap(clips: DubbingFullNarrationClip[]) {
  const audible = clips
    .filter((clip) => isClipAudible(clip) && getFullClipEffectiveDurationMs(clip) > 0)
    .sort((a, b) => a.startMs - b.startMs || a.id.localeCompare(b.id));

  let previous: DubbingFullNarrationClip | null = null;
  for (const clip of audible) {
    if (previous && clip.startMs < getFullClipEndMs(previous)) {
      return { previous, clip };
    }
    previous = clip;
  }
  return null;
}

function resolveAudibleClipOverlaps(clips: DubbingFullNarrationClip[]) {
  const ordered = [...clips].sort((a, b) => a.startMs - b.startMs || a.id.localeCompare(b.id));
  let previousAudibleEndMs = 0;
  let audibleIndex = 0;
  const nextById = new Map<string, DubbingFullNarrationClip>();

  ordered.forEach((clip) => {
    const effectiveDurationMs = getFullClipEffectiveDurationMs(clip);
    let startMs = clip.startMs;
    let track = clip.track;
    if (isClipAudible(clip) && effectiveDurationMs > 0) {
      startMs = Math.max(startMs, previousAudibleEndMs);
      previousAudibleEndMs = startMs + effectiveDurationMs;
      track = audibleIndex % 2 === 0 ? 0 : 1;
      audibleIndex += 1;
    }
    nextById.set(clip.id, {
      ...clip,
      startMs,
      track,
    });
  });

  return clips.map((clip) => nextById.get(clip.id) ?? clip);
}

function hasAudibleOverlapWithCandidate(
  clips: DubbingFullNarrationClip[],
  candidate: DubbingFullNarrationClip,
) {
  if (!isClipAudible(candidate) || getFullClipEffectiveDurationMs(candidate) <= 0) return false;
  return (
    findFirstAudibleOverlap([
      ...clips.filter((clip) => clip.id !== candidate.id),
      candidate,
    ]) !== null
  );
}

function findNextNonOverlappingStart(
  clips: DubbingFullNarrationClip[],
  requestedStartMs: number,
  durationMs: number,
) {
  let startMs = Math.max(0, Math.round(requestedStartMs));
  const audible = clips
    .filter((clip) => isClipAudible(clip) && getFullClipEffectiveDurationMs(clip) > 0)
    .sort((a, b) => a.startMs - b.startMs);

  for (const clip of audible) {
    const clipEndMs = getFullClipEndMs(clip);
    const proposedEndMs = startMs + durationMs;
    if (proposedEndMs <= clip.startMs || startMs >= clipEndMs) continue;
    startMs = clipEndMs;
  }
  return startMs;
}

function getDubbingTimelineStorageKey(projectId: string) {
  return `voicebox:dubbing-timeline:${projectId}`;
}

const SELECTED_DUBBING_PROJECT_STORAGE_KEY = 'voicebox:srt2voice:selected-project-id';

export function DubbingTab() {
  const platform = usePlatform();
  const [projects, setProjects] = useState<DubbingProjectListItemResponse[]>([]);
  const [projectSearch, setProjectSearch] = useState('');
  const [isProjectsLoading, setIsProjectsLoading] = useState(false);
  const [project, setProject] = useState<DubbingProjectResponse | null>(null);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(() =>
    window.localStorage.getItem(SELECTED_DUBBING_PROJECT_STORAGE_KEY),
  );
  const [projectsLoadError, setProjectsLoadError] = useState<string | null>(null);
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [timelinePlaybackSource, setTimelinePlaybackSource] = useState<TimelinePlaybackSource>('auto');
  const [fullNarrationClips, setFullNarrationClips] = useState<DubbingFullNarrationClip[]>([]);
  const [segmentClipStarts, setSegmentClipStarts] = useState<Record<string, number>>({});
  const [selectedProfileId, setSelectedProfileId] = useState<string>('');
  const [selectedEngine, setSelectedEngine] = useState<Srt2VoiceEngine>('qwen');
  const [selectedTadaModelSize, setSelectedTadaModelSize] = useState<'1B' | '3B'>('3B');
  const [language, setLanguage] = useState<LanguageCode>(SRT2VOICE_DEFAULT_LANGUAGE);
  const [instruct, setInstruct] = useState('');
  const [isImporting, setIsImporting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isAutoFitting, setIsAutoFitting] = useState(false);
  const [isGeneratingFullNarration, setIsGeneratingFullNarration] = useState(false);
  const [isPostProcessing, setIsPostProcessing] = useState(false);
  const [tempoSuggestion, setTempoSuggestion] = useState<DubbingTempoSuggestionResponse | null>(null);
  const [isSuggestingTempo, setIsSuggestingTempo] = useState(false);
  const [isApplyingTempo, setIsApplyingTempo] = useState(false);
  const [tempoAdjustmentPercent, setTempoAdjustmentPercent] = useState(0);
  const [isRestartingServerForVram, setIsRestartingServerForVram] = useState(false);
  const [, setIsRefreshing] = useState(false);
  const [isCancellingAll, setIsCancellingAll] = useState(false);
  const [segmentActionId, setSegmentActionId] = useState<string | null>(null);
  const [deletingProjectId, setDeletingProjectId] = useState<string | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [renamingProject, setRenamingProject] = useState<DubbingProjectListItemResponse | null>(null);
  const [renameProjectName, setRenameProjectName] = useState('');
  const [isRenamingProject, setIsRenamingProject] = useState(false);
  const [editedSegmentText, setEditedSegmentText] = useState('');
  const [editedSegmentStartTc, setEditedSegmentStartTc] = useState('');
  const [editedSegmentEndTc, setEditedSegmentEndTc] = useState('');
  const [isSavingSegmentText, setIsSavingSegmentText] = useState(false);
  const [isSavingSegmentTiming, setIsSavingSegmentTiming] = useState(false);
  const [projectPaceValue, setProjectPaceValue] = useState<number>(1);
  const [projectTemperatureValue, setProjectTemperatureValue] = useState<number>(QWEN_DEFAULT_TEMPERATURE);
  const [groupPaceValue, setGroupPaceValue] = useState<number>(1);
  const [isSavingProjectPace, setIsSavingProjectPace] = useState(false);
  const [isSavingProjectTemperature, setIsSavingProjectTemperature] = useState(false);
  const [isSavingGroupPace, setIsSavingGroupPace] = useState(false);
  const [serverRestartRefreshNonce, setServerRestartRefreshNonce] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const segmentCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const timelineAudioRef = useRef<HTMLAudioElement | null>(null);
  const timelinePlaybackSegmentRef = useRef<DubbingSegmentResponse | null>(null);
  const timelinePlaybackFullRef = useRef<{
    clipId: string;
    startMs: number;
    generationId: string;
    trimStartMs: number;
    effectiveDurationMs: number;
  } | null>(null);
  const timelineQueueRef = useRef<DubbingSegmentResponse[]>([]);
  const timelineFullClipQueueRef = useRef<DubbingFullNarrationClip[]>([]);
  const timelineGapTimeoutRef = useRef<number | null>(null);
  const timelineGapAnimationRef = useRef<number | null>(null);
  const timelineClipEndTimeoutRef = useRef<number | null>(null);
  const segmentClipStartsRef = useRef<Record<string, number>>({});
  const fullNarrationClipsRef = useRef<DubbingFullNarrationClip[]>([]);
  const lastFullNarrationStatusRef = useRef<{
    projectId: string | null;
    generationId: string | null;
    status: string | null;
  }>({ projectId: null, generationId: null, status: null });
  const restartedFullNarrationKeysRef = useRef<Set<string>>(new Set());
  const [timelinePlaybackSegmentId, setTimelinePlaybackSegmentId] = useState<string | null>(null);
  const [timelinePlaybackTimeMs, setTimelinePlaybackTimeMs] = useState(0);
  const [isTimelinePlaying, setIsTimelinePlaying] = useState(false);
  const [timelineEditorHeight, setTimelineEditorHeight] = useState(232);
  const [segmentLanes, setSegmentLanes] = useState<Record<string, -1 | 0 | 1>>({});
  const [, setSelectedSegmentVolume] = useState(100);
  const [editingSegmentId, setEditingSegmentId] = useState<string | null>(null);
  const { toast } = useToast();
  const { data: profiles } = useProfiles();
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;

  const selectedSegment = useMemo(
    () => project?.segments.find((segment) => segment.id === selectedSegmentId) ?? null,
    [project, selectedSegmentId],
  );
  const editingSegment = useMemo(
    () => project?.segments.find((segment) => segment.id === editingSegmentId) ?? null,
    [project, editingSegmentId],
  );
  const selectedPaceGroup = useMemo(() => {
    if (!project || !selectedSegment?.pace_group_id) return null;
    return project.pace_groups.find((group) => group.id === selectedSegment.pace_group_id) ?? null;
  }, [project, selectedSegment?.pace_group_id]);
  const generatedSegments = useMemo(
    () => project?.segments.filter((segment) => !!segment.generation_id) ?? [],
    [project?.segments],
  );
  const cutSegments = useMemo(
    () => project?.segments.filter((segment) => !!segment.cut_generation_id) ?? [],
    [project?.segments],
  );
  const sortedCutSegments = useMemo(
    () => [...cutSegments].sort((a, b) => a.start_ms - b.start_ms || a.srt_index - b.srt_index),
    [cutSegments],
  );
  const sortedGeneratedSegments = useMemo(
    () => [...generatedSegments].sort((a, b) => a.start_ms - b.start_ms || a.srt_index - b.srt_index),
    [generatedSegments],
  );
  const timelinePlayheadMs = useMemo(() => {
    return timelinePlaybackTimeMs;
  }, [timelinePlaybackTimeMs]);
  const fullNarrationStartMs = useMemo(
    () => Math.min(...(project?.segments.map((segment) => segment.start_ms) ?? [0])),
    [project?.segments],
  );
  const hasFullNarrationAudio =
    !!project?.full_narration_generation_id &&
    project.full_narration_status === 'completed' &&
    !!project.full_narration_duration_ms;
  const hasAutoCutTimeline =
    (project?.post_processed_segment_count ?? 0) > 0 ||
    fullNarrationClips.length > 1 ||
    fullNarrationClips.some((clip) => clip.trimStartMs > 0 || clip.trimEndMs > 0);
  const selectedTempoMultiplier = 1 + tempoAdjustmentPercent / 100;
  const effectiveTimelinePlaybackSource: Exclude<TimelinePlaybackSource, 'auto'> =
    timelinePlaybackSource === 'auto'
      ? hasFullNarrationAudio && fullNarrationClips.length > 0
        ? 'full'
        : 'cuts'
      : timelinePlaybackSource;
  const isFullNarrationActive =
    project?.full_narration_status === 'loading_model' || project?.full_narration_status === 'generating';
  const fullNarrationStatusLabel =
    project?.full_narration_status === 'loading_model'
      ? 'Loading model'
      : project?.full_narration_status === 'generating'
        ? 'Generating full SRT narration'
        : project?.full_narration_status === 'completed'
          ? 'Full SRT narration ready'
          : project?.full_narration_status === 'failed'
            ? 'Full SRT narration failed'
            : null;
  const getSegmentTimelineStartMs = useCallback(
    (segment: DubbingSegmentResponse) =>
      segmentClipStarts[segment.id] ??
      (segment.cut_generation_id && segment.cut_source_start_ms != null
        ? fullNarrationStartMs + segment.cut_source_start_ms
        : segment.start_ms),
    [fullNarrationStartMs, segmentClipStarts],
  );

  const selectAndScrollToSegment = useCallback((segmentId: string) => {
    setSelectedSegmentId(segmentId);
    window.requestAnimationFrame(() => {
      segmentCardRefs.current[segmentId]?.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    });
  }, []);

  const dubbingTimelineClips = useMemo<AudioTrackClip[]>(() => {
    if (!project) return [];
    const clips: AudioTrackClip[] = [];
    for (const segment of project.segments) {
      clips.push({
        id: `reference-${segment.id}`,
        startMs: segment.start_ms,
        durationMs: Math.max(300, segment.end_ms - segment.start_ms),
        track: 2,
        label: `#${segment.srt_index}`,
        sublabel: segment.text,
        variant: 'reference',
        editable: false,
      });
    }

    if (hasFullNarrationAudio) {
      for (const clip of fullNarrationClips) {
        clips.push({
          id: clip.id,
          startMs: clip.startMs,
          durationMs: clip.durationMs,
          track: clip.track,
          label: 'Full SRT narration beta',
          sublabel: 'continuous WAV',
          audioUrl: getFullNarrationAudioUrl(clip),
          trimStartMs: clip.trimStartMs,
          trimEndMs: clip.trimEndMs,
          volume: clip.volume,
          variant: 'info',
          canRegenerate: false,
          movable: true,
          trimmable: true,
        });
      }
    }

    if (effectiveTimelinePlaybackSource !== 'cuts') {
      return clips;
    }

    for (const segment of sortedCutSegments) {
      const generationId = segment.cut_generation_id ?? segment.generation_id;
      if (!generationId) continue;
      clips.push({
        id: segment.id,
        startMs: getSegmentTimelineStartMs(segment),
        durationMs: Math.max(300, segment.cut_duration_ms ?? segment.target_duration_ms),
        track: segment.cut_source_type === 'auto' ? -1 : 0,
        label: segment.text,
        sublabel: `#${segment.srt_index}`,
        audioUrl: apiClient.getAudioUrl(generationId),
        variant: 'success',
        canRegenerate: true,
      });
    }

    if (sortedCutSegments.length === 0) {
      for (const segment of sortedGeneratedSegments) {
        const generationId = segment.generation_id ?? segment.cut_generation_id;
        if (!generationId) continue;
        clips.push({
          id: segment.id,
          startMs: getSegmentTimelineStartMs(segment),
          durationMs: Math.max(500, segment.actual_duration_ms ?? segment.target_duration_ms),
          track: segmentLanes[segment.id] ?? 1,
          label: segment.text,
          sublabel: `#${segment.srt_index}`,
          audioUrl: apiClient.getAudioUrl(generationId),
          variant: segment.fit_status === 'warning' ? 'warning' : 'primary',
          canRegenerate: true,
        });
      }
    }

    return clips;
  }, [
    effectiveTimelinePlaybackSource,
    fullNarrationStartMs,
    fullNarrationClips,
    hasFullNarrationAudio,
    project,
    getSegmentTimelineStartMs,
    segmentLanes,
    sortedCutSegments,
    sortedGeneratedSegments,
    timelinePlaybackSource,
  ]);
  const activeEditableSegment = editingSegment ?? selectedSegment;
  const hasEditedSegmentChanges = activeEditableSegment
    ? editedSegmentText.trim() !== activeEditableSegment.text.trim()
    : false;
  const hasEditedTimingChanges = activeEditableSegment
    ? editedSegmentStartTc.trim() !== activeEditableSegment.start_tc ||
      editedSegmentEndTc.trim() !== activeEditableSegment.end_tc
    : false;

  const filteredProjects = useMemo(() => {
    const q = projectSearch.trim().toLowerCase();
    if (!q) return projects;
    return projects.filter((item) => item.name.toLowerCase().includes(q));
  }, [projects, projectSearch]);

  const dubbingCompatibleProfiles = useMemo(
    () => (profiles ?? []).filter((profile) => isProfileCompatibleWithSrt2VoiceEngine(profile, selectedEngine)),
    [profiles, selectedEngine],
  );
  const selectedProfile = useMemo(
    () => (profiles ?? []).find((profile) => profile.id === selectedProfileId) ?? null,
    [profiles, selectedProfileId],
  );
  const availableEngineOptions = SRT2VOICE_ENGINE_OPTIONS;
  const availableLanguageOptions = useMemo(() => {
    if (selectedEngine === 'tada' && selectedTadaModelSize === '1B') {
      return getLanguageOptionsForEngine('luxtts');
    }
    return getLanguageOptionsForEngine(selectedEngine);
  }, [selectedEngine, selectedTadaModelSize]);
  const selectedEngineValue = selectedEngine === 'tada' ? `tada:${selectedTadaModelSize}` : selectedEngine;
  const selectedModelSize =
    selectedEngine === 'qwen' || selectedEngine === 'qwen_custom_voice' || selectedEngine === 'qwen_voice_design'
      ? '1.7B'
      : selectedEngine === 'tada'
        ? selectedTadaModelSize
        : 'default';
  const isQwenEngine =
    selectedEngine === 'qwen' ||
    selectedEngine === 'qwen_custom_voice' ||
    selectedEngine === 'qwen_voice_design';

  const hasActiveGeneration = useMemo(
    () =>
      isRestartingServerForVram ||
      ((project?.full_narration_status === 'loading_model' ||
        project?.full_narration_status === 'generating' ||
        project?.segments.some((segment) => segment.status === 'generating')) ??
        false),
    [isRestartingServerForVram, project],
  );

  const resetTimelineState = () => {
    const audio = timelineAudioRef.current;
    if (audio) {
      audio.pause();
      audio.removeAttribute('src');
      audio.load();
    }
    if (timelineGapTimeoutRef.current != null) {
      window.clearTimeout(timelineGapTimeoutRef.current);
      timelineGapTimeoutRef.current = null;
    }
    if (timelineGapAnimationRef.current != null) {
      window.cancelAnimationFrame(timelineGapAnimationRef.current);
      timelineGapAnimationRef.current = null;
    }
    if (timelineClipEndTimeoutRef.current != null) {
      window.clearTimeout(timelineClipEndTimeoutRef.current);
      timelineClipEndTimeoutRef.current = null;
    }
    timelinePlaybackFullRef.current = null;
    timelinePlaybackSegmentRef.current = null;
    timelineQueueRef.current = [];
    setTimelinePlaybackSegmentId(null);
    setTimelinePlaybackTimeMs(0);
    setTimelinePlaybackSource('auto');
    setIsTimelinePlaying(false);
    setFullNarrationClips([]);
    setSegmentClipStarts({});
    setSegmentLanes({});
    setSelectedSegmentVolume(100);
    setEditingSegmentId(null);
  };

  const purgeProjectTimelineAudio = (projectId = project?.id) => {
    if (projectId) {
      window.localStorage.removeItem(getDubbingTimelineStorageKey(projectId));
    }
    setFullNarrationClips([]);
    fullNarrationClipsRef.current = [];
    setSegmentClipStarts({});
    setSegmentLanes({});
    setTimelinePlaybackSource('auto');
    handleStopTimelinePlayback();
  };

  const unloadCurrentProjectTimeline = () => {
    resetTimelineState();
    setProject(null);
    setSelectedSegmentId(null);
    setEditedSegmentText('');
    setEditedSegmentStartTc('');
    setEditedSegmentEndTc('');
  };

  const selectDubbingProject = (projectId: string) => {
    if (projectId === selectedProjectId && project?.id === projectId) return;
    unloadCurrentProjectTimeline();
    setSelectedProjectId(projectId);
  };

  useEffect(() => {
    if (selectedProjectId) {
      window.localStorage.setItem(SELECTED_DUBBING_PROJECT_STORAGE_KEY, selectedProjectId);
    } else {
      window.localStorage.removeItem(SELECTED_DUBBING_PROJECT_STORAGE_KEY);
    }
  }, [selectedProjectId]);

  const applyImportedProject = (imported: DubbingProjectResponse) => {
    if (project?.id !== imported.id) {
      resetTimelineState();
    }
    setProject(imported);
    setSelectedProjectId(imported.id);
    setSelectedSegmentId((currentSelected) => {
      if (currentSelected && imported.segments.some((segment) => segment.id === currentSelected)) {
        return currentSelected;
      }
      return imported.segments[0]?.id ?? null;
    });
    setSelectedProfileId(imported.profile_id ?? '');
    setSelectedEngine(isSrt2VoiceEngine(imported.engine) ? imported.engine : 'qwen');
    setLanguage(isLanguageCode(imported.language) ? imported.language : SRT2VOICE_DEFAULT_LANGUAGE);
    setInstruct(imported.style_prompt ?? '');
  };

  const loadProjects = async (preferredProjectId?: string, options?: { silent?: boolean }) => {
    setIsProjectsLoading(true);
    try {
      const items = await apiClient.listDubbingProjects();
      setProjectsLoadError(null);
      setProjects(items);
      const persistedProjectId = window.localStorage.getItem(SELECTED_DUBBING_PROJECT_STORAGE_KEY);
      const nextProjectId = preferredProjectId ?? selectedProjectId ?? persistedProjectId ?? items[0]?.id ?? null;
      if (nextProjectId && items.some((item) => item.id === nextProjectId)) {
        setSelectedProjectId(nextProjectId);
      } else if (!nextProjectId) {
        setSelectedProjectId(null);
        unloadCurrentProjectTimeline();
      }
    } catch (error) {
      setProjectsLoadError(error instanceof Error ? error.message : 'Unknown error');
      if (!options?.silent) {
        toast({
          title: 'Failed to load dubbing projects',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      }
      if (projects.length === 0) {
        throw error;
      }
    } finally {
      setIsProjectsLoading(false);
    }
  };

  const loadProject = async (projectId: string, options?: { silent?: boolean }) => {
    setIsRefreshing(true);
    try {
      const data = await apiClient.getDubbingProject(projectId);
      applyImportedProject(data);
      return data;
    } catch (error) {
      if (!options?.silent) {
        toast({
          title: 'Failed to load project',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      }
      throw error;
    } finally {
      setIsRefreshing(false);
    }
  };

  const waitForServerHealth = useCallback(async () => {
    const deadline = Date.now() + 45_000;
    let lastError: unknown = null;
    while (Date.now() < deadline) {
      try {
        const health = await apiClient.getHealth();
        if (health.status === 'healthy') return;
      } catch (error) {
        lastError = error;
      }
      await delay(750);
    }
    throw lastError instanceof Error ? lastError : new Error('Server did not become ready in time.');
  }, []);

  const reloadProjectAfterServerRestart = async (projectId: string) => {
    let lastError: unknown = null;
    for (let attempt = 0; attempt < 8; attempt += 1) {
      try {
        const loaded = await loadProject(projectId, { silent: true });
        await loadProjects(projectId, { silent: true });
        const fullNarrationStillActive =
          loaded.full_narration_status === 'loading_model' ||
          loaded.full_narration_status === 'generating';
        const completedFullNarrationWithoutAudio =
          loaded.full_narration_status === 'completed' &&
          (!loaded.full_narration_generation_id || !loaded.full_narration_duration_ms);
        if (fullNarrationStillActive || completedFullNarrationWithoutAudio) {
          throw new Error('Project is not fully refreshed after server restart yet.');
        }
        setServerRestartRefreshNonce((value) => value + 1);
        return;
      } catch (error) {
        lastError = error;
        await delay(500 + attempt * 250);
      }
    }
    throw lastError instanceof Error ? lastError : new Error('Project reload failed after server restart.');
  };

  const restartServerForVramRelease = useCallback(
    async (reason: string, projectId?: string | null) => {
      if (!platform.metadata.isTauri || isRestartingServerForVram) return;

      setIsRestartingServerForVram(true);
      try {
        toast({
          title: 'Releasing VRAM',
          description: `Restarting the local server after ${reason}.`,
        });
        await platform.lifecycle.restartServer();
        await waitForServerHealth();
        if (projectId) {
          await reloadProjectAfterServerRestart(projectId);
        }
        toast({
          title: 'VRAM released',
          description: 'The local server has restarted and is ready for the next generation.',
        });
      } catch (error) {
        toast({
          title: 'VRAM release restart failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      } finally {
        setIsRestartingServerForVram(false);
      }
    },
    [
      isRestartingServerForVram,
      platform.lifecycle,
      platform.metadata.isTauri,
      reloadProjectAfterServerRestart,
      toast,
      waitForServerHealth,
    ],
  );

  useEffect(() => {
    const enterSrt2Voice = async () => {
      try {
        await apiClient.releaseDubbingMemory();
      } catch (error) {
        console.debug('SRT2Voice memory release skipped on entry', error);
      }
      await loadProjects();
    };
    void enterSrt2Voice();
  }, []);

  useEffect(() => {
    if (!selectedProjectId) return;
    if (project?.id === selectedProjectId) return;
    void loadProject(selectedProjectId);
  }, [selectedProjectId]);

  useEffect(() => {
    if (!selectedProfileId) return;
    if (selectedProfile && isProfileCompatibleWithSrt2VoiceEngine(selectedProfile, selectedEngine)) return;
    setSelectedProfileId('');
  }, [selectedEngine, selectedProfile, selectedProfileId]);

  useEffect(() => {
    if (availableLanguageOptions.some((option) => option.value === language)) return;
    setLanguage((availableLanguageOptions[0]?.value ?? SRT2VOICE_DEFAULT_LANGUAGE) as LanguageCode);
  }, [availableLanguageOptions, language]);

  useEffect(() => {
    setEditedSegmentText(selectedSegment?.text ?? '');
    setEditedSegmentStartTc(selectedSegment?.start_tc ?? '');
    setEditedSegmentEndTc(selectedSegment?.end_tc ?? '');
  }, [selectedSegment?.id, selectedSegment?.text, selectedSegment?.start_tc, selectedSegment?.end_tc]);

  useEffect(() => {
    setProjectPaceValue(project?.pace_override ?? 1);
  }, [project?.id, project?.pace_override]);

  useEffect(() => {
    setProjectTemperatureValue(project?.temperature ?? QWEN_DEFAULT_TEMPERATURE);
  }, [project?.id, project?.temperature]);

  useEffect(() => {
    setTempoSuggestion(null);
    setTempoAdjustmentPercent(0);
  }, [project?.id, project?.full_narration_revision_ms, project?.full_narration_duration_ms]);

  useEffect(() => {
    segmentClipStartsRef.current = segmentClipStarts;
  }, [segmentClipStarts]);

  useEffect(() => {
    fullNarrationClipsRef.current = fullNarrationClips;
  }, [fullNarrationClips]);

  useEffect(() => {
    setGroupPaceValue(selectedPaceGroup?.pace_override ?? selectedPaceGroup?.effective_pace ?? 1);
  }, [selectedPaceGroup?.id, selectedPaceGroup?.pace_override, selectedPaceGroup?.effective_pace]);

  useEffect(() => {
    const generationId = project?.full_narration_generation_id;
    const durationMs = project?.full_narration_duration_ms;
    const audioRevisionMs = project?.full_narration_revision_ms ?? null;
    if (!hasFullNarrationAudio || !generationId || !durationMs) {
      setFullNarrationClips([]);
      return;
    }

    setFullNarrationClips((current) => {
      const isSameSource =
        current.length > 0 &&
        current.every(
          (clip) =>
            clip.generationId === generationId &&
            clip.audioRevisionMs === audioRevisionMs &&
            clip.durationMs === durationMs,
        );
      if (isSameSource) return current;

      const storedRaw = window.localStorage.getItem(getDubbingTimelineStorageKey(project.id));
      if (storedRaw) {
        try {
          const stored = JSON.parse(storedRaw) as PersistedDubbingTimeline;
          const restoredClips = Array.isArray(stored.clips)
            ? stored.clips.filter(
                (clip) => clip.generationId === generationId && clip.audioRevisionMs === audioRevisionMs,
              )
              .filter(
                (clip) =>
                  typeof clip.durationMs !== 'number' ||
                  Math.abs(clip.durationMs - durationMs) <= 1,
              )
            : [];
          if (
            stored.sourceGenerationId === generationId &&
            stored.sourceRevisionMs === audioRevisionMs &&
            (stored.sourceDurationMs == null || Math.abs(stored.sourceDurationMs - durationMs) <= 1) &&
            restoredClips.length > 0
          ) {
            return resolveAudibleClipOverlaps(restoredClips);
          }
        } catch {
          window.localStorage.removeItem(getDubbingTimelineStorageKey(project.id));
        }
      }

      return [
        {
          id: `${FULL_NARRATION_CLIP_PREFIX}-${audioRevisionMs ?? 'latest'}-0`,
          generationId,
          audioRevisionMs,
          startMs: fullNarrationStartMs,
          durationMs,
          trimStartMs: 0,
          trimEndMs: 0,
          track: 0,
          volume: 1,
        },
      ];
    });
  }, [
    fullNarrationStartMs,
    hasFullNarrationAudio,
    project?.id,
    project?.full_narration_duration_ms,
    project?.full_narration_generation_id,
    project?.full_narration_revision_ms,
    serverRestartRefreshNonce,
  ]);

  useEffect(() => {
    if (!project?.id || !project.full_narration_generation_id || fullNarrationClips.length === 0) return;
    const payload: PersistedDubbingTimeline = {
      sourceGenerationId: project.full_narration_generation_id,
      sourceRevisionMs: project.full_narration_revision_ms ?? null,
      sourceDurationMs: project.full_narration_duration_ms ?? null,
      clips: resolveAudibleClipOverlaps(fullNarrationClips),
    };
    window.localStorage.setItem(getDubbingTimelineStorageKey(project.id), JSON.stringify(payload));
  }, [fullNarrationClips, project?.full_narration_generation_id, project?.full_narration_revision_ms, project?.id]);

  useEffect(() => {
    const audio = new Audio();
    timelineAudioRef.current = audio;

    const clearClipEndTimeout = () => {
      if (timelineClipEndTimeoutRef.current != null) {
        window.clearTimeout(timelineClipEndTimeoutRef.current);
        timelineClipEndTimeoutRef.current = null;
      }
    };

    const advanceFullPlayback = () => {
      clearClipEndTimeout();
      const fullPlayback = timelinePlaybackFullRef.current;
      if (!fullPlayback) return;

      setTimelinePlaybackTimeMs(fullPlayback.startMs + fullPlayback.effectiveDurationMs);
      const queue = timelineFullClipQueueRef.current;
      const currentIndex = queue.findIndex((clip) => clip.id === fullPlayback.clipId);
      const nextClip = currentIndex >= 0 ? queue[currentIndex + 1] : null;
      if (!nextClip) {
        timelinePlaybackFullRef.current = null;
        timelineFullClipQueueRef.current = [];
        setTimelinePlaybackSegmentId(null);
        setIsTimelinePlaying(false);
        return;
      }

      const startNextFullClip = () => {
        const effectiveDurationMs = getFullClipEffectiveDurationMs(nextClip);
        timelinePlaybackFullRef.current = {
          clipId: nextClip.id,
          startMs: nextClip.startMs,
          generationId: nextClip.generationId,
          trimStartMs: nextClip.trimStartMs,
          effectiveDurationMs,
        };
        setSelectedSegmentId(nextClip.id);
        setTimelinePlaybackSegmentId(null);
        setTimelinePlaybackTimeMs(nextClip.startMs);
        audio.src = getFullNarrationAudioUrl(nextClip);
        audio.currentTime = Math.max(0, nextClip.trimStartMs / 1000);
        void audio.play().then(() => {
          clearClipEndTimeout();
          timelineClipEndTimeoutRef.current = window.setTimeout(() => {
            const active = timelinePlaybackFullRef.current;
            if (active?.clipId !== nextClip.id) return;
            audio.pause();
            advanceFullPlayback();
          }, Math.max(1, effectiveDurationMs));
        }).catch(() => setIsTimelinePlaying(false));
      };

      const gapMs = Math.max(0, nextClip.startMs - (fullPlayback.startMs + fullPlayback.effectiveDurationMs));
      if (gapMs > 0) {
        setIsTimelinePlaying(true);
        const gapStartedAt = performance.now();
        const gapStartMs = fullPlayback.startMs + fullPlayback.effectiveDurationMs;
        const animateGap = (now: number) => {
          const progress = Math.min(1, (now - gapStartedAt) / gapMs);
          setTimelinePlaybackTimeMs(Math.round(gapStartMs + (nextClip.startMs - gapStartMs) * progress));
          if (progress < 1) {
            timelineGapAnimationRef.current = window.requestAnimationFrame(animateGap);
          }
        };
        timelineGapAnimationRef.current = window.requestAnimationFrame(animateGap);
        timelineGapTimeoutRef.current = window.setTimeout(startNextFullClip, gapMs);
        return;
      }
      startNextFullClip();
    };

    const handleTimeUpdate = () => {
      const fullPlayback = timelinePlaybackFullRef.current;
      if (fullPlayback) {
        const clipElapsedMs = Math.max(0, Math.round(audio.currentTime * 1000) - fullPlayback.trimStartMs);
        if (clipElapsedMs >= fullPlayback.effectiveDurationMs) {
          clearClipEndTimeout();
          audio.pause();
          advanceFullPlayback();
          return;
        }
        setTimelinePlaybackTimeMs(fullPlayback.startMs + clipElapsedMs);
        return;
      }
      const segment = timelinePlaybackSegmentRef.current;
      if (!segment) return;
      const segmentStartMs = segmentClipStartsRef.current[segment.id] ?? segment.start_ms;
      setTimelinePlaybackTimeMs(segmentStartMs + Math.round(audio.currentTime * 1000));
    };

    const handleEnded = () => {
      const fullPlayback = timelinePlaybackFullRef.current;
      if (fullPlayback) {
        clearClipEndTimeout();
        advanceFullPlayback();
        return;
      }
      const segment = timelinePlaybackSegmentRef.current;
      if (!segment) {
        setIsTimelinePlaying(false);
        return;
      }

      const actualDurationMs = segment.cut_duration_ms ?? segment.actual_duration_ms ?? segment.target_duration_ms;
      const segmentStartMs = segmentClipStartsRef.current[segment.id] ?? segment.start_ms;
      const segmentEndMs = segmentStartMs + actualDurationMs;
      setTimelinePlaybackTimeMs(segmentEndMs);

      const queue = timelineQueueRef.current;
      const currentIndex = queue.findIndex((item) => item.id === segment.id);
      const nextSegment = currentIndex >= 0 ? queue[currentIndex + 1] : null;
      const nextGenerationId = nextSegment?.cut_generation_id ?? nextSegment?.generation_id;
      if (!nextSegment || !nextGenerationId) {
        setIsTimelinePlaying(false);
        return;
      }

      const startNextSegment = () => {
        timelinePlaybackSegmentRef.current = nextSegment;
        const nextSegmentStartMs = segmentClipStartsRef.current[nextSegment.id] ?? nextSegment.start_ms;
        setSelectedSegmentId(nextSegment.id);
        setTimelinePlaybackSegmentId(nextSegment.id);
        setTimelinePlaybackTimeMs(nextSegmentStartMs);
        audio.src = apiClient.getAudioUrl(nextGenerationId);
        audio.currentTime = 0;
        void audio.play().catch(() => setIsTimelinePlaying(false));
      };

      const nextSegmentStartMs = segmentClipStartsRef.current[nextSegment.id] ?? nextSegment.start_ms;
      const gapMs = Math.max(0, nextSegmentStartMs - segmentEndMs);
      if (gapMs > 0) {
        const gapStartedAt = performance.now();
        const animateGap = (now: number) => {
          const progress = Math.min(1, (now - gapStartedAt) / gapMs);
          setTimelinePlaybackTimeMs(Math.round(segmentEndMs + (nextSegmentStartMs - segmentEndMs) * progress));
          if (progress < 1) {
            timelineGapAnimationRef.current = window.requestAnimationFrame(animateGap);
          }
        };
        timelineGapAnimationRef.current = window.requestAnimationFrame(animateGap);
        timelineGapTimeoutRef.current = window.setTimeout(startNextSegment, gapMs);
        return;
      }
      startNextSegment();
    };

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('pause', () => setIsTimelinePlaying(false));
    audio.addEventListener('play', () => setIsTimelinePlaying(true));

    return () => {
      if (timelineGapTimeoutRef.current != null) {
        window.clearTimeout(timelineGapTimeoutRef.current);
        timelineGapTimeoutRef.current = null;
      }
      if (timelineGapAnimationRef.current != null) {
        window.cancelAnimationFrame(timelineGapAnimationRef.current);
        timelineGapAnimationRef.current = null;
      }
      clearClipEndTimeout();
      audio.pause();
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
      timelineAudioRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!project || !hasActiveGeneration || isRestartingServerForVram) return;
    const interval = window.setInterval(() => {
      void loadProject(project.id);
      void loadProjects(project.id);
    }, 2500);
    return () => window.clearInterval(interval);
  }, [project, hasActiveGeneration, isRestartingServerForVram]);

  useEffect(() => {
    if (!project) return;

    const status = project.full_narration_status ?? null;
    const generationId = project.full_narration_generation_id ?? null;
    const previous = lastFullNarrationStatusRef.current;
    const wasActive =
      previous.projectId === project.id &&
      previous.generationId === generationId &&
      (previous.status === 'loading_model' || previous.status === 'generating');
    const isTerminal = status === 'completed' || status === 'failed';

    lastFullNarrationStatusRef.current = {
      projectId: project.id,
      generationId,
      status,
    };

    if (!generationId || !wasActive || !isTerminal) return;

    const restartKey = `${project.id}:${generationId}:${project.full_narration_revision_ms ?? status}`;
    if (restartedFullNarrationKeysRef.current.has(restartKey)) return;
    restartedFullNarrationKeysRef.current.add(restartKey);

    if (AUTO_RESTART_SERVER_FOR_VRAM_RELEASE) {
      void restartServerForVramRelease('full SRT narration', project.id);
    }
  }, [
    project?.full_narration_generation_id,
    project?.full_narration_revision_ms,
    project?.full_narration_status,
    project?.id,
    restartServerForVramRelease,
  ]);

  const handlePickFile = () => {
    unloadCurrentProjectTimeline();
    inputRef.current?.click();
  };

  const withSegmentAction = async (segmentId: string, action: () => Promise<void>) => {
    setSegmentActionId(segmentId);
    try {
      await action();
    } finally {
      setSegmentActionId((current) => (current === segmentId ? null : current));
    }
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsImporting(true);
    try {
      const imported = await apiClient.importDubbingSrt(file);
      applyImportedProject(imported);
      await loadProjects(imported.id);
      toast({
        title: 'SRT2Voice project created',
        description: `${imported.segments.length} segments imported from ${file.name}.`,
      });
    } catch (error) {
      toast({
        title: 'SRT import failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsImporting(false);
      if (inputRef.current) inputRef.current.value = '';
    }
  };

  const ensureVoiceSelected = () => {
    if (selectedProfileId) return true;
        toast({
          title: 'Voice required',
          description: 'Select a Qwen cloned, CustomVoice, or VoiceDesign profile before generating.',
          variant: 'destructive',
        });
    return false;
  };

  const refreshProject = async () => {
    if (!project) return;
    await loadProject(project.id);
    await loadProjects(project.id);
  };

  const handleDeleteProject = async (projectId: string) => {
    setDeletingProjectId(projectId);
    try {
      await apiClient.deleteDubbingProject(projectId);
      const remainingProjects = projects.filter((item) => item.id !== projectId);
      const nextProjectId =
        selectedProjectId === projectId ? (remainingProjects[0]?.id ?? null) : selectedProjectId;
      setProjects(remainingProjects);
      setSelectedProjectId(nextProjectId);
      if (selectedProjectId === projectId) {
        unloadCurrentProjectTimeline();
      }
      if (nextProjectId) {
        await loadProject(nextProjectId);
      }
      toast({
        title: 'Project deleted',
        description: 'The SRT2Voice project was removed.',
      });
    } catch (error) {
      toast({
        title: 'Delete project failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setDeletingProjectId(null);
    }
  };

  const handleRenameProject = async (item: DubbingProjectListItemResponse) => {
    setRenamingProject(item);
    setRenameProjectName(item.name);
    setRenameDialogOpen(true);
  };

  const handleSaveProjectRename = async () => {
    if (!renamingProject) return;
    const nextName = renameProjectName.trim();
    if (!nextName) {
      toast({
        title: 'Name required',
        description: 'Enter a project name before saving.',
        variant: 'destructive',
      });
      return;
    }
    if (nextName === renamingProject.name) {
      setRenameDialogOpen(false);
      setRenamingProject(null);
      return;
    }

    setIsRenamingProject(true);
    try {
      const updated = await apiClient.updateDubbingProjectSettings(renamingProject.id, { name: nextName });
      setProjects((current) =>
        current.map((candidate) => (candidate.id === renamingProject.id ? { ...candidate, name: updated.name } : candidate)),
      );
      if (project?.id === renamingProject.id) {
        applyImportedProject(updated);
      }
      await loadProjects(renamingProject.id);
      setRenameDialogOpen(false);
      setRenamingProject(null);
      toast({
        title: 'Project renamed',
        description: `Project is now "${nextName}".`,
      });
    } catch (error) {
      toast({
        title: 'Rename failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsRenamingProject(false);
    }
  };

  const handleSaveSegmentText = async () => {
    const targetSegment = editingSegment ?? selectedSegment;
    if (!project || !targetSegment) return;

    const nextText = editedSegmentText.trim();
    if (!nextText) {
      toast({
        title: 'Text required',
        description: 'Segment text cannot be empty.',
        variant: 'destructive',
      });
      return;
    }

    setIsSavingSegmentText(true);
    try {
      const updatedSegment = await apiClient.updateDubbingSegment(project.id, targetSegment.id, {
        text: nextText,
      });
      purgeProjectTimelineAudio(project.id);
      setProject((current) =>
        current
          ? {
              ...current,
              segments: current.segments.map((segment) =>
                segment.id === updatedSegment.id ? updatedSegment : segment,
              ),
            }
          : current,
      );
      setEditedSegmentText(updatedSegment.text);
      setEditedSegmentStartTc(updatedSegment.start_tc);
      setEditedSegmentEndTc(updatedSegment.end_tc);
      setSelectedSegmentId(updatedSegment.id);
      setEditingSegmentId(updatedSegment.id);
      await loadProjects(project.id);
      toast({
        title: 'Segment updated',
        description: `Segment #${updatedSegment.srt_index} text saved. Existing audio was reset.`,
      });
    } catch (error) {
      toast({
        title: 'Save segment failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsSavingSegmentText(false);
    }
  };

  const handleUpdateSegmentTiming = async (
    segmentId: string,
    startMs: number,
    endMs: number,
    preserveAudio = false,
  ) => {
    if (!project) return;
    setIsSavingSegmentTiming(true);
    try {
      const updatedSegment = await apiClient.updateDubbingSegmentTiming(project.id, segmentId, {
        start_ms: startMs,
        end_ms: endMs,
        preserve_audio: preserveAudio,
      });
      if (!preserveAudio) {
        purgeProjectTimelineAudio(project.id);
      }
      setProject((current) =>
        current
          ? {
              ...current,
              segments: current.segments.map((segment) =>
                segment.id === updatedSegment.id ? updatedSegment : segment,
              ),
            }
          : current,
      );
      setEditedSegmentText(updatedSegment.text);
      setEditedSegmentStartTc(updatedSegment.start_tc);
      setEditedSegmentEndTc(updatedSegment.end_tc);
      setSelectedSegmentId(updatedSegment.id);
      setEditingSegmentId(updatedSegment.id);
      await loadProjects(project.id);
    } catch (error) {
      await refreshProject();
      toast({
        title: 'Timeline update failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsSavingSegmentTiming(false);
    }
  };

  const handleSaveSegmentTimingFields = async () => {
    const targetSegment = editingSegment ?? selectedSegment;
    if (!targetSegment) return;
    const startMs = parseSrtTimecode(editedSegmentStartTc);
    const endMs = parseSrtTimecode(editedSegmentEndTc);
    if (startMs == null || endMs == null) {
      toast({
        title: 'Invalid timecode',
        description: 'Use SRT format HH:MM:SS,mmm, for example 00:00:06,600.',
        variant: 'destructive',
      });
      return;
    }
    if (endMs <= startMs) {
      toast({
        title: 'Invalid time window',
        description: 'The segment end time must be after the start time.',
        variant: 'destructive',
      });
      return;
    }
    await handleUpdateSegmentTiming(targetSegment.id, startMs, endMs);
    setEditedSegmentStartTc(formatSrtTimecode(startMs));
    setEditedSegmentEndTc(formatSrtTimecode(endMs));
    toast({
      title: 'Timecode updated',
      description: `Segment #${targetSegment.srt_index} timing saved. Re-run post-process cuts if needed.`,
    });
  };

  const handleSaveProjectPace = async () => {
    if (!project) return;
    setIsSavingProjectPace(true);
    try {
      const updated = await apiClient.updateDubbingProjectSettings(project.id, {
        pace_override: Math.round(projectPaceValue * 100) / 100,
      });
      applyImportedProject(updated);
      await loadProjects(updated.id);
      toast({
        title: 'Project pace saved',
        description: `Project-level SRT2Voice pace set to ${projectPaceValue.toFixed(2)}x.`,
      });
    } catch (error) {
      toast({
        title: 'Project pace update failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsSavingProjectPace(false);
    }
  };

  const handleResetProjectPace = async () => {
    if (!project) return;
    setIsSavingProjectPace(true);
    try {
      const updated = await apiClient.updateDubbingProjectSettings(project.id, {
        pace_override: null,
      });
      applyImportedProject(updated);
      await loadProjects(updated.id);
      toast({
        title: 'Project pace reset',
        description: 'Automatic group pace is active again at project level.',
      });
    } catch (error) {
      toast({
        title: 'Project pace reset failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsSavingProjectPace(false);
    }
  };

  const handleSaveProjectTemperature = async () => {
    if (!project) return;
    setIsSavingProjectTemperature(true);
    try {
      const updated = await apiClient.updateDubbingProjectSettings(project.id, {
        temperature: Math.round(projectTemperatureValue * 100) / 100,
      });
      applyImportedProject(updated);
      await loadProjects(updated.id);
      toast({
        title: 'Project temperature saved',
        description: `Qwen sampling temperature set to ${projectTemperatureValue.toFixed(2)}.`,
      });
    } catch (error) {
      toast({
        title: 'Project temperature update failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsSavingProjectTemperature(false);
    }
  };

  const handleResetProjectTemperature = async () => {
    if (!project) return;
    setIsSavingProjectTemperature(true);
    try {
      const updated = await apiClient.updateDubbingProjectSettings(project.id, {
        temperature: null,
      });
      applyImportedProject(updated);
      await loadProjects(updated.id);
      toast({
        title: 'Project temperature reset',
        description: 'Qwen default sampling temperature is active again.',
      });
    } catch (error) {
      toast({
        title: 'Project temperature reset failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsSavingProjectTemperature(false);
    }
  };

  const handleSaveGroupPace = async () => {
    if (!project || !selectedPaceGroup) return;
    setIsSavingGroupPace(true);
    try {
      const updated = await apiClient.updateDubbingGroupPace(project.id, selectedPaceGroup.id, {
        pace_override: Math.round(groupPaceValue * 100) / 100,
      });
      applyImportedProject(updated);
      await loadProjects(updated.id);
      toast({
        title: 'Phrase pace saved',
        description: `${selectedPaceGroup.label} pace set to ${groupPaceValue.toFixed(2)}x.`,
      });
    } catch (error) {
      toast({
        title: 'Phrase pace update failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsSavingGroupPace(false);
    }
  };

  const handleResetGroupPace = async () => {
    if (!project || !selectedPaceGroup) return;
    setIsSavingGroupPace(true);
    try {
      const updated = await apiClient.updateDubbingGroupPace(project.id, selectedPaceGroup.id, {
        pace_override: null,
      });
      applyImportedProject(updated);
      await loadProjects(updated.id);
      toast({
        title: 'Phrase pace reset',
        description: `${selectedPaceGroup.label} now uses automatic group pacing again.`,
      });
    } catch (error) {
      toast({
        title: 'Phrase pace reset failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsSavingGroupPace(false);
    }
  };

  const handleGenerateSegment = async (segment = selectedSegment) => {
    if (!project || !segment || !ensureVoiceSelected()) return;
    const deliveryInstructions = isQwenEngine ? instruct.trim() : '';
    const temperature =
      isQwenEngine && project.temperature != null ? Math.round(projectTemperatureValue * 100) / 100 : undefined;

    setIsGenerating(true);
    await withSegmentAction(segment.id, async () => {
      try {
        await apiClient.generateDubbingSegment(project.id, segment.id, {
          profile_id: selectedProfileId,
          language,
          engine: selectedEngine,
          model_size: selectedModelSize,
          instruct: deliveryInstructions || undefined,
          style_prompt: deliveryInstructions || undefined,
          temperature,
        });
        await refreshProject();
        toast({
          title: 'Segment queued',
          description: `Segment #${segment.srt_index} is generating with Qwen.`,
        });
      } catch (error) {
        toast({
          title: 'Generation failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      } finally {
        setIsGenerating(false);
      }
    });
  };

  const handleAutoFitSegment = async (segment = selectedSegment) => {
    if (!project || !segment || !ensureVoiceSelected()) return;
    const deliveryInstructions = isQwenEngine ? instruct.trim() : '';
    const temperature =
      isQwenEngine && project.temperature != null ? Math.round(projectTemperatureValue * 100) / 100 : undefined;

    setIsAutoFitting(true);
    await withSegmentAction(segment.id, async () => {
      try {
        await apiClient.autoFitDubbingSegment(project.id, segment.id, {
          profile_id: selectedProfileId,
          language,
          engine: selectedEngine,
          model_size: selectedModelSize,
          instruct: deliveryInstructions || undefined,
          style_prompt: deliveryInstructions || undefined,
          temperature,
          max_attempts: 1,
        });
        await refreshProject();
        toast({
          title: 'Segment queued',
          description: `Segment #${segment.srt_index} is generating once with natural delivery.`,
        });
      } catch (error) {
        toast({
          title: 'Generation failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      } finally {
        setIsAutoFitting(false);
      }
    });
  };

  const handleGenerateFullNarration = async () => {
    if (!project || !ensureVoiceSelected()) return;
    const deliveryInstructions = isQwenEngine ? instruct.trim() : '';
    const temperature =
      isQwenEngine && project.temperature != null ? Math.round(projectTemperatureValue * 100) / 100 : undefined;

    setIsGeneratingFullNarration(true);
    try {
      purgeProjectTimelineAudio(project.id);
      const queued = await apiClient.generateDubbingFullNarration(project.id, {
        profile_id: selectedProfileId,
        language,
        engine: selectedEngine,
        model_size: selectedModelSize,
        instruct: deliveryInstructions || undefined,
        style_prompt: deliveryInstructions || undefined,
        temperature,
      });
      applyImportedProject(queued);
      await loadProjects(queued.id);
      toast({
        title: 'Full SRT narration started',
        description: 'The cleaned SRT text is being generated as one continuous narration.',
      });
    } catch (error) {
      toast({
        title: 'Full narration failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsGeneratingFullNarration(false);
    }
  };

  const handleRetryFailedSegment = async (segment: DubbingSegmentResponse) => {
    setSelectedSegmentId(segment.id);
    await handleAutoFitSegment(segment);
  };

  const playTimelineFromSegment = (segment: DubbingSegmentResponse, offsetMs = 0) => {
    const audioGenerationId = segment.cut_generation_id ?? segment.generation_id;
    const queue = sortedCutSegments.length > 0 ? sortedCutSegments : sortedGeneratedSegments;
    if (!audioGenerationId) {
      toast({
        title: 'No audio yet',
        description: 'Generate this segment first to listen to it.',
        variant: 'destructive',
      });
      return;
    }

    const audio = timelineAudioRef.current;
    if (!audio) return;

    if (timelineGapTimeoutRef.current != null) {
      window.clearTimeout(timelineGapTimeoutRef.current);
      timelineGapTimeoutRef.current = null;
    }
    if (timelineGapAnimationRef.current != null) {
      window.cancelAnimationFrame(timelineGapAnimationRef.current);
      timelineGapAnimationRef.current = null;
    }
    if (timelineClipEndTimeoutRef.current != null) {
      window.clearTimeout(timelineClipEndTimeoutRef.current);
      timelineClipEndTimeoutRef.current = null;
    }

    timelinePlaybackFullRef.current = null;
    timelinePlaybackSegmentRef.current = segment;
    const segmentStartMs = getSegmentTimelineStartMs(segment);
    timelineQueueRef.current = queue.filter((item) => getSegmentTimelineStartMs(item) >= segmentStartMs);
    setSelectedSegmentId(segment.id);
    setTimelinePlaybackSegmentId(segment.id);
    setTimelinePlaybackTimeMs(segmentStartMs + offsetMs);
    audio.src = apiClient.getAudioUrl(audioGenerationId);
    audio.currentTime = Math.max(0, offsetMs / 1000);
    void audio.play().catch((error) => {
      toast({
        title: 'Timeline playback failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    });
  };

  const handlePlaySegment = (segment: DubbingSegmentResponse) => {
    playTimelineFromSegment(segment);
  };

  const playFullNarrationClip = (clip: DubbingFullNarrationClip, offsetMs = 0) => {
    const audio = timelineAudioRef.current;
    if (!audio) return;
    const effectiveDurationMs = getFullClipEffectiveDurationMs(clip);
    const safeOffsetMs = Math.max(0, Math.min(offsetMs, Math.max(0, effectiveDurationMs - 1)));

    if (timelineGapTimeoutRef.current != null) {
      window.clearTimeout(timelineGapTimeoutRef.current);
      timelineGapTimeoutRef.current = null;
    }
    if (timelineGapAnimationRef.current != null) {
      window.cancelAnimationFrame(timelineGapAnimationRef.current);
      timelineGapAnimationRef.current = null;
    }
    if (timelineClipEndTimeoutRef.current != null) {
      window.clearTimeout(timelineClipEndTimeoutRef.current);
      timelineClipEndTimeoutRef.current = null;
    }

    timelinePlaybackSegmentRef.current = null;
    timelineQueueRef.current = [];
    timelineFullClipQueueRef.current = resolveAudibleClipOverlaps(fullNarrationClipsRef.current)
      .filter((candidate) => isClipAudible(candidate) && getFullClipEffectiveDurationMs(candidate) > 0)
      .sort((a, b) => a.startMs - b.startMs)
      .filter((candidate) => candidate.startMs >= clip.startMs);
    timelinePlaybackFullRef.current = {
      clipId: clip.id,
      startMs: clip.startMs,
      generationId: clip.generationId,
      trimStartMs: clip.trimStartMs,
      effectiveDurationMs,
    };
    setTimelinePlaybackSegmentId(null);
    setSelectedSegmentId(clip.id);
    setTimelinePlaybackTimeMs(clip.startMs + safeOffsetMs);
    audio.src = getFullNarrationAudioUrl(clip);
    audio.currentTime = Math.max(0, (clip.trimStartMs + safeOffsetMs) / 1000);
    void audio.play().then(() => {
      if (timelineClipEndTimeoutRef.current != null) {
        window.clearTimeout(timelineClipEndTimeoutRef.current);
      }
      timelineClipEndTimeoutRef.current = window.setTimeout(() => {
        const active = timelinePlaybackFullRef.current;
        if (active?.clipId !== clip.id) return;
        audio.pause();
        audio.dispatchEvent(new Event('ended'));
      }, Math.max(1, effectiveDurationMs - safeOffsetMs));
    }).catch((error) => {
      toast({
        title: 'Timeline playback failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    });
  };

  const findFullNarrationClipAtTime = (targetMs: number) => {
    const playableClips = resolveAudibleClipOverlaps(fullNarrationClipsRef.current)
      .filter((clip) => isClipAudible(clip) && getFullClipEffectiveDurationMs(clip) > 0)
      .sort((a, b) => a.startMs - b.startMs);
    return (
      playableClips.find((clip) => {
        const effectiveDurationMs = getFullClipEffectiveDurationMs(clip);
        return targetMs >= clip.startMs && targetMs <= clip.startMs + effectiveDurationMs;
      }) ??
      playableClips.find((clip) => clip.startMs >= targetMs) ??
      playableClips[0] ??
      null
    );
  };

  const handlePlayTimeline = () => {
    const audio = timelineAudioRef.current;
    if (!audio) return;

    if (isTimelinePlaying) {
      const fullPlayback = timelinePlaybackFullRef.current;
      if (fullPlayback) {
        const clipElapsedMs = Math.max(0, Math.round(audio.currentTime * 1000) - fullPlayback.trimStartMs);
        setTimelinePlaybackTimeMs(
          fullPlayback.startMs + Math.min(clipElapsedMs, fullPlayback.effectiveDurationMs),
        );
      } else {
        const segment = timelinePlaybackSegmentRef.current;
        if (segment) {
          const segmentStartMs = segmentClipStartsRef.current[segment.id] ?? segment.start_ms;
          setTimelinePlaybackTimeMs(segmentStartMs + Math.round(audio.currentTime * 1000));
        }
      }
      if (timelineGapTimeoutRef.current != null) {
        window.clearTimeout(timelineGapTimeoutRef.current);
        timelineGapTimeoutRef.current = null;
      }
      if (timelineGapAnimationRef.current != null) {
        window.cancelAnimationFrame(timelineGapAnimationRef.current);
        timelineGapAnimationRef.current = null;
      }
      if (timelineClipEndTimeoutRef.current != null) {
        window.clearTimeout(timelineClipEndTimeoutRef.current);
        timelineClipEndTimeoutRef.current = null;
      }
      timelinePlaybackFullRef.current = null;
      timelineFullClipQueueRef.current = [];
      timelinePlaybackSegmentRef.current = null;
      timelineQueueRef.current = [];
      audio.pause();
      setIsTimelinePlaying(false);
      return;
    }

    if (hasFullNarrationAudio && effectiveTimelinePlaybackSource === 'full') {
      const clip = findFullNarrationClipAtTime(timelinePlayheadMs);
      if (!clip) {
        toast({
          title: 'No full WAV clip',
          description: 'Generate the full SRT narration before playing this timeline.',
          variant: 'destructive',
        });
        return;
      }
      playFullNarrationClip(clip, Math.max(0, timelinePlayheadMs - clip.startMs));
      return;
    }

    const segmentSource = sortedCutSegments.length > 0 ? sortedCutSegments : sortedGeneratedSegments;
    const selectedGeneratedSegment =
      selectedSegment && (selectedSegment.cut_generation_id || selectedSegment.generation_id)
        ? segmentSource.find((segment) => segment.id === selectedSegment.id)
        : null;
    const fallbackSegment =
      selectedGeneratedSegment ??
      segmentSource.find((segment) => getSegmentTimelineStartMs(segment) >= timelinePlayheadMs) ??
      segmentSource[0];
    if (!fallbackSegment) {
      toast({
        title: 'No generated audio yet',
        description:
          effectiveTimelinePlaybackSource === 'cuts'
            ? 'Generate or post-process cuts before playing the cuts timeline.'
            : 'Generate at least one segment before playing the timeline.',
        variant: 'destructive',
      });
      return;
    }

    const offsetMs =
      timelinePlaybackSegmentId === fallbackSegment.id
        ? Math.max(0, timelinePlayheadMs - getSegmentTimelineStartMs(fallbackSegment))
        : 0;
    playTimelineFromSegment(fallbackSegment, offsetMs);
  };

  const handleStopTimelinePlayback = () => {
    const audio = timelineAudioRef.current;
    if (!audio) return;
    if (timelineGapTimeoutRef.current != null) {
      window.clearTimeout(timelineGapTimeoutRef.current);
      timelineGapTimeoutRef.current = null;
    }
    if (timelineGapAnimationRef.current != null) {
      window.cancelAnimationFrame(timelineGapAnimationRef.current);
      timelineGapAnimationRef.current = null;
    }
    if (timelineClipEndTimeoutRef.current != null) {
      window.clearTimeout(timelineClipEndTimeoutRef.current);
      timelineClipEndTimeoutRef.current = null;
    }
    audio.pause();
    audio.currentTime = 0;
    if (timelinePlaybackFullRef.current) {
      setTimelinePlaybackTimeMs(timelinePlaybackFullRef.current.startMs);
      timelinePlaybackFullRef.current = null;
      setTimelinePlaybackSegmentId(null);
      return;
    }
    const segment = timelinePlaybackSegmentRef.current;
    if (segment) {
      setTimelinePlaybackTimeMs(segment.start_ms);
    }
  };

  const handleTimelineSeek = (targetMs: number, shouldPlay = isTimelinePlaying) => {
    const shouldUseFullPlayback =
      hasFullNarrationAudio &&
      !!project?.full_narration_duration_ms &&
      effectiveTimelinePlaybackSource === 'full';
    if (shouldUseFullPlayback && project?.full_narration_duration_ms) {
      const clip = findFullNarrationClipAtTime(targetMs);
      if (clip) {
        setTimelinePlaybackTimeMs(targetMs);
        if (shouldPlay) {
          playFullNarrationClip(clip, Math.max(0, targetMs - clip.startMs));
        }
        return;
      }
    }

    const playableSegments = sortedCutSegments.length > 0 ? sortedCutSegments : sortedGeneratedSegments;
    const matchingGeneratedSegment = playableSegments.find((segment) => {
      const durationMs = segment.cut_duration_ms ?? segment.actual_duration_ms ?? segment.target_duration_ms;
      const segmentStartMs = getSegmentTimelineStartMs(segment);
      return targetMs >= segmentStartMs && targetMs <= segmentStartMs + durationMs;
    });

    if (!matchingGeneratedSegment) {
      const matchingSrtSegment = project?.segments.find(
        (segment) => targetMs >= segment.start_ms && targetMs <= segment.end_ms,
      );
      if (matchingSrtSegment) {
        setSelectedSegmentId(matchingSrtSegment.id);
      }
      setTimelinePlaybackTimeMs(targetMs);
      return;
    }

    setSelectedSegmentId(matchingGeneratedSegment.id);
    setTimelinePlaybackSegmentId(matchingGeneratedSegment.id);
    setTimelinePlaybackTimeMs(targetMs);
    if (shouldPlay) {
      playTimelineFromSegment(matchingGeneratedSegment, targetMs - getSegmentTimelineStartMs(matchingGeneratedSegment));
    }
  };

  const splitFullNarrationClip = (clipId?: string, splitTimeMs?: number) => {
    const clip =
      (clipId ? fullNarrationClips.find((candidate) => candidate.id === clipId) : null) ??
      fullNarrationClips.find((candidate) => {
        const effectiveDurationMs = candidate.durationMs - candidate.trimStartMs - candidate.trimEndMs;
        return timelinePlayheadMs > candidate.startMs && timelinePlayheadMs < candidate.startMs + effectiveDurationMs;
      });

    if (!clip) {
      toast({
        title: 'No full WAV clip selected',
        description: 'Select the full WAV clip or place the playhead inside it before cutting.',
        variant: 'destructive',
      });
      return;
    }

    const effectiveDurationMs = clip.durationMs - clip.trimStartMs - clip.trimEndMs;
    const rawSplitOffsetMs = splitTimeMs ?? timelinePlayheadMs - clip.startMs;
    const splitOffsetMs = Math.round(rawSplitOffsetMs);
    if (splitOffsetMs <= 50 || splitOffsetMs >= effectiveDurationMs - 50) {
      toast({
        title: 'Invalid split point',
        description: 'Place the playhead inside the full WAV clip, away from its edges.',
        variant: 'destructive',
      });
      return;
    }

    const now = Date.now();
    const leftClip: DubbingFullNarrationClip = {
      ...clip,
      id: `${clip.id}-left-${now}`,
      trimEndMs: clip.trimEndMs + (effectiveDurationMs - splitOffsetMs),
      track: 0,
    };
    const rightClip: DubbingFullNarrationClip = {
      ...clip,
      id: `${clip.id}-right-${now}`,
      startMs: clip.startMs + splitOffsetMs,
      trimStartMs: clip.trimStartMs + splitOffsetMs,
      track: 1,
    };

    setFullNarrationClips((current) =>
      resolveAudibleClipOverlaps(
        current.flatMap((candidate) => (candidate.id === clip.id ? [leftClip, rightClip] : [candidate])),
      ),
    );
    setSelectedSegmentId(rightClip.id);
    setTimelinePlaybackSource('full');
  };

  const handleTimelineCut = async (segmentId?: string) => {
    if (!project) return;
    if (
      isFullNarrationClipId(segmentId) ||
      segmentId === 'full-narration' ||
      (!segmentId && effectiveTimelinePlaybackSource === 'full')
    ) {
      splitFullNarrationClip(isFullNarrationClipId(segmentId) ? segmentId : undefined);
      return;
    }

    toast({
      title: 'Use the full WAV clip for manual cuts',
      description: 'Dubbing cuts now behave like Stories: select the full WAV clip and split it in place.',
    });
  };

  const handleTimelineVolumeChange = (value: number) => {
    setSelectedSegmentVolume(value);
    toast({
      title: 'Volume preview only',
      description: 'Per-segment SRT2Voice volume is not persisted yet.',
    });
  };

  const handleDownloadSegmentAudio = async (segment: DubbingSegmentResponse) => {
    const generationId = segment.generation_id;
    if (!generationId) return;
    await withSegmentAction(segment.id, async () => {
      try {
        const blob = await apiClient.exportGenerationAudio(generationId);
        await saveBlob(blob, `segment-${segment.srt_index}.wav`, platform.filesystem.saveFile);
      } catch (error) {
        toast({
          title: 'Export audio failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      }
    });
  };

  const handleExportSegmentPackage = async (segment: DubbingSegmentResponse) => {
    const generationId = segment.generation_id;
    if (!generationId) return;
    await withSegmentAction(segment.id, async () => {
      try {
        const blob = await apiClient.exportGeneration(generationId);
        await saveBlob(blob, `segment-${segment.srt_index}.voicebox.zip`, platform.filesystem.saveFile);
      } catch (error) {
        toast({
          title: 'Export package failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      }
    });
  };

  const handleRegenerateSegment = async (segment: DubbingSegmentResponse) => {
    const generationId = segment.generation_id;
    if (!generationId || !project) return;
    await withSegmentAction(segment.id, async () => {
      try {
        await apiClient.regenerateGeneration(generationId);
        await refreshProject();
        toast({
          title: 'Regeneration started',
          description: `Segment #${segment.srt_index} is being regenerated.`,
        });
      } catch (error) {
        toast({
          title: 'Regenerate failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      }
    });
  };

  const handleDeleteSegmentGeneration = async (segment: DubbingSegmentResponse) => {
    if (!project || (!segment.generation_id && !segment.cut_generation_id)) return;
    await withSegmentAction(segment.id, async () => {
      try {
        await apiClient.deleteDubbingSegmentGeneration(project.id, segment.id);
        await refreshProject();
        toast({
          title: segment.cut_generation_id ? 'Cut deleted' : 'Generation deleted',
          description: `Segment #${segment.srt_index} has been reset.`,
        });
      } catch (error) {
        toast({
          title: 'Delete failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      }
    });
  };

  const handleDeleteSegment = async (segment: DubbingSegmentResponse) => {
    if (!project) return;
    const confirmed = window.confirm(
      `Delete segment #${segment.srt_index}? This removes the SRT block and invalidates full narration/cuts.`,
    );
    if (!confirmed) return;

    await withSegmentAction(segment.id, async () => {
      try {
        const updatedProject = await apiClient.deleteDubbingSegment(project.id, segment.id);
        purgeProjectTimelineAudio(project.id);
        setProject(updatedProject);
        const fallbackSegment =
          updatedProject.segments.find((candidate) => candidate.segment_order >= segment.segment_order) ??
          updatedProject.segments[updatedProject.segments.length - 1] ??
          null;
        setSelectedSegmentId(fallbackSegment?.id ?? null);
        setEditingSegmentId(null);
        await loadProjects(updatedProject.id);
        toast({
          title: 'Segment deleted',
          description: `Segment #${segment.srt_index} was removed. Regenerate full narration/cuts when ready.`,
        });
      } catch (error) {
        toast({
          title: 'Delete segment failed',
          description: error instanceof Error ? error.message : 'Unknown error',
          variant: 'destructive',
        });
      }
    });
  };

  const applyAutoCutTimelineClips = (
    clips: DubbingAutoCutClipResponse[],
    sourceProject: DubbingProjectResponse | null = project,
  ) => {
    if (!sourceProject) return [];
    const orderedSegments = [...sourceProject.segments].sort((a, b) => a.segment_order - b.segment_order);
    const nextClips = resolveAudibleClipOverlaps(clips.map((clip, index): DubbingFullNarrationClip => ({
      id: clip.id,
      generationId: clip.generation_id,
      audioRevisionMs: sourceProject.full_narration_revision_ms ?? null,
      startMs: clip.start_ms,
      durationMs: clip.duration_ms,
      trimStartMs: clip.trim_start_ms,
      trimEndMs: clip.trim_end_ms,
      track: index % 2 === 0 ? 0 : 1,
      volume: clip.volume,
    })));
    setFullNarrationClips(nextClips);
    setTimelinePlaybackSource('full');
    setSelectedSegmentId(nextClips[0]?.id ?? null);
    setTimelinePlaybackTimeMs(orderedSegments[0]?.start_ms ?? 0);
    return nextClips;
  };

  const buildTimelineExportClips = () =>
    effectiveTimelinePlaybackSource === 'cuts'
      ? (sortedCutSegments.length > 0 ? sortedCutSegments : sortedGeneratedSegments)
          .map((segment) => {
            const generationId = segment.cut_generation_id ?? segment.generation_id;
            if (!generationId) return null;
            const durationMs = segment.cut_duration_ms ?? segment.actual_duration_ms ?? segment.target_duration_ms;
            return {
              id: segment.id,
              generation_id: generationId,
              start_ms: getSegmentTimelineStartMs(segment),
              duration_ms: durationMs,
              trim_start_ms: 0,
              trim_end_ms: 0,
              volume: 1,
            };
          })
          .filter((clip): clip is NonNullable<typeof clip> => clip !== null)
      : resolveAudibleClipOverlaps(fullNarrationClips)
          .filter((clip) => isClipAudible(clip))
          .map((clip) => ({
            id: clip.id,
            generation_id: clip.generationId,
            start_ms: clip.startMs,
            duration_ms: clip.durationMs,
            trim_start_ms: clip.trimStartMs,
            trim_end_ms: clip.trimEndMs,
            volume: clip.volume,
          }));

  const handleExportProjectAudio = async () => {
    if (!project) return;
    try {
      const timelineClips = buildTimelineExportClips();
      const blob = await apiClient.exportDubbingProjectAudio(project.id, { clips: timelineClips });
      await saveBlob(blob, `${project.name}.timeline.wav`, platform.filesystem.saveFile);
    } catch (error) {
      toast({
        title: 'Timeline export failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    }
  };

  const handleAutoCutTimeline = async () => {
    if (!project) return;
    if (!project.full_narration_generation_id || !project.full_narration_duration_ms) {
      toast({
        title: 'Full WAV required',
        description: 'Generate the full SRT narration before running Auto Cut.',
        variant: 'destructive',
      });
      return;
    }
    const orderedSegments = [...project.segments].sort((a, b) => a.segment_order - b.segment_order);
    if (orderedSegments.length === 0) return;

    setIsPostProcessing(true);
    try {
      const result = await apiClient.autoCutDubbingProject(project.id);
      const nextClips = applyAutoCutTimelineClips(result.clips);
      if (nextClips.length === 0) {
        throw new Error('Auto Cut returned no timeline clips.');
      }

      toast({
        title: 'Auto Cut complete',
        description: `${nextClips.length} word/RMS-aligned clip(s) were created from the full WAV.`,
      });
      if (AUTO_RESTART_SERVER_FOR_VRAM_RELEASE) {
        void restartServerForVramRelease('Auto Cut alignment', project.id);
      }
    } catch (error) {
      toast({
        title: 'Auto Cut failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsPostProcessing(false);
    }
  };

  const handleSuggestTempo = async () => {
    if (!project) return;
    if (!project.full_narration_generation_id || !project.full_narration_duration_ms) {
      toast({
        title: 'Full WAV required',
        description: 'Generate the full SRT narration before suggesting tempo.',
        variant: 'destructive',
      });
      return;
    }
    setIsSuggestingTempo(true);
    try {
      const suggestion = await apiClient.suggestDubbingTempo(project.id);
      setTempoSuggestion(suggestion);
      setTempoAdjustmentPercent(Math.max(-50, Math.min(50, (suggestion.multiplier - 1) * 100)));
      toast({
        title: 'Tempo suggestion ready',
        description: `Suggested global tempo: ${suggestion.multiplier.toFixed(3)}x (${suggestion.range}).`,
      });
    } catch (error) {
      toast({
        title: 'Tempo suggestion failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsSuggestingTempo(false);
    }
  };

  const handleApplySuggestedTempo = async () => {
    if (!project) return;
    setIsApplyingTempo(true);
    try {
      const result = await apiClient.applyDubbingTempo(project.id, {
        multiplier: Math.max(0.5, Math.min(1.5, selectedTempoMultiplier)),
      });
      const updatedProject = await loadProject(project.id, { silent: true });
      await loadProjects(project.id, { silent: true });
      const nextClips = applyAutoCutTimelineClips(result.clips, updatedProject);
      if (nextClips.length === 0) {
        throw new Error('Tempo was applied but Auto Cut returned no timeline clips.');
      }
      setTempoSuggestion(null);
      toast({
        title: 'Tempo applied',
        description: `Applied ${result.suggestion.multiplier.toFixed(3)}x and rebuilt ${nextClips.length} Auto Cut clip(s).`,
      });
    } catch (error) {
      toast({
        title: 'Apply tempo failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsApplyingTempo(false);
    }
  };

  const handleExportProjectPackage = async () => {
    if (!project) return;
    try {
      const timelineClips = buildTimelineExportClips();
      const blob = await apiClient.exportDubbingProjectPackage(project.id, { clips: timelineClips });
      await saveBlob(blob, `${project.name}.dubbing.zip`, platform.filesystem.saveFile);
    } catch (error) {
      toast({
        title: 'Package export failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    }
  };

  const handleCancelAllTasks = async () => {
    if (!project) return;
    setIsCancellingAll(true);
    try {
      const result = await apiClient.cancelDubbingProjectTasks(project.id);
      await refreshProject();
      toast({
        title: 'Tasks cancelled',
        description: result.message,
      });
    } catch (error) {
      toast({
        title: 'Cancel all failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setIsCancellingAll(false);
    }
  };

  const renderSegmentMenu = (segment: DubbingSegmentResponse) => {
    if (!segment.generation_id) return null;

    const isBusy = segmentActionId === segment.id;
    const canRetry = segment.status === 'failed';

    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-muted-foreground/60 hover:bg-muted hover:text-foreground"
            onClick={(event) => event.stopPropagation()}
            disabled={isBusy}
          >
            {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <MoreHorizontal className="h-4 w-4" />}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem
            onClick={(event) => {
              event.stopPropagation();
              handlePlaySegment(segment);
            }}
          >
            <Play className="mr-2 h-4 w-4" />
            Play
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={(event) => {
              event.stopPropagation();
              void handleDownloadSegmentAudio(segment);
            }}
          >
            <Download className="mr-2 h-4 w-4" />
            Export Audio
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={(event) => {
              event.stopPropagation();
              void handleExportSegmentPackage(segment);
            }}
          >
            <FileArchive className="mr-2 h-4 w-4" />
            Export Package
          </DropdownMenuItem>
          {canRetry ? (
            <DropdownMenuItem
              onClick={(event) => {
                event.stopPropagation();
                void handleRetryFailedSegment(segment);
              }}
            >
              <Wand2 className="mr-2 h-4 w-4" />
              Retry Failed Segment
            </DropdownMenuItem>
          ) : null}
          <DropdownMenuItem
            onClick={(event) => {
              event.stopPropagation();
              void handleRegenerateSegment(segment);
            }}
          >
            <RotateCcw className="mr-2 h-4 w-4" />
            Regenerate
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={(event) => {
              event.stopPropagation();
              void handleDeleteSegmentGeneration(segment);
            }}
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    );
  };

  return (
    <div className="flex h-full min-h-0 overflow-hidden -mx-8">
      <input
        ref={inputRef}
        type="file"
        accept=".srt"
        onChange={handleFileChange}
        className="hidden"
      />
      <Dialog
        open={renameDialogOpen}
        onOpenChange={(open) => {
          setRenameDialogOpen(open);
          if (!open) {
            setRenamingProject(null);
            setRenameProjectName('');
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit SRT2Voice Project</DialogTitle>
            <DialogDescription>Update the project name.</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="edit-srt2voice-name">Name</Label>
              <Input
                id="edit-srt2voice-name"
                value={renameProjectName}
                onChange={(event) => setRenameProjectName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    void handleSaveProjectRename();
                  }
                }}
                autoFocus
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={() => void handleSaveProjectRename()} disabled={isRenamingProject}>
              {isRenamingProject ? 'Saving...' : 'Save'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <div className="relative flex min-h-0 flex-1 gap-6 overflow-hidden">
        <div className="flex min-h-0 w-full max-w-[360px] shrink-0 flex-col overflow-hidden">
          <ListPane>
            <ListPaneHeader>
              <ListPaneTitleRow>
                <ListPaneTitle>SRT2Voice</ListPaneTitle>
                <ListPaneActions>
                  <Button
                    onClick={handlePickFile}
                    size="sm"
                    disabled={isImporting}
                    title={isImporting ? 'Importing...' : 'New SRT2Voice'}
                    aria-label={isImporting ? 'Importing SRT' : 'New SRT2Voice'}
                  >
                    {isImporting ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Plus className="mr-2 h-4 w-4" />
                    )}
                    New SRT2Voice
                  </Button>
                </ListPaneActions>
              </ListPaneTitleRow>
              <ListPaneSearch
                value={projectSearch}
                onChange={setProjectSearch}
                placeholder="Search SRT2Voice projects..."
              />
            </ListPaneHeader>

            <ListPaneScroll style={{ paddingBottom: isPlayerVisible ? '220px' : '140px' }}>
              {isProjectsLoading && projects.length === 0 ? (
                <div className="px-4 py-12 text-center text-sm text-muted-foreground">
                  Loading dubbing projects...
                </div>
              ) : projectsLoadError && projects.length === 0 ? (
                <div className="mx-4 rounded-2xl border-2 border-dashed border-destructive/30 px-5 py-12 text-center text-destructive">
                  <p className="text-sm">SRT2Voice server unavailable.</p>
                  <p className="mt-2 text-xs">{projectsLoadError}</p>
                </div>
              ) : filteredProjects.length === 0 ? (
                <div className="mx-4 rounded-2xl border-2 border-dashed border-muted px-5 py-12 text-center text-muted-foreground">
                  <p className="text-sm">No SRT2Voice project yet.</p>
                  <p className="mt-2 text-xs">Create a new project by importing an SRT file.</p>
                </div>
              ) : (
                <div className="space-y-1 px-4 pb-[300px]">
                  {filteredProjects.map((item) => {
                    const isActive = selectedProjectId === item.id;
                    return (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => selectDubbingProject(item.id)}
                        className={cn(
                          'block w-full rounded-lg border p-3 text-left transition-colors',
                          isActive
                            ? 'border-border bg-muted/70'
                            : 'border-transparent hover:bg-muted/30',
                        )}
                      >
                        <div className="mb-1.5 flex items-center gap-2">
                          <span className="text-[11px] font-medium text-muted-foreground">
                            {formatDate(item.updated_at)}
                          </span>
                          <div className="flex-1" />
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                className="h-7 w-7 text-muted-foreground hover:text-foreground"
                                disabled={deletingProjectId === item.id}
                                onClick={(event) => event.stopPropagation()}
                              >
                                {deletingProjectId === item.id ? (
                                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                ) : (
                                  <MoreHorizontal className="h-3.5 w-3.5" />
                                )}
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem
                                onClick={(event) => {
                                  event.stopPropagation();
                                  void handleRenameProject(item);
                                }}
                              >
                                <Pencil className="mr-2 h-4 w-4" />
                                Edit
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onClick={(event) => {
                                  event.stopPropagation();
                                  void handleDeleteProject(item.id);
                                }}
                              >
                                <Trash2 className="mr-2 h-4 w-4" />
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                        <div className="mb-2 text-[13px] leading-snug">
                          <span className="font-medium text-foreground">{item.name}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </ListPaneScroll>
          </ListPane>
        </div>

        <div className="flex min-h-0 flex-1 flex-col overflow-hidden pr-8">
          {!project ? (
            <div className="flex h-full items-center justify-center rounded-2xl border-2 border-dashed border-muted text-muted-foreground">
              <div className="max-w-md text-center">
                <h2 className="text-2xl font-bold text-foreground">SRT2Voice</h2>
                <p className="mt-3 text-sm">
                  Import an SRT file to create a speech timeline, then generate and edit a full narration.
                </p>
              </div>
            </div>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-hidden">
              <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                <div className="space-y-2">
                  <h1 className="text-3xl font-bold tracking-tight">{project.name}</h1>
                </div>

                <div className="flex flex-col items-start gap-3 xl:items-end">
                  <div className="flex flex-wrap justify-start gap-2 xl:justify-end">
                    <Button
                      variant="outline"
                      onClick={() => void handleGenerateFullNarration()}
                      disabled={!project || isGeneratingFullNarration || hasActiveGeneration}
                    >
                      {isGeneratingFullNarration || isFullNarrationActive ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Wand2 className="mr-2 h-4 w-4" />
                      )}
                      {isGeneratingFullNarration || isFullNarrationActive ? 'Narration running...' : 'Generate narration'}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => void handleAutoCutTimeline()}
                      disabled={!project || isPostProcessing || !hasFullNarrationAudio}
                    >
                      {isPostProcessing ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Scissors className="mr-2 h-4 w-4" />
                      )}
                      {isPostProcessing ? 'Auto Cutting...' : 'Auto Cut'}
                    </Button>
                    <Button variant="outline" onClick={() => void handleCancelAllTasks()} disabled={isCancellingAll}>
                      <Ban className="mr-2 h-4 w-4" />
                      {isCancellingAll ? 'Cancelling...' : 'Cancel All Tasks'}
                    </Button>
                  </div>
                  <div className="flex flex-wrap justify-start gap-2 xl:justify-end">
                    <Button variant="outline" onClick={() => void handleExportProjectAudio()}>
                      <Download className="mr-2 h-4 w-4" />
                      Export Timeline WAV
                    </Button>
                    <Button variant="outline" onClick={() => void handleExportProjectPackage()}>
                      <FileArchive className="mr-2 h-4 w-4" />
                      Export Package
                    </Button>
                  </div>
                </div>
              </div>

              {isFullNarrationActive ? (
                <div className="flex items-center gap-3 rounded-2xl border border-sky-500/30 bg-sky-500/10 px-4 py-3 text-sm text-sky-200">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <div className="min-w-0">
                    <div className="font-medium text-foreground">Audio generation is running</div>
                    <div className="truncate text-xs text-muted-foreground">
                      Continuous narration is being generated from cleaned SRT text. The timeline will show the full WAV when ready.
                    </div>
                  </div>
                </div>
              ) : null}

              <div className="grid min-h-0 flex-1 grid-cols-[340px_minmax(520px,1fr)] gap-6 overflow-hidden">
                <Card className="flex min-h-0 flex-col overflow-hidden">
                  <CardHeader>
                    <CardTitle>Generation Controls</CardTitle>
                    <CardDescription>Project-level settings for the active SRT.</CardDescription>
                  </CardHeader>
                  <CardContent className="flex min-h-0 flex-1 flex-col overflow-hidden px-6 pb-6">
                    <div
                      className={cn(
                        'min-h-0 flex-1 space-y-4 overflow-y-auto pr-2 pb-[320px]',
                        isPlayerVisible && BOTTOM_SAFE_AREA_PADDING,
                      )}
                    >
                      <div className="space-y-3 rounded-xl border border-border/60 bg-card/60 p-4">
                        <div className="space-y-1">
                          <div className="text-xs uppercase tracking-wide text-muted-foreground">Project</div>
                          <div className="font-medium">{project.name}</div>
                          <div className="text-xs capitalize text-muted-foreground">
                            Status: {project.status}
                          </div>
                        </div>

                        {project.full_narration_status ? (
                          <div
                            className={cn(
                              'rounded-lg border p-3 text-xs',
                              isFullNarrationActive
                                ? 'border-sky-500/30 bg-sky-500/10 text-sky-200'
                                : project.full_narration_status === 'failed'
                                  ? 'border-rose-500/30 bg-rose-500/10 text-rose-200'
                                  : 'border-border/60 bg-muted/30 text-muted-foreground',
                            )}
                          >
                            <div className="flex items-center gap-2">
                              {isFullNarrationActive ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                              <span className="font-medium text-foreground">
                                {fullNarrationStatusLabel ?? 'Full SRT beta'}
                              </span>
                            </div>
                            {project.full_narration_duration_ms ? (
                              <div className="mt-1 text-muted-foreground">
                                Duration: {formatDuration(project.full_narration_duration_ms)}
                                {project.full_narration_status === 'completed' &&
                                isPlausibleGenerationElapsed(
                                  project.full_narration_duration_ms,
                                  project.full_narration_generation_elapsed_ms,
                                )
                                  ? ` · Generated in ${formatSecondsWords(
                                      project.full_narration_generation_elapsed_ms,
                                    )}`
                                  : null}
                              </div>
                            ) : null}
                            {isPlausibleGenerationElapsed(
                              project.full_narration_duration_ms,
                              project.full_narration_generation_elapsed_ms,
                            ) &&
                            project.full_narration_status !== 'completed' ? (
                              <div className="mt-1 text-muted-foreground">
                                Generation stopped after {formatSeconds(project.full_narration_generation_elapsed_ms)}
                              </div>
                            ) : null}
                            {project.full_narration_error ? (
                              <div className="mt-1 text-rose-300">{project.full_narration_error}</div>
                            ) : null}
                          </div>
                        ) : null}
                        {project.post_processed_segment_count > 0 ? (
                          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs text-muted-foreground">
                            <div className="font-medium text-foreground">Post-processed cuts ready</div>
                            <div className="mt-1">
                              {project.post_processed_segment_count} segment cut(s) derived from the full narration WAV.
                            </div>
                          </div>
                        ) : null}
                        {hasAutoCutTimeline ? (
                          <div
                            className={cn(
                              'rounded-lg border p-3 text-xs',
                              tempoSuggestion?.range === 'safe'
                                ? 'border-emerald-500/30 bg-emerald-500/10'
                                : tempoSuggestion?.range === 'warning'
                                  ? 'border-amber-500/30 bg-amber-500/10'
                                  : tempoSuggestion?.range === 'critical'
                                    ? 'border-rose-500/30 bg-rose-500/10'
                                    : 'border-border/60 bg-muted/30',
                            )}
                          >
                            <div className="space-y-3">
                              <div className="flex items-center justify-between gap-3">
                                <div className="font-medium text-foreground">Suggested Tempo</div>
                                <div className="text-right text-muted-foreground">
                                  <span className="font-medium text-foreground">
                                    {selectedTempoMultiplier.toFixed(3)}x
                                  </span>
                                  <span className="ml-2">
                                    {tempoAdjustmentPercent > 0 ? '+' : ''}
                                    {tempoAdjustmentPercent.toFixed(0)}%
                                  </span>
                                </div>
                              </div>
                              {tempoSuggestion ? (
                                <div className="text-muted-foreground">
                                  {`${tempoSuggestion.multiplier.toFixed(3)}x · ${formatDelta(tempoSuggestion.delta_ms)} · ${tempoSuggestion.message}`}
                                </div>
                              ) : null}
                              <div className="space-y-2">
                                <Slider
                                  min={-50}
                                  max={50}
                                  step={1}
                                  value={[tempoAdjustmentPercent]}
                                  onValueChange={(values) => setTempoAdjustmentPercent(values[0] ?? 0)}
                                  disabled={isSuggestingTempo || isApplyingTempo || isFullNarrationActive}
                                />
                                <div className="flex justify-between text-[11px] text-muted-foreground">
                                  <span>Slower -50%</span>
                                  <span>0%</span>
                                  <span>Faster +50%</span>
                                </div>
                              </div>
                              <div className="flex justify-end gap-2">
                                <Button
                                  type="button"
                                  size="sm"
                                  variant="outline"
                                  title="Estimate a global atempo factor from word alignment."
                                  onClick={() => void handleSuggestTempo()}
                                  disabled={isSuggestingTempo || isApplyingTempo || isFullNarrationActive}
                                >
                                  {isSuggestingTempo ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                                  Suggest
                                </Button>
                                <Button
                                  type="button"
                                  size="sm"
                                  onClick={() => void handleApplySuggestedTempo()}
                                  disabled={isSuggestingTempo || isApplyingTempo || isFullNarrationActive}
                                >
                                  {isApplyingTempo ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                                  Apply
                                </Button>
                              </div>
                            </div>
                          </div>
                        ) : null}
                      </div>

                      <div className="space-y-2">
                        <div className="text-xs uppercase tracking-wide text-muted-foreground">Engine</div>
                        <Select
                          value={selectedEngineValue}
                          onValueChange={(value) => {
                            const option = availableEngineOptions.find((item) => item.value === value);
                            if (!option) return;
                            setSelectedEngine(option.engine);
                            if (option.engine === 'tada' && option.modelSize) {
                              setSelectedTadaModelSize(option.modelSize);
                            }
                          }}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {availableEngineOptions.map((option) => (
                              <SelectItem key={option.value} value={option.value}>
                                {option.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <div className="text-xs uppercase tracking-wide text-muted-foreground">Voice</div>
                        <Select value={selectedProfileId} onValueChange={setSelectedProfileId}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select a dubbing voice" />
                          </SelectTrigger>
                          <SelectContent>
                            {dubbingCompatibleProfiles.map((profile) => (
                              <SelectItem key={profile.id} value={profile.id}>
                                {profile.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <div className="text-xs uppercase tracking-wide text-muted-foreground">Language</div>
                        <Select
                          value={language}
                          onValueChange={(value) => {
                            if (isLanguageCode(value)) {
                              setLanguage(value);
                            }
                          }}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {availableLanguageOptions.map((option) => (
                              <SelectItem key={option.value} value={option.value}>
                                {option.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      {isQwenEngine ? (
                        <>
                          <div className="space-y-2">
                            <div className="text-xs uppercase tracking-wide text-muted-foreground">
                              Delivery Instructions
                            </div>
                            <Textarea
                              value={instruct}
                              onChange={(event) => setInstruct(event.target.value)}
                              placeholder="Calm voice, clear articulation, pedagogical tone, moderate pace, serious but warm."
                              className="min-h-[132px] resize-y"
                              maxLength={2000}
                            />
                          </div>

                          <div className="space-y-3 rounded-xl border border-border/60 bg-card/60 p-4">
                            <div className="space-y-1">
                              <div className="text-xs uppercase tracking-wide text-muted-foreground">
                                Project Pace Override
                              </div>
                            </div>
                            <div className="flex items-center justify-between text-sm">
                              <span>Current</span>
                              <span className="font-medium">{projectPaceValue.toFixed(2)}x</span>
                            </div>
                            <Slider
                              min={0.8}
                              max={1.2}
                              step={0.01}
                              value={[projectPaceValue]}
                              onValueChange={(values) => setProjectPaceValue(values[0] ?? 1)}
                            />
                            <div className="flex gap-2">
                              <Button
                                type="button"
                                size="sm"
                                onClick={() => void handleSaveProjectPace()}
                                disabled={isSavingProjectPace}
                              >
                                {isSavingProjectPace ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                                Save Project Pace
                              </Button>
                              <Button
                                type="button"
                                size="sm"
                                variant="outline"
                                onClick={() => void handleResetProjectPace()}
                                disabled={isSavingProjectPace}
                              >
                                Reset Auto
                              </Button>
                            </div>
                          </div>

                          <div className="space-y-3 rounded-xl border border-border/60 bg-card/60 p-4">
                            <div className="space-y-1">
                              <div className="text-xs uppercase tracking-wide text-muted-foreground">
                                Project Temperature
                              </div>
                            </div>
                            <div className="flex items-center justify-between text-sm">
                              <span>Current</span>
                              <span className="font-medium">{projectTemperatureValue.toFixed(2)}</span>
                            </div>
                            <Slider
                              min={0.1}
                              max={1.2}
                              step={0.01}
                              value={[projectTemperatureValue]}
                              onValueChange={(values) => setProjectTemperatureValue(values[0] ?? QWEN_DEFAULT_TEMPERATURE)}
                            />
                            <div className="flex gap-2">
                              <Button
                                type="button"
                                size="sm"
                                onClick={() => void handleSaveProjectTemperature()}
                                disabled={isSavingProjectTemperature}
                              >
                                {isSavingProjectTemperature ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                                Save Temperature
                              </Button>
                              <Button
                                type="button"
                                size="sm"
                                variant="outline"
                                onClick={() => void handleResetProjectTemperature()}
                                disabled={isSavingProjectTemperature}
                              >
                                Reset Default
                              </Button>
                            </div>
                          </div>
                        </>
                      ) : null}

                      {/* Phrase-group pace belongs to the abandoned segmented-generation workflow.
                          It stays hidden while we evaluate whether phrase-level tempo still has a
                          role in the full-narration SRT2Voice workflow. Do not expose without
                          reworking the UX and timing model. */}
                      <div className="hidden">
                        {selectedPaceGroup ? (
                          <>
                            <div className="flex items-center justify-between text-sm">
                              <span>{selectedPaceGroup.label}</span>
                              <span className="font-medium">{groupPaceValue.toFixed(2)}x</span>
                            </div>
                            <div className="text-xs text-muted-foreground">
                              Segments {selectedPaceGroup.segment_orders.join(', ')} · auto/effective pace {selectedPaceGroup.effective_pace.toFixed(2)}x
                            </div>
                            <Slider
                              min={0.8}
                              max={1.2}
                              step={0.01}
                              value={[groupPaceValue]}
                              onValueChange={(values) => setGroupPaceValue(values[0] ?? 1)}
                            />
                            <div className="flex gap-2">
                              <Button
                                type="button"
                                size="sm"
                                onClick={() => void handleSaveGroupPace()}
                                disabled={isSavingGroupPace}
                              >
                                {isSavingGroupPace ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                                Save Phrase Pace
                              </Button>
                              <Button
                                type="button"
                                size="sm"
                                variant="outline"
                                onClick={() => void handleResetGroupPace()}
                                disabled={isSavingGroupPace}
                              >
                                Reset Auto
                              </Button>
                            </div>
                          </>
                        ) : (
                          <div className="text-sm text-muted-foreground">
                            Select a segment to target its phrase group.
                          </div>
                        )}
                      </div>

                    </div>

                    <div className="mt-4 border-t border-border/60 bg-background/95 pt-4 backdrop-blur supports-[backdrop-filter]:bg-background/80">
                      <div className="grid gap-2">
                        <Button
                          className="h-auto w-full whitespace-normal py-3 text-left"
                          onClick={() => void handleAutoFitSegment()}
                          disabled={!selectedSegment || isAutoFitting}
                        >
                          {isAutoFitting ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <Wand2 className="mr-2 h-4 w-4" />
                          )}
                          {isAutoFitting ? 'Generating segment...' : 'Generate Selected Segment'}
                        </Button>

                        <Button
                          className="h-auto w-full whitespace-normal py-3 text-left"
                          variant="secondary"
                          onClick={() => void handleGenerateSegment()}
                          disabled={!selectedSegment || isGenerating || isAutoFitting}
                        >
                          {isGenerating ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <Wand2 className="mr-2 h-4 w-4" />
                          )}
                          {isGenerating ? 'Queueing segment...' : 'Manual Generate Selected Segment'}
                        </Button>

                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="flex min-h-0 flex-col overflow-hidden">
                  <CardHeader>
                    <CardTitle>Segments</CardTitle>
                    <CardDescription>
                      Click a segment text to edit its wording and timing.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="flex min-h-0 flex-1 flex-col overflow-hidden">
                    <div
                      className={cn(
                        'max-h-[430px] min-h-[300px] overflow-y-auto pr-2 pb-[300px] overscroll-contain',
                        isPlayerVisible && BOTTOM_SAFE_AREA_PADDING,
                      )}
                    >
                      <div className="space-y-3">
                        {project.segments.map((segment) => {
                          const isSelected = segment.id === selectedSegmentId;
                          const isProcessing = segment.status === 'generating';
                          const failureSummary = summarizeSegmentFailure(segment);
                          const readability = getSegmentReadability(segment);
                          const showGenerationStatus =
                            segment.status !== 'pending' ||
                            segment.fit_status !== 'unknown' ||
                            segment.delta_ms != null ||
                            !!segment.generation_error;

                          return (
                            <div
                              key={segment.id}
                              ref={(node) => {
                                segmentCardRefs.current[segment.id] = node;
                              }}
                              onClick={() => setSelectedSegmentId(segment.id)}
                              onDoubleClick={() => {
                                setSelectedSegmentId(segment.id);
                                setEditedSegmentText(segment.text);
                                setEditedSegmentStartTc(segment.start_tc);
                                setEditedSegmentEndTc(segment.end_tc);
                                setEditingSegmentId(segment.id);
                              }}
                              onKeyDown={(event) => {
                                if (
                                  event.target instanceof HTMLInputElement ||
                                  event.target instanceof HTMLTextAreaElement
                                ) {
                                  return;
                                }
                                if (event.key === 'Enter' || event.key === ' ') {
                                  event.preventDefault();
                                  setSelectedSegmentId(segment.id);
                                }
                              }}
                              role="button"
                              tabIndex={0}
                              className={`w-full rounded-xl border p-4 text-left transition-colors ${
                                isProcessing
                                  ? 'border-sky-500/50 bg-sky-500/5 shadow-[0_0_0_1px_rgba(14,165,233,0.15)]'
                                  : isSelected
                                    ? 'border-accent/50 bg-accent/5'
                                    : 'border-border/60 bg-card/60 hover:bg-card'
                              }`}
                            >
                              <div className="flex flex-col gap-3">
                                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                                  <div className="flex min-w-0 flex-wrap items-center gap-3">
                                    <div className="font-medium">#{segment.srt_index}</div>
                                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                      <TimerReset className="h-3.5 w-3.5" />
                                      {formatDuration(segment.target_duration_ms)}
                                    </div>
                                    <span className="text-xs text-muted-foreground">
                                      {segment.start_tc} {'->'} {segment.end_tc}
                                    </span>
                                    {isProcessing ? (
                                      <span className="inline-flex items-center gap-1 rounded-full border border-sky-500/20 bg-sky-500/10 px-2 py-0.5 text-xs text-sky-300">
                                        <Loader2 className="h-3 w-3 animate-spin" />
                                        processing
                                      </span>
                                    ) : null}
                                  </div>
                                  <div
                                    className="flex items-center justify-end gap-1"
                                    onClick={(event) => event.stopPropagation()}
                                  >
                                    <Button
                                      type="button"
                                      size="icon"
                                      variant="ghost"
                                      className="h-8 w-8 text-muted-foreground hover:text-destructive"
                                      onClick={() => void handleDeleteSegment(segment)}
                                      disabled={segmentActionId === segment.id}
                                      title="Delete SRT segment"
                                      aria-label={`Delete segment #${segment.srt_index}`}
                                    >
                                      {segmentActionId === segment.id ? (
                                        <Loader2 className="h-4 w-4 animate-spin" />
                                      ) : (
                                        <Trash2 className="h-4 w-4" />
                                      )}
                                    </Button>
                                    {renderSegmentMenu(segment)}
                                  </div>
                                </div>

                                <div className="flex flex-wrap items-center gap-2 text-xs">
                                  <span
                                    className={`rounded-full border px-2 py-0.5 ${readabilityBadgeClasses(readability.cpsWarning)}`}
                                    title={`${readability.characterCount} visible characters over ${formatDuration(
                                      segment.target_duration_ms,
                                    )}. Target: ${TARGET_CPS} CPS.`}
                                  >
                                    {readability.cps.toFixed(1)} CPS
                                  </span>
                                  <span
                                    className={`rounded-full border px-2 py-0.5 ${readabilityBadgeClasses(
                                      readability.wordsWarning,
                                    )}`}
                                    title={`${readability.wordCount} words over ${formatDuration(
                                      segment.target_duration_ms,
                                    )}. French narration target: ${TARGET_WORDS_PER_SECOND} words/s.`}
                                  >
                                    {readability.wordsPerSecond.toFixed(1)} w/s
                                  </span>
                                  {showGenerationStatus ? (
                                    <>
                                      {segment.fit_status !== 'warning' ? (
                                        <span
                                          className={`rounded-full border px-2 py-0.5 uppercase tracking-wide ${fitBadgeClasses(segment.fit_status)}`}
                                        >
                                          {segment.fit_status}
                                        </span>
                                      ) : null}
                                      {segment.status !== 'pending' ? (
                                        <span className="capitalize text-muted-foreground">{segment.status}</span>
                                      ) : null}
                                      {segment.delta_ms != null ? (
                                        <span className="text-muted-foreground">
                                          Delta {formatDelta(segment.delta_ms)}
                                        </span>
                                      ) : null}
                                      {segment.generation_error ? (
                                        <span className="rounded-full border border-rose-500/20 bg-rose-500/10 px-2 py-0.5 text-rose-300">
                                          runtime error
                                        </span>
                                      ) : segment.fit_status === 'warning' ? (
                                        <span className="rounded-full border border-amber-500/20 bg-amber-500/10 px-2 py-0.5 text-amber-300">
                                          timing overflow
                                        </span>
                                      ) : null}
                                    </>
                                  ) : null}
                                </div>

                                {editingSegmentId === segment.id ? (
                                  <div
                                    className="space-y-3 rounded-xl border border-border/60 bg-background/80 p-3"
                                    onClick={(event) => event.stopPropagation()}
                                    onDoubleClick={(event) => event.stopPropagation()}
                                  >
                                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                                      <div className="space-y-1">
                                        <div className="text-xs uppercase tracking-wide text-muted-foreground">
                                          Start
                                        </div>
                                        <Input
                                          value={editedSegmentStartTc}
                                          onChange={(event) => setEditedSegmentStartTc(event.target.value)}
                                          onKeyDown={(event) => event.stopPropagation()}
                                          placeholder="00:00:00,000"
                                          className="font-mono text-xs"
                                        />
                                      </div>
                                      <div className="space-y-1">
                                        <div className="text-xs uppercase tracking-wide text-muted-foreground">End</div>
                                        <Input
                                          value={editedSegmentEndTc}
                                          onChange={(event) => setEditedSegmentEndTc(event.target.value)}
                                          onKeyDown={(event) => event.stopPropagation()}
                                          placeholder="00:00:00,000"
                                          className="font-mono text-xs"
                                        />
                                      </div>
                                    </div>
                                    <Textarea
                                      value={editedSegmentText}
                                      onChange={(event) => setEditedSegmentText(event.target.value)}
                                      onKeyDown={(event) => event.stopPropagation()}
                                      className="min-h-[120px] resize-y text-sm leading-6"
                                      maxLength={5000}
                                      autoFocus
                                    />
                                    <div className="flex gap-2">
                                      <Button
                                        type="button"
                                        size="sm"
                                        onClick={() => void handleSaveSegmentText()}
                                        disabled={!hasEditedSegmentChanges || isSavingSegmentText}
                                      >
                                        {isSavingSegmentText ? (
                                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        ) : null}
                                        Save Text
                                      </Button>
                                      <Button
                                        type="button"
                                        size="sm"
                                        variant="outline"
                                        onClick={() => void handleSaveSegmentTimingFields()}
                                        disabled={!hasEditedTimingChanges || isSavingSegmentTiming}
                                      >
                                        {isSavingSegmentTiming ? (
                                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        ) : null}
                                        Save Timecode
                                      </Button>
                                      <Button
                                        type="button"
                                        size="sm"
                                        variant="destructive"
                                        onClick={() => void handleDeleteSegment(segment)}
                                        disabled={segmentActionId === segment.id}
                                      >
                                        {segmentActionId === segment.id ? (
                                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        ) : (
                                          <Trash2 className="mr-2 h-4 w-4" />
                                        )}
                                        Delete Segment
                                      </Button>
                                      <Button
                                        type="button"
                                        size="sm"
                                        variant="outline"
                                        onClick={() => {
                                          setEditedSegmentText(segment.text);
                                          setEditedSegmentStartTc(segment.start_tc);
                                          setEditedSegmentEndTc(segment.end_tc);
                                          setEditingSegmentId(null);
                                        }}
                                        disabled={isSavingSegmentText || isSavingSegmentTiming}
                                      >
                                        Cancel
                                      </Button>
                                    </div>
                                  </div>
                                ) : (
                                  <p className="text-sm leading-6">{segment.text}</p>
                                )}

                                {failureSummary ? (
                                  <p className="line-clamp-2 text-xs text-amber-300">{failureSummary}</p>
                                ) : null}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}
        </div>
      </div>
      {project && dubbingTimelineClips.length > 0 ? (
        <AudioTrackEditor
          clips={dubbingTimelineClips}
          selectedClipId={
            selectedSegmentId && dubbingTimelineClips.some((clip) => clip.id === selectedSegmentId)
              ? selectedSegmentId
              : null
          }
          currentTimeMs={timelinePlayheadMs}
          isPlaying={isTimelinePlaying}
          height={timelineEditorHeight}
          onHeightChange={setTimelineEditorHeight}
          onSelectClip={(clipId) => {
            if (!clipId) {
              setSelectedSegmentId(null);
              return;
            }
            if (clipId.startsWith('reference-')) {
              selectAndScrollToSegment(clipId.slice('reference-'.length));
              return;
            }
            setTimelinePlaybackSource(isFullNarrationClipId(clipId) ? 'full' : 'cuts');
            if (project.segments.some((segment) => segment.id === clipId)) {
              selectAndScrollToSegment(clipId);
            } else {
              setSelectedSegmentId(clipId);
            }
          }}
          onSeek={(timeMs) => handleTimelineSeek(timeMs, isTimelinePlaying)}
          onPreviewSeek={setTimelinePlaybackTimeMs}
          onPlayPause={handlePlayTimeline}
          onStop={handleStopTimelinePlayback}
          onMoveClip={(clipId, startMs, track) => {
            if (isFullNarrationClipId(clipId)) {
              setFullNarrationClips((current) => {
                const source = current.find((clip) => clip.id === clipId);
                if (!source) return current;
                const nextClip = { ...source, startMs, track };
                if (hasAudibleOverlapWithCandidate(current, nextClip)) {
                  toast({
                    title: 'Audible overlap blocked',
                    description: 'Mute one of the clips before allowing overlap.',
                    variant: 'destructive',
                  });
                  return current;
                }
                return current.map((clip) => (clip.id === clipId ? nextClip : clip));
              });
              return;
            }
            const segment = project.segments.find((candidate) => candidate.id === clipId);
            if (!segment) return;
            setSegmentClipStarts((current) => ({ ...current, [clipId]: startMs }));
            setSegmentLanes((current) => ({
              ...current,
              [clipId]: track === 1 || track === 0 || track === -1 ? track : 0,
            }));
          }}
          onTrimClip={(clipId, trimStartMs, trimEndMs) => {
            if (isFullNarrationClipId(clipId)) {
              setFullNarrationClips((current) => {
                const source = current.find((clip) => clip.id === clipId);
                if (!source) return current;
                const nextClip = { ...source, trimStartMs, trimEndMs };
                if (hasAudibleOverlapWithCandidate(current, nextClip)) {
                  toast({
                    title: 'Audible overlap blocked',
                    description: 'The trim would make this clip overlap another audible clip.',
                    variant: 'destructive',
                  });
                  return current;
                }
                return current.map((clip) => (clip.id === clipId ? nextClip : clip));
              });
              return;
            }
            toast({
              title: 'Trim is not persisted yet',
              description: 'Only full WAV clips can be trimmed in place for now.',
            });
          }}
          onSplitClip={(clipId, splitTimeMs) => {
            if (isFullNarrationClipId(clipId)) {
              splitFullNarrationClip(clipId, splitTimeMs);
              return;
            }
            setSelectedSegmentId(clipId);
            void handleTimelineCut(clipId);
          }}
          onDeleteClip={(clipId) => {
            if (isFullNarrationClipId(clipId)) {
              setFullNarrationClips((current) => current.filter((clip) => clip.id !== clipId));
              setSelectedSegmentId(null);
              return;
            }
            const segment = project.segments.find((candidate) => candidate.id === clipId);
            if (segment) void handleDeleteSegmentGeneration(segment);
          }}
          onDuplicateClip={(clipId) => {
            if (isFullNarrationClipId(clipId)) {
              setFullNarrationClips((current) => {
                const source = current.find((clip) => clip.id === clipId);
                if (!source) return current;
                const durationMs = getFullClipEffectiveDurationMs(source);
                const startMs = findNextNonOverlappingStart(
                  current,
                  source.startMs + durationMs,
                  durationMs,
                );
                const duplicate: DubbingFullNarrationClip = {
                  ...source,
                  id: `${source.id}-copy-${Date.now()}`,
                  startMs,
                  track: source.track === 0 ? 1 : 0,
                };
                return [...current, duplicate];
              });
              return;
            }
            toast({
              title: 'Duplicate is full-WAV only for now',
              description: 'Generated segment duplication will be wired after segment clips become persistent.',
            });
          }}
          onRegenerateClip={(clipId) => {
            const segment = project.segments.find((candidate) => candidate.id === clipId);
            if (segment) void handleRegenerateSegment(segment);
          }}
          onVolumeChange={(clipId, volume) => {
            if (isFullNarrationClipId(clipId)) {
              setFullNarrationClips((current) => {
                const source = current.find((clip) => clip.id === clipId);
                if (!source) return current;
                const nextClip = { ...source, volume };
                if (hasAudibleOverlapWithCandidate(current, nextClip)) {
                  toast({
                    title: 'Audible overlap blocked',
                    description: 'This clip cannot be unmuted while it overlaps another audible clip.',
                    variant: 'destructive',
                  });
                  return current;
                }
                return current.map((clip) => (clip.id === clipId ? nextClip : clip));
              });
              return;
            }
            setSelectedSegmentId(clipId);
            handleTimelineVolumeChange(Math.round(volume * 100));
          }}
        />
      ) : null}
    </div>
  );
}
