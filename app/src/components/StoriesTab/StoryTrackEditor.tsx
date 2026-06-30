import {
  Check,
  Copy,
  GalleryVerticalEnd,
  GripHorizontal,
  Minus,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Scissors,
  Square,
  Trash2,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Slider } from '@/components/ui/slider';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { StoryItemDetail } from '@/lib/api/types';
import {
  useDuplicateStoryItem,
  useMoveStoryItem,
  useRemoveStoryItem,
  useSetStoryItemVersion,
  useSplitStoryItem,
  useTrimStoryItem,
  useUpdateStoryItemVolume,
} from '@/lib/hooks/useStories';
import { cn } from '@/lib/utils/cn';
import { useGenerationStore } from '@/stores/generationStore';
import { useStoryStore } from '@/stores/storyStore';

// Clip waveform component with trim support
function ClipWaveform({
  generationId,
  versionId,
  width,
  trimStartMs,
  trimEndMs,
  duration,
}: {
  generationId: string;
  versionId?: string;
  width: number;
  trimStartMs: number;
  trimEndMs: number;
  duration: number;
}) {
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);

  // Calculate the full waveform width based on the original duration
  // The visible portion (width) represents the effective duration after trimming
  const effectiveDurationMs = duration * 1000 - trimStartMs - trimEndMs;
  const fullWaveformWidth =
    effectiveDurationMs > 0 ? (width / effectiveDurationMs) * (duration * 1000) : width;

  // Calculate how much to offset the waveform to hide the trimmed start
  const offsetX =
    effectiveDurationMs > 0 ? (trimStartMs / (duration * 1000)) * fullWaveformWidth : 0;

  useEffect(() => {
    if (!waveformRef.current || fullWaveformWidth < 20) return;

    // Get CSS colors
    const root = document.documentElement;
    const getCSSVar = (varName: string) => {
      const value = getComputedStyle(root).getPropertyValue(varName).trim();
      return value ? `hsl(${value})` : '';
    };

    const waveColor = getCSSVar('--accent-foreground');

    // Hand WaveSurfer a muted <audio> element so the MediaElement backend
    // can never bleed audio. Web Audio is doing the actual playback in
    // useStoryPlayback; this clip waveform exists purely for the visual.
    // Without this, long imported clips (MP3 / M4A) end up audible from
    // wavesurfer's own element on top of the timeline, and that element
    // doesn't get paused by stopAllSources().
    const mediaElement = document.createElement('audio');
    mediaElement.muted = true;
    mediaElement.preload = 'metadata';

    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      media: mediaElement,
      waveColor,
      progressColor: waveColor,
      cursorWidth: 0,
      barWidth: 1,
      barRadius: 1,
      barGap: 1,
      height: 28,
      normalize: true,
      interact: false,
    });

    wavesurferRef.current = wavesurfer;

    const audioUrl = versionId
      ? apiClient.getVersionAudioUrl(versionId)
      : apiClient.getAudioUrl(generationId);
    wavesurfer.load(audioUrl).catch(() => {
      // Ignore load errors
    });

    return () => {
      wavesurfer.destroy();
      wavesurferRef.current = null;
    };
  }, [generationId, versionId, fullWaveformWidth]);

  return (
    <div className="w-full h-full opacity-60 overflow-hidden">
      {/* Inner container that holds the full waveform, offset to show only visible portion */}
      <div
        ref={waveformRef}
        style={{
          width: `${fullWaveformWidth}px`,
          transform: `translateX(-${offsetX}px)`,
        }}
        className="h-full"
      />
    </div>
  );
}

// Per-clip volume popover. Local state drives the slider during a drag so
// each pointer-move pixel doesn't fire a PATCH; commits on release.
function ClipVolumePopover({
  storyId,
  itemId,
  volume,
  onChange,
}: {
  storyId: string;
  itemId: string;
  volume: number;
  onChange: (value: number) => void;
}) {
  const [localVolume, setLocalVolume] = useState(volume);
  // Re-sync when the selected clip changes or the persisted value updates
  // out-of-band (split/duplicate carry the value forward).
  useEffect(() => {
    setLocalVolume(volume);
  }, [volume, itemId, storyId]);

  const display = Math.round(localVolume * 100);
  const Icon = localVolume === 0 ? VolumeX : Volume2;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          title={`Volume — ${display}%`}
          aria-label="Adjust clip volume"
        >
          <Icon className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="center" className="w-56 p-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-muted-foreground">Volume</span>
          <span className="text-xs tabular-nums">{display}%</span>
        </div>
        <Slider
          value={[localVolume * 100]}
          onValueChange={([v]) => setLocalVolume(v / 100)}
          onValueCommit={([v]) => onChange(v / 100)}
          min={0}
          max={200}
          step={1}
          aria-label="Clip volume"
        />
        <div className="flex justify-between mt-2 text-[10px] text-muted-foreground tabular-nums">
          <span>0%</span>
          <span>100%</span>
          <span>200%</span>
        </div>
      </PopoverContent>
    </Popover>
  );
}

interface StoryTrackEditorProps {
  storyId: string;
  items: StoryItemDetail[];
}

const TRACK_HEIGHT = 48;
const TIME_RULER_HEIGHT = 24; // h-6 = 1.5rem = 24px
const SCRUB_BAR_HEIGHT = 16;
const LABEL_COL_WIDTH = 64; // w-16 = 4rem = 64px
// Zoom is expressed to the user as how many seconds of timeline are visible
// at once. Min scope = the most you can zoom IN; max scope = the entire
// project. Default scope is what we land on when the editor first measures.
const MIN_VISIBLE_SECONDS = 10;
const DEFAULT_VISIBLE_SECONDS = 60;
const FALLBACK_PIXELS_PER_SECOND = 50; // used until containerWidth is measured
const DEFAULT_TRACKS = [1, 0, -1]; // Default 3 tracks
const MIN_EDITOR_HEIGHT = 120;
const MAX_EDITOR_HEIGHT = 500;

export function StoryTrackEditor({ storyId, items }: StoryTrackEditorProps) {
  const [pixelsPerSecond, setPixelsPerSecond] = useState(FALLBACK_PIXELS_PER_SECOND);
  const hasAppliedDefaultZoomRef = useRef(false);
  const [draggingItem, setDraggingItem] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [dragPosition, setDragPosition] = useState({ x: 0, y: 0 });
  const [isResizing, setIsResizing] = useState(false);
  const [containerWidth, setContainerWidth] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const tracksRef = useRef<HTMLDivElement>(null);
  const resizeStartY = useRef(0);
  const resizeStartHeight = useRef(0);
  const moveItem = useMoveStoryItem();
  const trimItem = useTrimStoryItem();
  const splitItem = useSplitStoryItem();
  const duplicateItem = useDuplicateStoryItem();
  const removeItem = useRemoveStoryItem();
  const setItemVersion = useSetStoryItemVersion();
  const updateVolume = useUpdateStoryItemVolume();
  const { toast } = useToast();
  const addPendingGeneration = useGenerationStore((s) => s.addPendingGeneration);
  // User-added empty tracks. Live in component state because a track only
  // earns its keep once a clip lands on it — no need to persist an unused
  // row across reloads.
  const [extraTracks, setExtraTracks] = useState<number[]>([]);

  // Selection state
  const selectedClipId = useStoryStore((state) => state.selectedClipId);
  const setSelectedClipId = useStoryStore((state) => state.setSelectedClipId);

  // Selected clip item (for version picker)
  const selectedItem = useMemo(
    () => (selectedClipId ? items.find((i) => i.id === selectedClipId) : undefined),
    [selectedClipId, items],
  );
  const selectedItemVersions = selectedItem?.versions;
  const hasMultipleVersions = selectedItemVersions && selectedItemVersions.length > 1;

  // Determine which version label is active for the selected clip
  const activeVersionLabel = useMemo(() => {
    if (!selectedItem || !selectedItemVersions) return null;
    // If the item has a pinned version_id, find its label
    if (selectedItem.version_id) {
      const pinned = selectedItemVersions.find((v) => v.id === selectedItem.version_id);
      return pinned?.label ?? null;
    }
    // Otherwise use the generation's default version
    const defaultVersion = selectedItemVersions.find((v) => v.is_default);
    return defaultVersion?.label ?? null;
  }, [selectedItem, selectedItemVersions]);

  const handleSetVersion = useCallback(
    (versionId: string | null) => {
      if (!selectedClipId) return;
      setItemVersion.mutate(
        {
          storyId,
          itemId: selectedClipId,
          data: { version_id: versionId },
        },
        {
          onError: (error) => {
            toast({
              title: 'Failed to set version',
              description: error instanceof Error ? error.message : String(error),
              variant: 'destructive',
            });
          },
        },
      );
    },
    [selectedClipId, storyId, setItemVersion, toast],
  );

  // Trim state
  const [trimmingItem, setTrimmingItem] = useState<string | null>(null);
  const [trimSide, setTrimSide] = useState<'start' | 'end' | null>(null);
  const [trimStartX, setTrimStartX] = useState(0);
  const [tempTrimValues, setTempTrimValues] = useState<{
    trim_start_ms: number;
    trim_end_ms: number;
  } | null>(null);

  // Track editor height from store (shared with FloatingGenerateBox)
  const editorHeight = useStoryStore((state) => state.trackEditorHeight);
  const setEditorHeight = useStoryStore((state) => state.setTrackEditorHeight);

  // Playback state
  const isPlaying = useStoryStore((state) => state.isPlaying);
  const currentTimeMs = useStoryStore((state) => state.currentTimeMs);
  const playbackStoryId = useStoryStore((state) => state.playbackStoryId);
  const play = useStoryStore((state) => state.play);
  const pause = useStoryStore((state) => state.pause);
  const stop = useStoryStore((state) => state.stop);
  const seek = useStoryStore((state) => state.seek);
  const setActiveStory = useStoryStore((state) => state.setActiveStory);

  const isActiveStory = playbackStoryId === storyId;
  const isCurrentlyPlaying = isPlaying && isActiveStory;

  // Auto-activate this story when the editor is shown so playhead is visible
  useEffect(() => {
    if (items.length > 0 && !isActiveStory) {
      const totalDuration = Math.max(
        ...items.map((item) => {
          const trimStart = item.trim_start_ms || 0;
          const trimEnd = item.trim_end_ms || 0;
          const effectiveDuration = item.duration * 1000 - trimStart - trimEnd;
          return item.start_time_ms + effectiveDuration;
        }),
        0,
      );
      setActiveStory(storyId, items, totalDuration);
    }
  }, [storyId, items, isActiveStory, setActiveStory]);

  // Sort items by start time for play
  const sortedItems = useMemo(() => {
    return [...items].sort((a, b) => a.start_time_ms - b.start_time_ms);
  }, [items]);

  const handlePlayPause = () => {
    if (isCurrentlyPlaying) {
      pause();
    } else {
      play(storyId, sortedItems);
    }
  };

  const handleStop = () => {
    stop();
  };

  // Calculate unique tracks from items, always showing at least 3 default
  // tracks. ``extraTracks`` lets the user open a fresh row without first
  // having to drag a clip there.
  const tracks = useMemo(() => {
    const trackSet = new Set([
      ...DEFAULT_TRACKS,
      ...items.map((item) => item.track),
      ...extraTracks,
    ]);
    return Array.from(trackSet).sort((a, b) => b - a); // Higher tracks on top
  }, [items, extraTracks]);

  const handleAddTrackAbove = useCallback(() => {
    setExtraTracks((prev) => {
      const all = new Set([...DEFAULT_TRACKS, ...items.map((i) => i.track), ...prev]);
      const next = (all.size > 0 ? Math.max(...all) : 0) + 1;
      return [...prev, next];
    });
  }, [items]);

  const handleAddTrackBelow = useCallback(() => {
    setExtraTracks((prev) => {
      const all = new Set([...DEFAULT_TRACKS, ...items.map((i) => i.track), ...prev]);
      const next = (all.size > 0 ? Math.min(...all) : 0) - 1;
      return [...prev, next];
    });
  }, [items]);

  // Track container width for full-width minimum
  useEffect(() => {
    const container = tracksRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });

    observer.observe(container);
    // Set initial width
    setContainerWidth(container.clientWidth);

    return () => observer.disconnect();
  }, []);

  // Horizontal scrollbar state
  const [timelineScrollLeft, setTimelineScrollLeft] = useState(0);
  const [scrollbarTrackWidth, setScrollbarTrackWidth] = useState(0);
  const scrollbarTrackRef = useRef<HTMLDivElement>(null);
  const scrollbarDragRef = useRef<{
    mode: 'pan' | 'left' | 'right';
    startX: number;
    startScrollLeft: number;
    startPixelsPerSecond: number;
  } | null>(null);
  // Anchor the visible left/right edge time during a zoom drag so the edge
  // the user isn't dragging stays pinned in place across pixelsPerSecond changes.
  const zoomAnchorRef = useRef<{ type: 'left' | 'right'; timeMs: number } | null>(null);

  // Mirror the timeline's scrollLeft into state so the scrollbar thumb tracks it
  useEffect(() => {
    const el = tracksRef.current;
    if (!el) return;
    const onScroll = () => setTimelineScrollLeft(el.scrollLeft);
    el.addEventListener('scroll', onScroll);
    setTimelineScrollLeft(el.scrollLeft);
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  // Track scrollbar track width for thumb sizing
  useEffect(() => {
    const el = scrollbarTrackRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setScrollbarTrackWidth(entry.contentRect.width);
      }
    });
    ro.observe(el);
    setScrollbarTrackWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  // Calculate effective duration (accounting for trims)
  const getEffectiveDuration = (item: StoryItemDetail) => {
    return item.duration * 1000 - (item.trim_start_ms || 0) - (item.trim_end_ms || 0);
  };

  // Calculate total duration (using effective durations)
  const totalDurationMs = useMemo(() => {
    if (items.length === 0) return 10000; // Default 10 seconds
    return Math.max(...items.map((item) => item.start_time_ms + getEffectiveDuration(item)), 10000);
  }, [items, getEffectiveDuration]);

  // Zoom bounds are framed in seconds-of-timeline-visible-at-once (the
  // "scope") rather than abstract pixels-per-second so the bar reflects
  // something meaningful: fully zoomed out shows the entire project, fully
  // zoomed in shows MIN_VISIBLE_SECONDS. Convert to pixels using the visible
  // track area (container minus the sticky label column).
  const visibleTrackWidth = Math.max(0, containerWidth - LABEL_COL_WIDTH);
  const projectSeconds = totalDurationMs / 1000;
  const { minPps, maxPps } = useMemo(() => {
    if (visibleTrackWidth <= 0 || projectSeconds <= 0) {
      return { minPps: 10, maxPps: 200 };
    }
    const min = visibleTrackWidth / projectSeconds;
    const max = visibleTrackWidth / MIN_VISIBLE_SECONDS;
    // For projects shorter than MIN_VISIBLE_SECONDS the entire bar collapses
    // to one point; clamp so the range stays non-inverted.
    return { minPps: min, maxPps: Math.max(max, min) };
  }, [visibleTrackWidth, projectSeconds]);

  // Apply the default scope (60 s, or the whole project if shorter) once we
  // have a real measurement to convert it into pixels-per-second.
  useEffect(() => {
    if (hasAppliedDefaultZoomRef.current) return;
    if (visibleTrackWidth <= 0) return;
    const defaultScope = Math.min(DEFAULT_VISIBLE_SECONDS, Math.max(projectSeconds, MIN_VISIBLE_SECONDS));
    setPixelsPerSecond(visibleTrackWidth / defaultScope);
    hasAppliedDefaultZoomRef.current = true;
  }, [visibleTrackWidth, projectSeconds]);

  // Re-clamp the current zoom whenever the bounds shift (project length
  // changed, window resized) so the user can't end up parked outside the
  // valid range from a previous session.
  useEffect(() => {
    setPixelsPerSecond((prev) => Math.max(minPps, Math.min(maxPps, prev)));
  }, [minPps, maxPps]);

  // Calculate timeline width - at least full container width
  const contentWidth = (totalDurationMs / 1000) * pixelsPerSecond + 200; // Content width with padding
  const timelineWidth = Math.max(contentWidth, containerWidth);

  // Generate time markers
  const timeMarkers = useMemo(() => {
    const markers: number[] = [];
    // Determine interval based on zoom level
    let intervalMs = 5000; // 5 seconds
    if (pixelsPerSecond > 100) intervalMs = 1000;
    else if (pixelsPerSecond > 50) intervalMs = 2000;
    else if (pixelsPerSecond < 20) intervalMs = 10000;

    for (let ms = 0; ms <= totalDurationMs + intervalMs; ms += intervalMs) {
      markers.push(ms);
    }
    return markers;
  }, [totalDurationMs, pixelsPerSecond]);

  const formatTime = (ms: number): string => {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const msToPixels = useCallback((ms: number) => (ms / 1000) * pixelsPerSecond, [pixelsPerSecond]);

  const pixelsToMs = useCallback((px: number) => (px / pixelsPerSecond) * 1000, [pixelsPerSecond]);

  const handleZoomIn = () => {
    setPixelsPerSecond((prev) => Math.min(prev * 1.5, maxPps));
  };

  const handleZoomOut = () => {
    setPixelsPerSecond((prev) => Math.max(prev / 1.5, minPps));
  };

  // Resize handlers
  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsResizing(true);
      resizeStartY.current = e.clientY;
      resizeStartHeight.current = editorHeight;
    },
    [editorHeight],
  );

  const handleResizeMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing) return;
      const deltaY = resizeStartY.current - e.clientY;
      const newHeight = Math.min(
        MAX_EDITOR_HEIGHT,
        Math.max(MIN_EDITOR_HEIGHT, resizeStartHeight.current + deltaY),
      );
      setEditorHeight(newHeight);
    },
    [isResizing, setEditorHeight],
  );

  const handleResizeEnd = useCallback(() => {
    setIsResizing(false);
  }, []);

  // Add global mouse listeners for resizing
  useEffect(() => {
    if (isResizing) {
      window.addEventListener('mousemove', handleResizeMove);
      window.addEventListener('mouseup', handleResizeEnd);
      return () => {
        window.removeEventListener('mousemove', handleResizeMove);
        window.removeEventListener('mouseup', handleResizeEnd);
      };
    }
  }, [isResizing, handleResizeMove, handleResizeEnd]);

  const handleTimelineClick = (e: React.MouseEvent<HTMLElement>) => {
    if (!tracksRef.current || draggingItem || trimmingItem) return;
    const rect = tracksRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left + tracksRef.current.scrollLeft - LABEL_COL_WIDTH;
    const timeMs = Math.max(0, pixelsToMs(x));
    seek(timeMs);
    // Deselect clip when clicking on timeline
    setSelectedClipId(null);
  };

  const handleClipClick = (e: React.MouseEvent, item: StoryItemDetail) => {
    e.stopPropagation();
    if (draggingItem || trimmingItem) return;
    setSelectedClipId(item.id);
  };

  const handleTrimStart = (e: React.MouseEvent, item: StoryItemDetail, side: 'start' | 'end') => {
    e.stopPropagation();
    if (!tracksRef.current) return;
    setTrimmingItem(item.id);
    setTrimSide(side);
    setSelectedClipId(item.id);
    setTrimStartX(e.clientX);
    trimStartItemRef.current = {
      item,
      initialTrimStart: item.trim_start_ms || 0,
      initialTrimEnd: item.trim_end_ms || 0,
    };
  };

  const trimStartItemRef = useRef<{
    item: StoryItemDetail;
    initialTrimStart: number;
    initialTrimEnd: number;
  } | null>(null);

  const handleTrimMove = useCallback(
    (e: MouseEvent) => {
      if (!trimmingItem || !trimSide || !trimStartItemRef.current) return;

      const deltaX = e.clientX - trimStartX;
      const deltaMs = pixelsToMs(deltaX); // Signed delta in milliseconds

      const { item, initialTrimStart, initialTrimEnd } = trimStartItemRef.current;
      const originalDurationMs = item.duration * 1000;

      let newTrimStart = initialTrimStart;
      let newTrimEnd = initialTrimEnd;

      if (trimSide === 'start') {
        // Moving right increases trim_start (trims more from start)
        // Moving left decreases trim_start (restores from start)
        newTrimStart = Math.round(
          Math.max(
            0,
            Math.min(initialTrimStart + deltaMs, originalDurationMs - initialTrimEnd - 100),
          ),
        );
      } else {
        // Moving right decreases trim_end (restores from end)
        // Moving left increases trim_end (trims more from end)
        newTrimEnd = Math.round(
          Math.max(
            0,
            Math.min(initialTrimEnd - deltaMs, originalDurationMs - initialTrimStart - 100),
          ),
        );
      }

      // Validate that we don't exceed duration
      if (newTrimStart + newTrimEnd >= originalDurationMs - 100) {
        return; // Don't allow trimming to less than 100ms
      }

      // Update temporary trim values for visual feedback
      setTempTrimValues({
        trim_start_ms: newTrimStart,
        trim_end_ms: newTrimEnd,
      });
    },
    [trimmingItem, trimSide, trimStartX, pixelsToMs],
  );

  const handleTrimEnd = useCallback(() => {
    if (!trimmingItem || !trimSide || !trimStartItemRef.current) {
      setTrimmingItem(null);
      setTrimSide(null);
      setTempTrimValues(null);
      trimStartItemRef.current = null;
      return;
    }

    const { initialTrimStart, initialTrimEnd } = trimStartItemRef.current;

    // Use temporary trim values if available, otherwise use initial values
    // Ensure values are integers for the backend
    const finalTrimStart = Math.round(tempTrimValues?.trim_start_ms ?? initialTrimStart);
    const finalTrimEnd = Math.round(tempTrimValues?.trim_end_ms ?? initialTrimEnd);

    // Only update if values changed
    if (finalTrimStart !== initialTrimStart || finalTrimEnd !== initialTrimEnd) {
      trimItem.mutate(
        {
          storyId,
          itemId: trimmingItem,
          data: {
            trim_start_ms: finalTrimStart,
            trim_end_ms: finalTrimEnd,
          },
        },
        {
          onError: (error) => {
            toast({
              title: 'Failed to trim clip',
              description: error instanceof Error ? error.message : String(error),
              variant: 'destructive',
            });
          },
        },
      );
    }

    setTrimmingItem(null);
    setTrimSide(null);
    setTempTrimValues(null);
    trimStartItemRef.current = null;
  }, [trimmingItem, trimSide, tempTrimValues, storyId, trimItem, toast]);

  const handleSplit = useCallback(() => {
    if (!selectedClipId || splitItem.isPending) return;

    const item = items.find((i) => i.id === selectedClipId);
    if (!item) return;

    // currentTimeMs is driven by audio playback and arrives as a float;
    // the backend's StoryItemSplit.split_time_ms is `int`, so round before
    // sending or pydantic rejects the request.
    const splitTimeMs = Math.round(currentTimeMs - item.start_time_ms);
    const effectiveDuration = getEffectiveDuration(item);

    if (splitTimeMs <= 0 || splitTimeMs >= effectiveDuration) {
      toast({
        title: 'Invalid split point',
        description: 'Playhead must be within the selected clip',
        variant: 'destructive',
      });
      return;
    }

    splitItem.mutate(
      {
        storyId,
        itemId: selectedClipId,
        data: { split_time_ms: splitTimeMs },
      },
      {
        onSuccess: () => {
          setSelectedClipId(null);
        },
        onError: (error) => {
          toast({
            title: 'Failed to split clip',
            description: error instanceof Error ? error.message : String(error),
            variant: 'destructive',
          });
        },
      },
    );
  }, [
    selectedClipId,
    items,
    currentTimeMs,
    getEffectiveDuration,
    storyId,
    splitItem,
    toast,
    setSelectedClipId,
  ]);

  const handleDuplicate = useCallback(() => {
    if (!selectedClipId) return;

    duplicateItem.mutate(
      {
        storyId,
        itemId: selectedClipId,
      },
      {
        onError: (error) => {
          toast({
            title: 'Failed to duplicate clip',
            description: error instanceof Error ? error.message : String(error),
            variant: 'destructive',
          });
        },
      },
    );
  }, [selectedClipId, storyId, duplicateItem, toast]);

  const handleDelete = useCallback(() => {
    if (!selectedClipId) return;

    removeItem.mutate(
      {
        storyId,
        itemId: selectedClipId,
      },
      {
        onSuccess: () => {
          setSelectedClipId(null);
        },
        onError: (error) => {
          toast({
            title: 'Failed to delete clip',
            description: error instanceof Error ? error.message : String(error),
            variant: 'destructive',
          });
        },
      },
    );
  }, [selectedClipId, storyId, removeItem, toast, setSelectedClipId]);

  const handleRegenerate = useCallback(async () => {
    if (!selectedItem) return;
    try {
      await apiClient.regenerateGeneration(selectedItem.generation_id);
      addPendingGeneration(selectedItem.generation_id);
    } catch (error) {
      toast({
        title: 'Failed to regenerate',
        description: error instanceof Error ? error.message : String(error),
        variant: 'destructive',
      });
    }
  }, [selectedItem, addPendingGeneration, toast]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle shortcuts when editor is focused or no input is focused
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (e.key === ' ') {
        e.preventDefault();
        handlePlayPause();
      } else if (e.key === 'Escape') {
        setSelectedClipId(null);
      } else if (e.key === 's' || e.key === 'S') {
        if (selectedClipId) {
          e.preventDefault();
          handleSplit();
        }
      } else if (e.key === 'd' || e.key === 'D') {
        if (selectedClipId && (e.metaKey || e.ctrlKey)) {
          e.preventDefault();
          handleDuplicate();
        }
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedClipId) {
          e.preventDefault();
          handleDelete();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    selectedClipId,
    handleSplit,
    handleDuplicate,
    handleDelete,
    setSelectedClipId,
    handlePlayPause,
  ]);

  // Add global mouse listeners for trimming
  useEffect(() => {
    if (trimmingItem) {
      window.addEventListener('mousemove', handleTrimMove);
      window.addEventListener('mouseup', handleTrimEnd);
      return () => {
        window.removeEventListener('mousemove', handleTrimMove);
        window.removeEventListener('mouseup', handleTrimEnd);
      };
    }
  }, [trimmingItem, handleTrimMove, handleTrimEnd]);

  const handleDragStart = (e: React.MouseEvent, item: StoryItemDetail) => {
    e.stopPropagation();
    if (!tracksRef.current) return;

    const rect = e.currentTarget.getBoundingClientRect();
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
    setDragPosition({
      // Subtract label column width because clips live in a sub-container offset
      // by LABEL_COL_WIDTH, so dragPosition.x is stored in timeline-local coords.
      x:
        rect.left -
        tracksRef.current.getBoundingClientRect().left +
        tracksRef.current.scrollLeft -
        LABEL_COL_WIDTH,
      // Subtract ruler height since clips are positioned relative to tracks area, not the scrollable container
      y: rect.top - tracksRef.current.getBoundingClientRect().top - TIME_RULER_HEIGHT,
    });
    setDraggingItem(item.id);
  };

  const handleDragMove = useCallback(
    (e: React.MouseEvent) => {
      if (!draggingItem || !tracksRef.current) return;

      const rect = tracksRef.current.getBoundingClientRect();
      const x =
        e.clientX -
        rect.left +
        tracksRef.current.scrollLeft -
        dragOffset.x -
        LABEL_COL_WIDTH;
      // Subtract ruler height since clips are positioned relative to tracks area
      const y = e.clientY - rect.top - dragOffset.y - TIME_RULER_HEIGHT;

      setDragPosition({ x: Math.max(0, x), y });
    },
    [draggingItem, dragOffset],
  );

  const handleDragEnd = useCallback(() => {
    if (!draggingItem || !tracksRef.current) {
      setDraggingItem(null);
      return;
    }

    const item = items.find((i) => i.id === draggingItem);
    if (!item) {
      setDraggingItem(null);
      return;
    }

    // Calculate new time from x position
    const newTimeMs = Math.max(0, Math.round(pixelsToMs(dragPosition.x)));

    // Calculate new track from y position
    const trackIndex = Math.floor(dragPosition.y / TRACK_HEIGHT);
    const clampedTrackIndex = Math.max(0, Math.min(trackIndex, tracks.length - 1));
    const newTrack = tracks[clampedTrackIndex] ?? 0;

    // Check if position changed
    if (newTimeMs !== item.start_time_ms || newTrack !== item.track) {
      moveItem.mutate(
        {
          storyId,
          itemId: item.id,
          data: {
            start_time_ms: newTimeMs,
            track: newTrack,
          },
        },
        {
          onError: (error) => {
            toast({
              title: 'Failed to move item',
              description: error instanceof Error ? error.message : String(error),
              variant: 'destructive',
            });
          },
        },
      );
    }

    setDraggingItem(null);
  }, [draggingItem, dragPosition, items, tracks, pixelsToMs, storyId, moveItem, toast]);

  // Get track index for rendering
  const getTrackIndex = (trackNumber: number) => tracks.indexOf(trackNumber);

  // Calculate clip position and dimensions
  const getClipStyle = (item: StoryItemDetail) => {
    const isDragging = draggingItem === item.id;
    const trackIndex = getTrackIndex(item.track);
    const effectiveDuration = getEffectiveDuration(item);
    const width = msToPixels(effectiveDuration);
    const left = isDragging ? dragPosition.x : msToPixels(item.start_time_ms);
    const top = isDragging ? dragPosition.y : trackIndex * TRACK_HEIGHT;

    return {
      width: `${width}px`,
      left: `${left}px`,
      top: `${top}px`,
      height: `${TRACK_HEIGHT - 4}px`,
    };
  };

  // Playhead position
  const playheadLeft = msToPixels(currentTimeMs);

  // Auto-scroll timeline to follow playhead during playback
  useEffect(() => {
    if (!isCurrentlyPlaying || !tracksRef.current) return;

    const container = tracksRef.current;
    const containerWidth = container.clientWidth;
    const scrollLeft = container.scrollLeft;
    const halfwayPoint = scrollLeft + containerWidth / 2;

    // If playhead is past the halfway point, scroll to keep it centered
    if (playheadLeft > halfwayPoint) {
      const targetScroll = playheadLeft - containerWidth / 2;
      container.scrollLeft = targetScroll;
    }
  }, [isCurrentlyPlaying, playheadLeft]);

  // Calculate tracks area height
  const tracksAreaHeight = tracks.length * TRACK_HEIGHT;
  const timelineContainerHeight = editorHeight - 40 - SCRUB_BAR_HEIGHT;

  // Scrollbar thumb geometry
  const maxTimelineScroll = Math.max(0, timelineWidth - containerWidth);
  const visibleRatio = timelineWidth > 0 ? Math.min(1, containerWidth / timelineWidth) : 1;
  const thumbWidth = Math.max(24, visibleRatio * scrollbarTrackWidth);
  const thumbRange = Math.max(0, scrollbarTrackWidth - thumbWidth);
  const thumbLeft =
    maxTimelineScroll > 0 && thumbRange > 0
      ? (timelineScrollLeft / maxTimelineScroll) * thumbRange
      : 0;
  const canScrollHorizontally = maxTimelineScroll > 0;

  const handleScrollbarMouseDown = useCallback(
    (mode: 'pan' | 'left' | 'right') => (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      scrollbarDragRef.current = {
        mode,
        startX: e.clientX,
        startScrollLeft: timelineScrollLeft,
        startPixelsPerSecond: pixelsPerSecond,
      };
    },
    [timelineScrollLeft, pixelsPerSecond],
  );

  // After a zoom drag updates pixelsPerSecond, snap scrollLeft so the anchored
  // edge (left or right of the visible window) stays at the same time.
  useEffect(() => {
    const anchor = zoomAnchorRef.current;
    if (!anchor || !tracksRef.current) return;
    const timePx = (anchor.timeMs / 1000) * pixelsPerSecond;
    tracksRef.current.scrollLeft =
      anchor.type === 'left' ? Math.max(0, timePx) : Math.max(0, timePx - containerWidth);
  }, [pixelsPerSecond, containerWidth]);

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      const drag = scrollbarDragRef.current;
      if (!drag || !tracksRef.current) return;
      const deltaX = e.clientX - drag.startX;

      if (drag.mode === 'pan') {
        if (thumbRange <= 0) return;
        const deltaScroll = (deltaX / thumbRange) * maxTimelineScroll;
        tracksRef.current.scrollLeft = Math.max(
          0,
          Math.min(maxTimelineScroll, drag.startScrollLeft + deltaScroll),
        );
        return;
      }

      if (scrollbarTrackWidth <= 0 || containerWidth <= 0) return;

      // Recompute the thumb width that corresponded to the drag start, then
      // apply the mouse delta to the dragged edge.
      const startTimelinePx =
        (totalDurationMs / 1000) * drag.startPixelsPerSecond + 200;
      const startThumbWidth = Math.max(
        30,
        Math.min(scrollbarTrackWidth, (containerWidth / startTimelinePx) * scrollbarTrackWidth),
      );
      const newThumbWidth = Math.max(
        30,
        Math.min(
          scrollbarTrackWidth,
          drag.mode === 'right' ? startThumbWidth + deltaX : startThumbWidth - deltaX,
        ),
      );

      const newTimelinePx = (containerWidth / newThumbWidth) * scrollbarTrackWidth;
      const rawPps = (newTimelinePx - 200) / (totalDurationMs / 1000);
      const newPps = Math.max(minPps, Math.min(maxPps, rawPps));

      zoomAnchorRef.current =
        drag.mode === 'right'
          ? {
              type: 'left',
              timeMs: (drag.startScrollLeft / drag.startPixelsPerSecond) * 1000,
            }
          : {
              type: 'right',
              timeMs:
                ((drag.startScrollLeft + containerWidth) / drag.startPixelsPerSecond) * 1000,
            };

      setPixelsPerSecond(newPps);
    };
    const onMouseUp = () => {
      scrollbarDragRef.current = null;
      zoomAnchorRef.current = null;
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [maxTimelineScroll, thumbRange, scrollbarTrackWidth, containerWidth, totalDurationMs, minPps, maxPps]);

  if (items.length === 0) {
    return null;
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 border-t bg-background/95 backdrop-blur supports-backdrop-filter:bg-background/60 z-50">
      <div
        className="border-t bg-background/30 backdrop-blur-2xl overflow-hidden relative"
        ref={containerRef}
      >
        {/* Resize handle at top */}
        <button
          type="button"
          className="absolute top-0 left-0 right-0 h-2 cursor-ns-resize flex items-center justify-center hover:bg-muted/50 transition-colors z-20 group"
          onMouseDown={handleResizeStart}
          aria-label="Resize track editor"
        >
          <GripHorizontal className="h-3 w-3 text-muted-foreground/50 group-hover:text-muted-foreground" />
        </button>

        {/* Toolbar */}
        <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30 mt-2">
          {/* Play controls - left side */}
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={handlePlayPause}
              title="Play/Pause (Space)"
              aria-label={isCurrentlyPlaying ? 'Pause' : 'Play'}
            >
              {isCurrentlyPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={handleStop}
              disabled={!isCurrentlyPlaying}
              aria-label="Stop"
            >
              <Square className="h-3 w-3" />
            </Button>
            <span className="text-xs text-muted-foreground tabular-nums ml-2">
              {formatTime(currentTimeMs)} / {formatTime(totalDurationMs)}
            </span>
          </div>

          {/* Clip editing controls - center */}
          {selectedClipId && (
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={handleSplit}
                title="Split at playhead (S)"
                aria-label="Split at playhead"
              >
                <Scissors className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={handleDuplicate}
                title="Duplicate (Cmd/Ctrl+D)"
                aria-label="Duplicate clip"
              >
                <Copy className="h-4 w-4" />
              </Button>
              {selectedItem && (
                <ClipVolumePopover
                  storyId={storyId}
                  itemId={selectedItem.id}
                  volume={selectedItem.volume}
                  onChange={(value) =>
                    updateVolume.mutate(
                      {
                        storyId,
                        itemId: selectedItem.id,
                        data: { volume: value },
                      },
                      {
                        onError: (error) => {
                          toast({
                            title: 'Failed to update volume',
                            description: error instanceof Error ? error.message : String(error),
                            variant: 'destructive',
                          });
                        },
                      },
                    )
                  }
                />
              )}
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={handleDelete}
                title="Delete (Delete/Backspace)"
                aria-label="Delete clip"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
              {selectedItem?.engine !== 'import' && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7"
                  onClick={handleRegenerate}
                  title="Regenerate"
                  aria-label="Regenerate clip"
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
              )}
              {hasMultipleVersions && (
                <>
                  <div className="w-px h-4 bg-border mx-1" />
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        className="h-7 gap-1.5 px-2 text-xs"
                        title="Change version/take"
                      >
                        <GalleryVerticalEnd className="h-3.5 w-3.5" />
                        <span className="max-w-[80px] truncate">
                          {activeVersionLabel ?? 'default'}
                        </span>
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="center" className="min-w-[160px]">
                      {selectedItemVersions.map((version) => {
                        const isActive = selectedItem?.version_id
                          ? version.id === selectedItem.version_id
                          : version.is_default;
                        return (
                          <DropdownMenuItem
                            key={version.id}
                            onClick={() => handleSetVersion(version.id)}
                            className="gap-2 text-xs"
                          >
                            <Check
                              className={cn('h-3 w-3', isActive ? 'opacity-100' : 'opacity-0')}
                            />
                            <span className="truncate">{version.label}</span>
                            {version.effects_chain && version.effects_chain.length > 0 && (
                              <span className="text-muted-foreground ml-auto text-[10px]">
                                {version.effects_chain.length} fx
                              </span>
                            )}
                          </DropdownMenuItem>
                        );
                      })}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </>
              )}
            </div>
          )}

          {/* Zoom controls - right side */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Zoom:</span>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={handleZoomOut}
              aria-label="Zoom out"
            >
              <Minus className="h-3 w-3" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={handleZoomIn}
              aria-label="Zoom in"
            >
              <Plus className="h-3 w-3" />
            </Button>
          </div>
        </div>

        {/* Timeline scroll container */}
        {/* biome-ignore lint/a11y/noStaticElementInteractions: Container handles drag events for child clips */}
        <div
          ref={tracksRef}
          className="overflow-auto relative"
          style={{ height: `${timelineContainerHeight}px` }}
          onMouseMove={draggingItem ? handleDragMove : undefined}
          onMouseUp={draggingItem ? handleDragEnd : undefined}
          onMouseLeave={draggingItem ? handleDragEnd : undefined}
        >
          {/* Ruler row: corner spacer + time ruler, sticky to top */}
          <div
            className="flex sticky top-0 z-30"
            style={{ width: `${timelineWidth + LABEL_COL_WIDTH}px` }}
          >
            <div className="w-16 h-6 shrink-0 border-b border-r bg-muted/30 sticky left-0 z-40" />
            <button
              type="button"
              className="h-6 border-b bg-muted/20 cursor-pointer text-left relative"
              style={{ width: `${timelineWidth}px` }}
              onClick={handleTimelineClick}
              aria-label="Seek timeline"
            >
              {timeMarkers.map((ms) => (
                <div
                  key={ms}
                  className="absolute top-0 h-full flex flex-col justify-end pointer-events-none"
                  style={{ left: `${msToPixels(ms)}px` }}
                >
                  <div className="h-2 w-px bg-border" />
                  <span className="text-[10px] text-muted-foreground ml-1 select-none">
                    {formatTime(ms)}
                  </span>
                </div>
              ))}
            </button>
          </div>

          {/* Tracks area (rows with sticky labels + clips sub-container) */}
          <div
            className="relative"
            style={{
              width: `${timelineWidth + LABEL_COL_WIDTH}px`,
              height: `${tracksAreaHeight}px`,
            }}
          >
            {/* Per-track rows: label and background as flex siblings guarantee alignment */}
            {tracks.map((trackNumber, index) => {
              const isFirst = index === 0;
              const isLast = index === tracks.length - 1;
              return (
                <div
                  key={trackNumber}
                  className="absolute left-0 right-0 flex"
                  style={{
                    top: `${index * TRACK_HEIGHT}px`,
                    height: `${TRACK_HEIGHT}px`,
                  }}
                >
                  <div className="w-16 shrink-0 border-b border-r flex items-center justify-center sticky left-0 z-20 h-full bg-background">
                    <div className="absolute inset-0 bg-muted/20 pointer-events-none" />
                    <span className="relative text-[10px] text-muted-foreground select-none">
                      {trackNumber}
                    </span>
                    {isFirst && (
                      <button
                        type="button"
                        onClick={handleAddTrackAbove}
                        title="Add track above"
                        aria-label="Add track above"
                        className="absolute top-0 right-0 left-0 h-3 flex items-center justify-center text-muted-foreground/50 hover:text-foreground hover:bg-muted/40 transition-colors"
                      >
                        <Plus className="h-2.5 w-2.5" />
                      </button>
                    )}
                    {isLast && (
                      <button
                        type="button"
                        onClick={handleAddTrackBelow}
                        title="Add track below"
                        aria-label="Add track below"
                        className="absolute bottom-0 right-0 left-0 h-3 flex items-center justify-center text-muted-foreground/50 hover:text-foreground hover:bg-muted/40 transition-colors"
                      >
                        <Plus className="h-2.5 w-2.5" />
                      </button>
                    )}
                  </div>
                  <div
                    className={cn(
                      'border-b flex-1 pointer-events-none',
                      index % 2 === 0 ? 'bg-background' : 'bg-muted/10',
                    )}
                  />
                </div>
              );
            })}

            {/* Clip/playhead/seek layer offset past the label column */}
            <div
              className="absolute top-0 bottom-0"
              style={{ left: `${LABEL_COL_WIDTH}px`, width: `${timelineWidth}px` }}
            >
              {/* Click area for seeking - z-index lower than clips */}
              <button
                type="button"
                className="absolute inset-0 z-0 cursor-pointer"
                onClick={handleTimelineClick}
                aria-label="Seek timeline"
              />

              {/* Audio clips */}
              {items.map((item) => {
                const isDragging = draggingItem === item.id;
                const isSelected = selectedClipId === item.id;
                const isTrimming = trimmingItem === item.id;

                // Use temporary trim values during trimming for visual feedback
                const displayTrimStart =
                  isTrimming && tempTrimValues
                    ? tempTrimValues.trim_start_ms
                    : item.trim_start_ms || 0;
                const displayTrimEnd =
                  isTrimming && tempTrimValues ? tempTrimValues.trim_end_ms : item.trim_end_ms || 0;
                const effectiveDuration = item.duration * 1000 - displayTrimStart - displayTrimEnd;

                const style = getClipStyle({
                  ...item,
                  trim_start_ms: displayTrimStart,
                  trim_end_ms: displayTrimEnd,
                });
                const clipWidth = msToPixels(effectiveDuration);

                return (
                  <div
                    key={item.id}
                    className={cn(
                      'absolute rounded select-none overflow-visible z-10',
                      isSelected && 'ring-2 ring-primary ring-offset-1',
                      isTrimming && 'ring-2 ring-accent',
                    )}
                    style={style}
                  >
                    <button
                      type="button"
                      className={cn(
                        'w-full h-full rounded cursor-move overflow-hidden',
                        'bg-accent/80 hover:bg-accent border border-accent-foreground/20',
                        'flex flex-col justify-center',
                        isDragging && 'opacity-80 shadow-lg z-20',
                        !isDragging && 'transition-all duration-100',
                      )}
                      onClick={(e) => handleClipClick(e, item)}
                      onMouseDown={(e) => {
                        // Only start drag if not clicking on trim handles
                        if (!(e.target as HTMLElement).closest('.trim-handle')) {
                          handleDragStart(e, item);
                        }
                      }}
                    >
                      {/* Clip label */}
                      <div className="absolute top-0 left-1 right-1 z-10">
                        <p className="text-[9px] font-medium text-accent-foreground truncate">
                          {item.engine === 'import' ? item.text : item.profile_name}
                        </p>
                      </div>
                      {/* Waveform */}
                      <div className="absolute inset-0 top-3">
                        <ClipWaveform
                          generationId={item.generation_id}
                          versionId={item.version_id}
                          width={clipWidth}
                          trimStartMs={displayTrimStart}
                          trimEndMs={displayTrimEnd}
                          duration={item.duration}
                        />
                      </div>
                    </button>

                    {/* Trim handles */}
                    {isSelected && (
                      <>
                        {/* Left trim handle */}
                        <button
                          type="button"
                          className="trim-handle absolute left-0 top-0 bottom-0 w-2 cursor-ew-resize hover:bg-primary/30 bg-primary/20 z-30 rounded-l"
                          onMouseDown={(e) => handleTrimStart(e, item, 'start')}
                          aria-label="Trim start"
                        />
                        {/* Right trim handle */}
                        <button
                          type="button"
                          className="trim-handle absolute right-0 top-0 bottom-0 w-2 cursor-ew-resize hover:bg-primary/30 bg-primary/20 z-30 rounded-r"
                          onMouseDown={(e) => handleTrimStart(e, item, 'end')}
                          aria-label="Trim end"
                        />
                      </>
                    )}
                  </div>
                );
              })}

              {/* Playhead - always visible */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-accent z-30 pointer-events-none rounded-full"
                style={{ left: `${playheadLeft}px` }}
              >
                <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-3 h-3 bg-accent rounded-full" />
              </div>
            </div>
          </div>
        </div>

        {/* Horizontal timeline scrollbar + zoom handles */}
        <div
          className="flex border-t bg-background/40"
          style={{ height: `${SCRUB_BAR_HEIGHT}px` }}
        >
          <div className="w-16 shrink-0 border-r" />
          <div
            ref={scrollbarTrackRef}
            className="relative flex-1 overflow-hidden select-none px-1"
          >
            <div
              className="absolute top-1 bottom-1 bg-foreground/10 hover:bg-foreground/15 transition-colors group rounded-full"
              style={{ width: `${thumbWidth}px`, left: `${thumbLeft}px` }}
            >
              {/* Left zoom handle */}
              {/* biome-ignore lint/a11y/noStaticElementInteractions: mouse-driven edge handle */}
              <div
                role="slider"
                aria-label="Zoom from left edge"
                aria-valuenow={Math.round(pixelsPerSecond)}
                aria-valuemin={Math.round(minPps)}
                aria-valuemax={Math.round(maxPps)}
                className="absolute top-0 bottom-0 left-0 w-1.5 cursor-ew-resize bg-foreground/25 hover:bg-foreground/40 transition-colors rounded-l-full"
                onMouseDown={handleScrollbarMouseDown('left')}
              />
              {/* Pan area */}
              {/* biome-ignore lint/a11y/noStaticElementInteractions: mouse-driven drag area */}
              <div
                className={cn(
                  'absolute top-0 bottom-0 left-1.5 right-1.5',
                  canScrollHorizontally ? 'cursor-grab active:cursor-grabbing' : 'cursor-default',
                )}
                onMouseDown={canScrollHorizontally ? handleScrollbarMouseDown('pan') : undefined}
              />
              {/* Right zoom handle */}
              {/* biome-ignore lint/a11y/noStaticElementInteractions: mouse-driven edge handle */}
              <div
                role="slider"
                aria-label="Zoom from right edge"
                aria-valuenow={Math.round(pixelsPerSecond)}
                aria-valuemin={Math.round(minPps)}
                aria-valuemax={Math.round(maxPps)}
                className="absolute top-0 bottom-0 right-0 w-1.5 cursor-ew-resize bg-foreground/25 hover:bg-foreground/40 transition-colors rounded-r-full"
                onMouseDown={handleScrollbarMouseDown('right')}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
