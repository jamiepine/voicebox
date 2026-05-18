import {
  Copy,
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
import type { KeyboardEvent, MouseEvent, ReactNode } from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Slider } from '@/components/ui/slider';
import { cn } from '@/lib/utils/cn';
import { ClipWaveform } from './ClipWaveform';
import { TimelineScrollbar } from './TimelineScrollbar';

export interface AudioTrackClip {
  id: string;
  startMs: number;
  durationMs: number;
  track: number;
  label: string;
  sublabel?: string;
  audioUrl?: string;
  trimStartMs?: number;
  trimEndMs?: number;
  volume?: number;
  variant?: 'primary' | 'accent' | 'warning' | 'success' | 'info' | 'reference';
  canRegenerate?: boolean;
  editable?: boolean;
  movable?: boolean;
  trimmable?: boolean;
}

interface AudioTrackEditorProps {
  clips: AudioTrackClip[];
  selectedClipId: string | null;
  currentTimeMs: number;
  isPlaying: boolean;
  height: number;
  onHeightChange: (height: number) => void;
  onSelectClip: (clipId: string | null) => void;
  onSeek: (timeMs: number) => void;
  onPreviewSeek?: (timeMs: number) => void;
  onPlayPause: () => void;
  onStop: () => void;
  onMoveClip: (clipId: string, startMs: number, track: number) => void;
  onTrimClip: (clipId: string, trimStartMs: number, trimEndMs: number) => void;
  onSplitClip?: (clipId: string, splitTimeMs: number) => void;
  onDuplicateClip?: (clipId: string) => void;
  onDeleteClip?: (clipId: string) => void;
  onRegenerateClip?: (clipId: string) => void;
  onVolumeChange?: (clipId: string, volume: number) => void;
  timelineControls?: ReactNode;
  toolbarExtra?: ReactNode;
}

const TRACK_HEIGHT = 48;
const TIME_RULER_HEIGHT = 24;
const SCRUB_BAR_HEIGHT = 16;
const LABEL_COL_WIDTH = 64;
const MIN_VISIBLE_SECONDS = 10;
const DEFAULT_VISIBLE_SECONDS = 60;
const FALLBACK_PIXELS_PER_SECOND = 50;
const DEFAULT_TRACKS = [1, 0, -1];
const MIN_EDITOR_HEIGHT = 120;
const MAX_EDITOR_HEIGHT = 500;

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function getClipClasses(variant: AudioTrackClip['variant'], isSelected: boolean) {
  const base = 'border text-left shadow-sm';
  if (isSelected) return cn(base, 'border-accent bg-accent text-accent-foreground');
  if (variant === 'warning') return cn(base, 'border-amber-500/50 bg-amber-300 text-amber-950');
  if (variant === 'success') return cn(base, 'border-emerald-500/50 bg-emerald-500/80 text-white');
  if (variant === 'info') return cn(base, 'border-sky-500/40 bg-sky-500/80 text-white');
  if (variant === 'reference') return cn(base, 'border-border bg-background/80 text-muted-foreground');
  return cn(base, 'border-primary/30 bg-primary/70 text-primary-foreground');
}

function ClipVolumeButton({
  volume,
  onChange,
}: {
  volume: number;
  onChange: (value: number) => void;
}) {
  const [localVolume, setLocalVolume] = useState(volume);

  useEffect(() => {
    setLocalVolume(volume);
  }, [volume]);

  const display = Math.round(localVolume * 100);
  const Icon = localVolume === 0 ? VolumeX : Volume2;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          title={`Volume - ${display}%`}
          aria-label="Adjust clip volume"
        >
          <Icon className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="center" className="w-56 p-3">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Volume</span>
          <span className="text-xs tabular-nums">{display}%</span>
        </div>
        <Slider
          value={[localVolume * 100]}
          min={0}
          max={200}
          step={1}
          onValueChange={([value]) => setLocalVolume((value ?? 100) / 100)}
          onValueCommit={([value]) => onChange((value ?? 100) / 100)}
          aria-label="Clip volume"
        />
      </PopoverContent>
    </Popover>
  );
}

export function AudioTrackEditor({
  clips,
  selectedClipId,
  currentTimeMs,
  isPlaying,
  height,
  onHeightChange,
  onSelectClip,
  onSeek,
  onPreviewSeek,
  onPlayPause,
  onStop,
  onMoveClip,
  onTrimClip,
  onSplitClip,
  onDuplicateClip,
  onDeleteClip,
  onRegenerateClip,
  onVolumeChange,
  timelineControls,
  toolbarExtra,
}: AudioTrackEditorProps) {
  const [pixelsPerSecond, setPixelsPerSecond] = useState(FALLBACK_PIXELS_PER_SECOND);
  const hasAppliedDefaultZoomRef = useRef(false);
  const [containerWidth, setContainerWidth] = useState(0);
  const [timelineScrollLeft, setTimelineScrollLeft] = useState(0);
  const [scrollbarTrackWidth, setScrollbarTrackWidth] = useState(0);
  const [extraTracks, setExtraTracks] = useState<number[]>([]);
  const [isResizing, setIsResizing] = useState(false);
  const [draggingClipId, setDraggingClipId] = useState<string | null>(null);
  const [isDraggingPlayhead, setIsDraggingPlayhead] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [dragPosition, setDragPosition] = useState({ x: 0, y: 0 });
  const [trimmingClipId, setTrimmingClipId] = useState<string | null>(null);
  const [trimSide, setTrimSide] = useState<'start' | 'end' | null>(null);
  const [trimStartX, setTrimStartX] = useState(0);
  const [tempTrimValues, setTempTrimValues] = useState<{
    trimStartMs: number;
    trimEndMs: number;
  } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const tracksRef = useRef<HTMLDivElement>(null);
  const scrollbarTrackRef = useRef<HTMLDivElement>(null);
  const resizeStartY = useRef(0);
  const resizeStartHeight = useRef(0);
  const trimStartClipRef = useRef<{
    clip: AudioTrackClip;
    initialTrimStart: number;
    initialTrimEnd: number;
  } | null>(null);
  const scrollbarDragRef = useRef<{
    mode: 'pan' | 'left' | 'right';
    startX: number;
    startScrollLeft: number;
    startPixelsPerSecond: number;
  } | null>(null);
  const zoomAnchorRef = useRef<{ type: 'left' | 'right'; timeMs: number } | null>(null);

  const selectedClip = useMemo(
    () => clips.find((clip) => clip.id === selectedClipId),
    [clips, selectedClipId],
  );

  const tracks = useMemo(() => {
    const trackSet = new Set([...DEFAULT_TRACKS, ...clips.map((clip) => clip.track), ...extraTracks]);
    return Array.from(trackSet).sort((a, b) => b - a);
  }, [clips, extraTracks]);

  const getEffectiveDuration = useCallback((clip: AudioTrackClip) => {
    return clip.durationMs - (clip.trimStartMs ?? 0) - (clip.trimEndMs ?? 0);
  }, []);

  const totalDurationMs = useMemo(() => {
    if (clips.length === 0) return 10000;
    return Math.max(...clips.map((clip) => clip.startMs + getEffectiveDuration(clip)), 10000);
  }, [clips, getEffectiveDuration]);

  const visibleTrackWidth = Math.max(0, containerWidth - LABEL_COL_WIDTH);
  const projectSeconds = totalDurationMs / 1000;
  const { minPps, maxPps } = useMemo(() => {
    if (visibleTrackWidth <= 0 || projectSeconds <= 0) return { minPps: 10, maxPps: 200 };
    const min = visibleTrackWidth / projectSeconds;
    const max = visibleTrackWidth / MIN_VISIBLE_SECONDS;
    return { minPps: min, maxPps: Math.max(max, min) };
  }, [visibleTrackWidth, projectSeconds]);

  useEffect(() => {
    if (hasAppliedDefaultZoomRef.current || visibleTrackWidth <= 0) return;
    const defaultScope = Math.min(DEFAULT_VISIBLE_SECONDS, Math.max(projectSeconds, MIN_VISIBLE_SECONDS));
    setPixelsPerSecond(visibleTrackWidth / defaultScope);
    hasAppliedDefaultZoomRef.current = true;
  }, [visibleTrackWidth, projectSeconds]);

  useEffect(() => {
    setPixelsPerSecond((prev) => Math.max(minPps, Math.min(maxPps, prev)));
  }, [minPps, maxPps]);

  const contentWidth = (totalDurationMs / 1000) * pixelsPerSecond + 200;
  const timelineWidth = Math.max(contentWidth, containerWidth);
  const tracksAreaHeight = tracks.length * TRACK_HEIGHT;
  const timelineContainerHeight = height - 40 - SCRUB_BAR_HEIGHT;
  const maxTimelineScroll = Math.max(0, timelineWidth - containerWidth);
  const visibleRatio = timelineWidth > 0 ? Math.min(1, containerWidth / timelineWidth) : 1;
  const thumbWidth = Math.max(24, visibleRatio * scrollbarTrackWidth);
  const thumbRange = Math.max(0, scrollbarTrackWidth - thumbWidth);
  const thumbLeft =
    maxTimelineScroll > 0 && thumbRange > 0
      ? (timelineScrollLeft / maxTimelineScroll) * thumbRange
      : 0;
  const canScrollHorizontally = maxTimelineScroll > 0;

  const timeMarkers = useMemo(() => {
    const markers: number[] = [];
    let intervalMs = 5000;
    if (pixelsPerSecond > 100) intervalMs = 1000;
    else if (pixelsPerSecond > 50) intervalMs = 2000;
    else if (pixelsPerSecond < 20) intervalMs = 10000;
    for (let ms = 0; ms <= totalDurationMs + intervalMs; ms += intervalMs) {
      markers.push(ms);
    }
    return markers;
  }, [totalDurationMs, pixelsPerSecond]);

  const msToPixels = useCallback((ms: number) => (ms / 1000) * pixelsPerSecond, [pixelsPerSecond]);
  const pixelsToMs = useCallback((px: number) => (px / pixelsPerSecond) * 1000, [pixelsPerSecond]);

  useEffect(() => {
    const container = tracksRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) setContainerWidth(entry.contentRect.width);
    });
    observer.observe(container);
    setContainerWidth(container.clientWidth);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const el = tracksRef.current;
    if (!el) return;
    const onScroll = () => setTimelineScrollLeft(el.scrollLeft);
    el.addEventListener('scroll', onScroll);
    setTimelineScrollLeft(el.scrollLeft);
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    const el = scrollbarTrackRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) setScrollbarTrackWidth(entry.contentRect.width);
    });
    observer.observe(el);
    setScrollbarTrackWidth(el.clientWidth);
    return () => observer.disconnect();
  }, []);

  const handleZoomIn = () => setPixelsPerSecond((prev) => Math.min(prev * 1.5, maxPps));
  const handleZoomOut = () => setPixelsPerSecond((prev) => Math.max(prev / 1.5, minPps));

  const handleResizeStart = useCallback(
    (event: MouseEvent) => {
      event.preventDefault();
      setIsResizing(true);
      resizeStartY.current = event.clientY;
      resizeStartHeight.current = height;
    },
    [height],
  );

  useEffect(() => {
    if (!isResizing) return;
    const handleMove = (event: globalThis.MouseEvent) => {
      const deltaY = resizeStartY.current - event.clientY;
      const nextHeight = Math.min(
        MAX_EDITOR_HEIGHT,
        Math.max(MIN_EDITOR_HEIGHT, resizeStartHeight.current + deltaY),
      );
      onHeightChange(nextHeight);
    };
    const handleUp = () => setIsResizing(false);
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, [isResizing, onHeightChange]);

  const handleTimelineClick = (event: MouseEvent<HTMLElement>) => {
    if (!tracksRef.current || draggingClipId || trimmingClipId) return;
    const rect = tracksRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left + tracksRef.current.scrollLeft - LABEL_COL_WIDTH;
    onSeek(Math.max(0, pixelsToMs(x)));
    onSelectClip(null);
  };

  const handlePlayheadMouseDown = (event: MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const timelineLayer = event.currentTarget.parentElement;
    const scroller = tracksRef.current;
    if (!timelineLayer || !scroller) return;

    setIsDraggingPlayhead(true);
    const rect = timelineLayer.getBoundingClientRect();
    const timeFromClientX = (clientX: number) => {
      const x = clientX - rect.left + scroller.scrollLeft;
      return Math.max(0, Math.round(pixelsToMs(x)));
    };

    const handleMove = (moveEvent: globalThis.MouseEvent) => {
      const timeMs = timeFromClientX(moveEvent.clientX);
      if (onPreviewSeek) onPreviewSeek(timeMs);
      else onSeek(timeMs);
    };

    const handleUp = (upEvent: globalThis.MouseEvent) => {
      onSeek(timeFromClientX(upEvent.clientX));
      setIsDraggingPlayhead(false);
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };

    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp, { once: true });
  };

  const handlePlayheadKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    let nextTimeMs: number | null = null;
    const smallStepMs = 100;
    const largeStepMs = 1000;

    switch (event.key) {
      case 'ArrowLeft':
        nextTimeMs = currentTimeMs - smallStepMs;
        break;
      case 'ArrowRight':
        nextTimeMs = currentTimeMs + smallStepMs;
        break;
      case 'PageDown':
        nextTimeMs = currentTimeMs - largeStepMs;
        break;
      case 'PageUp':
        nextTimeMs = currentTimeMs + largeStepMs;
        break;
      case 'Home':
        nextTimeMs = 0;
        break;
      case 'End':
        nextTimeMs = totalDurationMs;
        break;
      default:
        return;
    }

    event.preventDefault();
    onSeek(Math.max(0, Math.min(totalDurationMs, Math.round(nextTimeMs))));
  };

  const handleTrimStart = (event: MouseEvent, clip: AudioTrackClip, side: 'start' | 'end') => {
    event.stopPropagation();
    setTrimmingClipId(clip.id);
    setTrimSide(side);
    onSelectClip(clip.id);
    setTrimStartX(event.clientX);
    trimStartClipRef.current = {
      clip,
      initialTrimStart: clip.trimStartMs ?? 0,
      initialTrimEnd: clip.trimEndMs ?? 0,
    };
  };

  const handleTrimMove = useCallback(
    (event: globalThis.MouseEvent) => {
      if (!trimmingClipId || !trimSide || !trimStartClipRef.current) return;
      const deltaMs = pixelsToMs(event.clientX - trimStartX);
      const { clip, initialTrimStart, initialTrimEnd } = trimStartClipRef.current;
      let trimStart = initialTrimStart;
      let trimEnd = initialTrimEnd;
      if (trimSide === 'start') {
        trimStart = Math.round(Math.max(0, Math.min(initialTrimStart + deltaMs, clip.durationMs - initialTrimEnd - 100)));
      } else {
        trimEnd = Math.round(Math.max(0, Math.min(initialTrimEnd - deltaMs, clip.durationMs - initialTrimStart - 100)));
      }
      if (trimStart + trimEnd >= clip.durationMs - 100) return;
      setTempTrimValues({ trimStartMs: trimStart, trimEndMs: trimEnd });
    },
    [pixelsToMs, trimSide, trimStartX, trimmingClipId],
  );

  const handleTrimEnd = useCallback(() => {
    if (!trimmingClipId || !trimSide || !trimStartClipRef.current) {
      setTrimmingClipId(null);
      setTrimSide(null);
      setTempTrimValues(null);
      trimStartClipRef.current = null;
      return;
    }
    const { initialTrimStart, initialTrimEnd } = trimStartClipRef.current;
    const finalTrimStart = Math.round(tempTrimValues?.trimStartMs ?? initialTrimStart);
    const finalTrimEnd = Math.round(tempTrimValues?.trimEndMs ?? initialTrimEnd);
    if (finalTrimStart !== initialTrimStart || finalTrimEnd !== initialTrimEnd) {
      onTrimClip(trimmingClipId, finalTrimStart, finalTrimEnd);
    }
    setTrimmingClipId(null);
    setTrimSide(null);
    setTempTrimValues(null);
    trimStartClipRef.current = null;
  }, [onTrimClip, tempTrimValues, trimSide, trimmingClipId]);

  useEffect(() => {
    if (!trimmingClipId) return;
    window.addEventListener('mousemove', handleTrimMove);
    window.addEventListener('mouseup', handleTrimEnd);
    return () => {
      window.removeEventListener('mousemove', handleTrimMove);
      window.removeEventListener('mouseup', handleTrimEnd);
    };
  }, [handleTrimEnd, handleTrimMove, trimmingClipId]);

  const handleDragStart = (event: MouseEvent, clip: AudioTrackClip) => {
    event.stopPropagation();
    if (!tracksRef.current) return;
    const rect = event.currentTarget.getBoundingClientRect();
    setDragOffset({ x: event.clientX - rect.left, y: event.clientY - rect.top });
    setDragPosition({
      x: rect.left - tracksRef.current.getBoundingClientRect().left + tracksRef.current.scrollLeft - LABEL_COL_WIDTH,
      y: rect.top - tracksRef.current.getBoundingClientRect().top - TIME_RULER_HEIGHT,
    });
    setDraggingClipId(clip.id);
  };

  const handleDragMove = useCallback(
    (event: MouseEvent) => {
      if (!draggingClipId || !tracksRef.current) return;
      const rect = tracksRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left + tracksRef.current.scrollLeft - dragOffset.x - LABEL_COL_WIDTH;
      const y = event.clientY - rect.top - dragOffset.y - TIME_RULER_HEIGHT;
      setDragPosition({ x: Math.max(0, x), y });
    },
    [dragOffset, draggingClipId],
  );

  const handleDragEnd = useCallback(() => {
    if (!draggingClipId) return;
    const clip = clips.find((item) => item.id === draggingClipId);
    if (!clip) {
      setDraggingClipId(null);
      return;
    }
    const nextStartMs = Math.max(0, Math.round(pixelsToMs(dragPosition.x)));
    const trackIndex = Math.floor(dragPosition.y / TRACK_HEIGHT);
    const nextTrack = tracks[Math.max(0, Math.min(trackIndex, tracks.length - 1))] ?? 0;
    if (nextStartMs !== clip.startMs || nextTrack !== clip.track) {
      onMoveClip(clip.id, nextStartMs, nextTrack);
    }
    setDraggingClipId(null);
  }, [clips, dragPosition, draggingClipId, onMoveClip, pixelsToMs, tracks]);

  const handleSplit = () => {
    if (!selectedClip || !onSplitClip) return;
    onSplitClip(selectedClip.id, Math.round(currentTimeMs - selectedClip.startMs));
  };

  const handleScrollbarMouseDown = useCallback(
    (mode: 'pan' | 'left' | 'right') => (event: MouseEvent) => {
      event.preventDefault();
      event.stopPropagation();
      scrollbarDragRef.current = {
        mode,
        startX: event.clientX,
        startScrollLeft: timelineScrollLeft,
        startPixelsPerSecond: pixelsPerSecond,
      };
    },
    [pixelsPerSecond, timelineScrollLeft],
  );

  useEffect(() => {
    const anchor = zoomAnchorRef.current;
    if (!anchor || !tracksRef.current) return;
    const timePx = (anchor.timeMs / 1000) * pixelsPerSecond;
    tracksRef.current.scrollLeft =
      anchor.type === 'left' ? Math.max(0, timePx) : Math.max(0, timePx - containerWidth);
  }, [containerWidth, pixelsPerSecond]);

  useEffect(() => {
    const handleMove = (event: globalThis.MouseEvent) => {
      const drag = scrollbarDragRef.current;
      if (!drag || !tracksRef.current) return;
      const deltaX = event.clientX - drag.startX;
      if (drag.mode === 'pan') {
        if (thumbRange <= 0) return;
        tracksRef.current.scrollLeft = Math.max(
          0,
          Math.min(maxTimelineScroll, drag.startScrollLeft + (deltaX / thumbRange) * maxTimelineScroll),
        );
        return;
      }
      if (scrollbarTrackWidth <= 0 || containerWidth <= 0) return;
      const startTimelinePx = (totalDurationMs / 1000) * drag.startPixelsPerSecond + 200;
      const startThumbWidth = Math.max(
        30,
        Math.min(scrollbarTrackWidth, (containerWidth / startTimelinePx) * scrollbarTrackWidth),
      );
      const nextThumbWidth = Math.max(
        30,
        Math.min(scrollbarTrackWidth, drag.mode === 'right' ? startThumbWidth + deltaX : startThumbWidth - deltaX),
      );
      const nextTimelinePx = (containerWidth / nextThumbWidth) * scrollbarTrackWidth;
      const rawPps = (nextTimelinePx - 200) / (totalDurationMs / 1000);
      const nextPps = Math.max(minPps, Math.min(maxPps, rawPps));
      zoomAnchorRef.current =
        drag.mode === 'right'
          ? { type: 'left', timeMs: (drag.startScrollLeft / drag.startPixelsPerSecond) * 1000 }
          : {
              type: 'right',
              timeMs: ((drag.startScrollLeft + containerWidth) / drag.startPixelsPerSecond) * 1000,
            };
      setPixelsPerSecond(nextPps);
    };
    const handleUp = () => {
      scrollbarDragRef.current = null;
      zoomAnchorRef.current = null;
    };
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, [containerWidth, maxPps, maxTimelineScroll, minPps, scrollbarTrackWidth, thumbRange, totalDurationMs]);

  useEffect(() => {
    if (!isPlaying || !tracksRef.current) return;
    const playheadLeft = msToPixels(currentTimeMs);
    const container = tracksRef.current;
    const halfway = container.scrollLeft + container.clientWidth / 2;
    if (playheadLeft > halfway) {
      container.scrollLeft = playheadLeft - container.clientWidth / 2;
    }
  }, [currentTimeMs, isPlaying, msToPixels]);

  if (clips.length === 0) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 border-t bg-background/95 backdrop-blur supports-backdrop-filter:bg-background/60">
      <div className="relative overflow-hidden border-t bg-background/30 backdrop-blur-2xl" ref={containerRef}>
        <button
          type="button"
          className="absolute left-0 right-0 top-0 z-20 flex h-2 cursor-ns-resize items-center justify-center transition-colors hover:bg-muted/50"
          onMouseDown={handleResizeStart}
          aria-label="Resize track editor"
        >
          <GripHorizontal className="h-3 w-3 text-muted-foreground/50" />
        </button>

        <div className="mt-2 flex items-center justify-between border-b bg-muted/30 px-3 py-2">
          <div className="flex items-center gap-2">
            <Button type="button" variant="ghost" size="icon" className="h-7 w-7" onClick={onPlayPause} aria-label={isPlaying ? 'Pause' : 'Play'}>
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            <Button type="button" variant="ghost" size="icon" className="h-7 w-7" onClick={onStop} disabled={!isPlaying} aria-label="Stop">
              <Square className="h-3 w-3" />
            </Button>
            <span className="ml-2 text-xs tabular-nums text-muted-foreground">
              {formatTime(currentTimeMs)} / {formatTime(totalDurationMs)}
            </span>
            {timelineControls ? <div className="ml-2">{timelineControls}</div> : null}
          </div>

          {selectedClip && selectedClip.editable !== false ? (
            <div className="absolute left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-full border bg-background/70 px-2 py-1 shadow-sm">
              {onSplitClip ? (
                <Button type="button" variant="ghost" size="icon" className="h-7 w-7" onClick={handleSplit} title="Split at playhead">
                  <Scissors className="h-4 w-4" />
                </Button>
              ) : null}
              {onDuplicateClip ? (
                <Button type="button" variant="ghost" size="icon" className="h-7 w-7" onClick={() => onDuplicateClip(selectedClip.id)} title="Duplicate">
                  <Copy className="h-4 w-4" />
                </Button>
              ) : null}
              {onVolumeChange ? (
                <ClipVolumeButton
                  volume={selectedClip.volume ?? 1}
                  onChange={(value) => onVolumeChange(selectedClip.id, value)}
                />
              ) : null}
              {onDeleteClip ? (
                <Button type="button" variant="ghost" size="icon" className="h-7 w-7" onClick={() => onDeleteClip(selectedClip.id)} title="Delete">
                  <Trash2 className="h-4 w-4" />
                </Button>
              ) : null}
              {selectedClip.canRegenerate && onRegenerateClip ? (
                <Button type="button" variant="ghost" size="icon" className="h-7 w-7" onClick={() => onRegenerateClip(selectedClip.id)} title="Regenerate">
                  <RotateCcw className="h-4 w-4" />
                </Button>
              ) : null}
              {toolbarExtra}
            </div>
          ) : null}

          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Zoom:</span>
            <Button type="button" variant="ghost" size="icon" className="h-6 w-6" onClick={handleZoomOut} aria-label="Zoom out">
              <Minus className="h-3 w-3" />
            </Button>
            <Button type="button" variant="ghost" size="icon" className="h-6 w-6" onClick={handleZoomIn} aria-label="Zoom in">
              <Plus className="h-3 w-3" />
            </Button>
          </div>
        </div>

        <div
          ref={tracksRef}
          className="relative overflow-auto"
          style={{ height: `${timelineContainerHeight}px` }}
          onMouseMove={draggingClipId ? handleDragMove : undefined}
          onMouseUp={draggingClipId ? handleDragEnd : undefined}
          onMouseLeave={draggingClipId ? handleDragEnd : undefined}
        >
          <div className="sticky top-0 z-30 flex" style={{ width: `${timelineWidth + LABEL_COL_WIDTH}px` }}>
            <div className="sticky left-0 z-40 h-6 w-16 shrink-0 border-b border-r bg-muted/30" />
            <button
              type="button"
              className="relative h-6 cursor-pointer border-b bg-muted/20 text-left"
              style={{ width: `${timelineWidth}px` }}
              onClick={handleTimelineClick}
            >
              {timeMarkers.map((ms) => (
                <div key={ms} className="pointer-events-none absolute top-0 flex h-full flex-col justify-end" style={{ left: `${msToPixels(ms)}px` }}>
                  <div className="h-2 w-px bg-border" />
                  <span className="ml-1 select-none text-[10px] text-muted-foreground">{formatTime(ms)}</span>
                </div>
              ))}
            </button>
          </div>

          <div className="relative" style={{ width: `${timelineWidth + LABEL_COL_WIDTH}px`, height: `${tracksAreaHeight}px` }}>
            {tracks.map((trackNumber, index) => (
              <div key={trackNumber} className="absolute left-0 right-0 flex" style={{ top: `${index * TRACK_HEIGHT}px`, height: `${TRACK_HEIGHT}px` }}>
                <div className="sticky left-0 z-20 flex h-full w-16 shrink-0 items-center justify-center border-b border-r bg-background">
                  <div className="pointer-events-none absolute inset-0 bg-muted/20" />
                  <span className="relative select-none text-[10px] text-muted-foreground">{trackNumber}</span>
                  {index === 0 ? (
                    <button
                      type="button"
                      onClick={() => setExtraTracks((prev) => [...prev, Math.max(...tracks) + 1])}
                      className="absolute left-0 right-0 top-0 flex h-3 items-center justify-center text-muted-foreground/50 hover:bg-muted/40 hover:text-foreground"
                    >
                      <Plus className="h-2.5 w-2.5" />
                    </button>
                  ) : null}
                  {index === tracks.length - 1 ? (
                    <button
                      type="button"
                      onClick={() => setExtraTracks((prev) => [...prev, Math.min(...tracks) - 1])}
                      className="absolute bottom-0 left-0 right-0 flex h-3 items-center justify-center text-muted-foreground/50 hover:bg-muted/40 hover:text-foreground"
                    >
                      <Plus className="h-2.5 w-2.5" />
                    </button>
                  ) : null}
                </div>
                <div className={cn('flex-1 border-b pointer-events-none', index % 2 === 0 ? 'bg-background' : 'bg-muted/10')} />
              </div>
            ))}

            <div className="absolute bottom-0 top-0" style={{ left: `${LABEL_COL_WIDTH}px`, width: `${timelineWidth}px` }}>
              <button type="button" className="absolute inset-0 z-0 cursor-pointer" onClick={handleTimelineClick} />
              {clips.map((clip) => {
                const isSelected = selectedClipId === clip.id;
                const isDragging = draggingClipId === clip.id;
                const isTrimming = trimmingClipId === clip.id;
                const displayTrimStart = isTrimming && tempTrimValues ? tempTrimValues.trimStartMs : clip.trimStartMs ?? 0;
                const displayTrimEnd = isTrimming && tempTrimValues ? tempTrimValues.trimEndMs : clip.trimEndMs ?? 0;
                const isEditable = clip.editable !== false;
                const effectiveDuration = clip.durationMs - displayTrimStart - displayTrimEnd;
                const width = msToPixels(effectiveDuration);
                const left = isDragging ? dragPosition.x : msToPixels(clip.startMs);
                const trackIndex = tracks.indexOf(clip.track);
                const top = isDragging ? dragPosition.y : trackIndex * TRACK_HEIGHT;
                const isMovable = isEditable && clip.movable !== false;
                const isTrimmable = isEditable && clip.trimmable !== false;

                return (
                  <div
                    key={clip.id}
                    className={cn('absolute z-10 select-none overflow-visible rounded', isSelected && 'ring-2 ring-primary ring-offset-1')}
                    style={{ left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${TRACK_HEIGHT - 4}px` }}
                  >
                    <button
                      type="button"
                      className={cn(
                        'h-full w-full overflow-hidden rounded transition-all',
                        isMovable ? 'cursor-move' : 'cursor-default',
                        getClipClasses(clip.variant, isSelected),
                        isDragging && 'opacity-80 shadow-lg',
                      )}
                      onClick={(event) => {
                        event.stopPropagation();
                        if (!draggingClipId && !trimmingClipId) onSelectClip(clip.id);
                      }}
                      onMouseDown={(event) => {
                        if (isMovable && !(event.target as HTMLElement).closest('.trim-handle')) {
                          handleDragStart(event, clip);
                        }
                      }}
                    >
                      <div className="absolute left-1 right-1 top-0 z-10">
                        <p className="truncate text-[9px] font-medium">{clip.label}</p>
                        {clip.sublabel && clip.variant !== 'reference' ? (
                          <p className="truncate text-[8px] opacity-80">{clip.sublabel}</p>
                        ) : null}
                      </div>
                      {clip.audioUrl ? (
                        <div className="absolute inset-0 top-3">
                          <ClipWaveform
                            audioUrl={clip.audioUrl}
                            width={width}
                            trimStartMs={displayTrimStart}
                            trimEndMs={displayTrimEnd}
                            durationMs={clip.durationMs}
                          />
                        </div>
                      ) : (
                        <div className="absolute inset-x-2 bottom-1 top-4 overflow-hidden text-[9px] leading-tight opacity-80">
                          {clip.sublabel ?? clip.label}
                        </div>
                      )}
                    </button>
                    {isSelected && isTrimmable ? (
                      <>
                        <button
                          type="button"
                          className="trim-handle absolute bottom-0 left-0 top-0 z-30 w-2 cursor-ew-resize rounded-l bg-primary/20 hover:bg-primary/30"
                          onMouseDown={(event) => handleTrimStart(event, clip, 'start')}
                          aria-label="Trim start"
                        />
                        <button
                          type="button"
                          className="trim-handle absolute bottom-0 right-0 top-0 z-30 w-2 cursor-ew-resize rounded-r bg-primary/20 hover:bg-primary/30"
                          onMouseDown={(event) => handleTrimStart(event, clip, 'end')}
                          aria-label="Trim end"
                        />
                      </>
                    ) : null}
                  </div>
                );
              })}
              <div
                className={cn(
                  'absolute bottom-0 top-0 z-30 w-1 rounded-full bg-accent',
                  isDraggingPlayhead ? 'cursor-grabbing' : 'cursor-grab',
                )}
                style={{ left: `${msToPixels(currentTimeMs)}px` }}
                onMouseDown={handlePlayheadMouseDown}
                onKeyDown={handlePlayheadKeyDown}
                role="slider"
                tabIndex={0}
                aria-label="Timeline playhead"
                aria-valuemin={0}
                aria-valuemax={Math.round(totalDurationMs)}
                aria-valuenow={Math.round(currentTimeMs)}
                aria-valuetext={formatTime(currentTimeMs)}
              >
                <div className="absolute -top-1 left-1/2 h-3 w-3 -translate-x-1/2 rounded-full bg-accent" />
              </div>
            </div>
          </div>
        </div>

        <TimelineScrollbar
          trackRef={scrollbarTrackRef}
          height={SCRUB_BAR_HEIGHT}
          labelWidth={LABEL_COL_WIDTH}
          thumbWidth={thumbWidth}
          thumbLeft={thumbLeft}
          canScrollHorizontally={canScrollHorizontally}
          pixelsPerSecond={pixelsPerSecond}
          minPixelsPerSecond={minPps}
          maxPixelsPerSecond={maxPps}
          onMouseDown={handleScrollbarMouseDown}
        />
      </div>
    </div>
  );
}
