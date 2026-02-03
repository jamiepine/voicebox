import { HugeiconsIcon } from '@hugeicons/react';
import {
  Copy01Icon,
  DragDropHorizontalIcon,
  RemoveIcon,
  PauseIcon,
  PlayIcon,
  Add01Icon,
  Scissor01Icon,
  SquareIcon,
  Delete01Icon,
} from '@hugeicons/core-free-icons';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { StoryItemDetail } from '@/lib/api/types';
import {
  useDuplicateStoryItem,
  useMoveStoryItem,
  useRemoveStoryItem,
  useSplitStoryItem,
  useTrimStoryItem,
} from '@/lib/hooks/useStories';
import { cn } from '@/lib/utils/cn';
import { useStoryStore } from '@/stores/storyStore';

// Clip waveform component with trim support
function ClipWaveform({
  generationId,
  width,
  trimStartMs,
  trimEndMs,
  duration,
}: {
  generationId: string;
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

    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
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

    const audioUrl = apiClient.getAudioUrl(generationId);
    wavesurfer.load(audioUrl).catch(() => {
      // Ignore load errors
    });

    return () => {
      wavesurfer.destroy();
      wavesurferRef.current = null;
    };
  }, [generationId, fullWaveformWidth]);

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

interface StoryTrackEditorProps {
  storyId: string;
  items: StoryItemDetail[];
}

const TRACK_HEIGHT = 48;
const TIME_RULER_HEIGHT = 24; // h-6 = 1.5rem = 24px
const MIN_PIXELS_PER_SECOND = 10;
const MAX_PIXELS_PER_SECOND = 200;
const DEFAULT_PIXELS_PER_SECOND = 50;
const DEFAULT_TRACKS = [1, 0, -1]; // Default 3 tracks
const MIN_EDITOR_HEIGHT = 120;
const MAX_EDITOR_HEIGHT = 500;

export function StoryTrackEditor({ storyId, items }: StoryTrackEditorProps) {
  const [pixelsPerSecond, setPixelsPerSecond] = useState(DEFAULT_PIXELS_PER_SECOND);
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
  const { toast } = useToast();

  // Selection state
  const selectedClipId = useStoryStore((state) => state.selectedClipId);
  const setSelectedClipId = useStoryStore((state) => state.setSelectedClipId);

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

  // Calculate unique tracks from items, always showing at least 3 default tracks
  const tracks = useMemo(() => {
    const trackSet = new Set([...DEFAULT_TRACKS, ...items.map((item) => item.track)]);
    return Array.from(trackSet).sort((a, b) => b - a); // Higher tracks on top
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

  // Calculate effective duration (accounting for trims)
  const getEffectiveDuration = (item: StoryItemDetail) => {
    return item.duration * 1000 - (item.trim_start_ms || 0) - (item.trim_end_ms || 0);
  };

  // Calculate total duration (using effective durations)
  const totalDurationMs = useMemo(() => {
    if (items.length === 0) return 10000; // Default 10 seconds
    return Math.max(...items.map((item) => item.start_time_ms + getEffectiveDuration(item)), 10000);
  }, [items, getEffectiveDuration]);

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
    setPixelsPerSecond((prev) => Math.min(prev * 1.5, MAX_PIXELS_PER_SECOND));
  };

  const handleZoomOut = () => {
    setPixelsPerSecond((prev) => Math.max(prev / 1.5, MIN_PIXELS_PER_SECOND));
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

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!tracksRef.current || draggingItem || trimmingItem) return;
    const rect = tracksRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left + tracksRef.current.scrollLeft;
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
    if (!selectedClipId) return;

    const item = items.find((i) => i.id === selectedClipId);
    if (!item) return;

    const splitTimeMs = currentTimeMs - item.start_time_ms;
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
      x: rect.left - tracksRef.current.getBoundingClientRect().left + tracksRef.current.scrollLeft,
      // Subtract ruler height since clips are positioned relative to tracks area, not the scrollable container
      y: rect.top - tracksRef.current.getBoundingClientRect().top - TIME_RULER_HEIGHT,
    });
    setDraggingItem(item.id);
  };

  const handleDragMove = useCallback(
    (e: React.MouseEvent) => {
      if (!draggingItem || !tracksRef.current) return;

      const rect = tracksRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + tracksRef.current.scrollLeft - dragOffset.x;
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
  const timelineContainerHeight = editorHeight - 40; // Subtract toolbar height

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
          <HugeiconsIcon icon={DragDropHorizontalIcon} size={12} className="h-3 w-3 text-muted-foreground/50 group-hover:text-muted-foreground" />
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
            >
              {isCurrentlyPlaying ? <HugeiconsIcon icon={PauseIcon} size={16} className="h-4 w-4" /> : <HugeiconsIcon icon={PlayIcon} size={16} className="h-4 w-4" />}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={handleStop}
              disabled={!isCurrentlyPlaying}
            >
              <HugeiconsIcon icon={SquareIcon} size={12} className="h-3 w-3" />
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
              >
                <HugeiconsIcon icon={Scissor01Icon} size={16} className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={handleDuplicate}
                title="Duplicate (Cmd/Ctrl+D)"
              >
                <HugeiconsIcon icon={Copy01Icon} size={16} className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={handleDelete}
                title="Delete (Delete/Backspace)"
              >
                <HugeiconsIcon icon={Delete01Icon} size={16} className="h-4 w-4" />
              </Button>
            </div>
          )}

          {/* Zoom controls - right side */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Zoom:</span>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleZoomOut}>
              <HugeiconsIcon icon={RemoveIcon} size={12} className="h-3 w-3" />
            </Button>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleZoomIn}>
              <HugeiconsIcon icon={Add01Icon} size={12} className="h-3 w-3" />
            </Button>
          </div>
        </div>

        {/* Timeline container with track labels sidebar */}
        <div className="flex" style={{ height: `${timelineContainerHeight}px` }}>
          {/* Track labels sidebar - fixed width */}
          <div className="w-16 shrink-0 border-r bg-muted/20 overflow-hidden">
            {/* Spacer for time ruler */}
            <div className="h-6 border-b bg-muted/30" />
            {/* Track labels */}
            <div style={{ height: `${tracksAreaHeight}px` }}>
              {tracks.map((trackNumber, index) => (
                <div
                  key={trackNumber}
                  className={cn(
                    'border-b flex items-center justify-center',
                    index % 2 === 0 ? 'bg-background' : 'bg-muted/10',
                  )}
                  style={{ height: `${TRACK_HEIGHT}px` }}
                >
                  <span className="text-[10px] text-muted-foreground select-none">
                    {trackNumber}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Scrollable timeline area */}
          {/* biome-ignore lint/a11y/noStaticElementInteractions: Container handles drag events for child clips */}
          <div
            ref={tracksRef}
            className="overflow-auto relative flex-1"
            onMouseMove={draggingItem ? handleDragMove : undefined}
            onMouseUp={draggingItem ? handleDragEnd : undefined}
            onMouseLeave={draggingItem ? handleDragEnd : undefined}
          >
            {/* Time ruler - clickable to seek */}
            <button
              type="button"
              className="h-6 border-b bg-muted/20 sticky top-0 z-10 cursor-pointer text-left"
              style={{ width: `${timelineWidth}px` }}
              onClick={(e) => handleTimelineClick(e as unknown as React.MouseEvent<HTMLDivElement>)}
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

            {/* Tracks area */}
            <div
              className="relative"
              style={{ width: `${timelineWidth}px`, height: `${tracksAreaHeight}px` }}
            >
              {/* Track backgrounds - pointer-events-none to allow clicks to pass through */}
              {tracks.map((trackNumber, index) => (
                <div
                  key={trackNumber}
                  className={cn(
                    'absolute left-0 right-0 border-b pointer-events-none',
                    index % 2 === 0 ? 'bg-background' : 'bg-muted/10',
                  )}
                  style={{
                    top: `${index * TRACK_HEIGHT}px`,
                    height: `${TRACK_HEIGHT}px`,
                  }}
                />
              ))}

              {/* Click area for seeking - z-index lower than clips */}
              <button
                type="button"
                className="absolute inset-0 z-0 cursor-pointer"
                onClick={(e) => handleTimelineClick(e as unknown as React.MouseEvent<HTMLDivElement>)}
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
                          {item.profile_name}
                        </p>
                      </div>
                      {/* Waveform */}
                      <div className="absolute inset-0 top-3">
                        <ClipWaveform
                          generationId={item.generation_id}
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
      </div>
    </div>
  );
}
