import { useCallback, useEffect, useMemo } from 'react';
import type { AudioTrackClip } from '@/components/AudioTimeline/AudioTrackEditor';
import { AudioTrackEditor } from '@/components/AudioTimeline/AudioTrackEditor';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { StoryItemDetail } from '@/lib/api/types';
import {
  useDuplicateStoryItem,
  useMoveStoryItem,
  useRemoveStoryItem,
  useSplitStoryItem,
  useTrimStoryItem,
  useUpdateStoryItemVolume,
} from '@/lib/hooks/useStories';
import { useGenerationStore } from '@/stores/generationStore';
import { useStoryStore } from '@/stores/storyStore';

interface StoryTrackEditorProps {
  storyId: string;
  items: StoryItemDetail[];
}

function getEffectiveDuration(item: StoryItemDetail) {
  return item.duration * 1000 - (item.trim_start_ms || 0) - (item.trim_end_ms || 0);
}

export function StoryTrackEditor({ storyId, items }: StoryTrackEditorProps) {
  const moveItem = useMoveStoryItem();
  const trimItem = useTrimStoryItem();
  const splitItem = useSplitStoryItem();
  const duplicateItem = useDuplicateStoryItem();
  const removeItem = useRemoveStoryItem();
  const updateVolume = useUpdateStoryItemVolume();
  const { toast } = useToast();
  const addPendingGeneration = useGenerationStore((state) => state.addPendingGeneration);

  const selectedClipId = useStoryStore((state) => state.selectedClipId);
  const setSelectedClipId = useStoryStore((state) => state.setSelectedClipId);
  const editorHeight = useStoryStore((state) => state.trackEditorHeight);
  const setEditorHeight = useStoryStore((state) => state.setTrackEditorHeight);
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

  useEffect(() => {
    if (items.length === 0 || isActiveStory) return;
    const totalDuration = Math.max(
      ...items.map((item) => item.start_time_ms + getEffectiveDuration(item)),
      0,
    );
    setActiveStory(storyId, items, totalDuration);
  }, [storyId, items, isActiveStory, setActiveStory]);

  const sortedItems = useMemo(
    () => [...items].sort((a, b) => a.start_time_ms - b.start_time_ms),
    [items],
  );

  const clips = useMemo<AudioTrackClip[]>(
    () =>
      items.map((item) => ({
        id: item.id,
        startMs: item.start_time_ms,
        durationMs: item.duration * 1000,
        track: item.track,
        label: item.engine === 'import' ? item.text : item.profile_name,
        sublabel: item.engine === 'import' ? undefined : item.text,
        audioUrl: item.version_id
          ? apiClient.getVersionAudioUrl(item.version_id)
          : apiClient.getAudioUrl(item.generation_id),
        trimStartMs: item.trim_start_ms || 0,
        trimEndMs: item.trim_end_ms || 0,
        volume: item.volume,
        variant: 'accent',
        canRegenerate: item.engine !== 'import',
      })),
    [items],
  );

  const handlePlayPause = () => {
    if (isCurrentlyPlaying) {
      pause();
    } else {
      play(storyId, sortedItems);
    }
  };

  const handleMoveClip = useCallback(
    (clipId: string, startMs: number, track: number) => {
      moveItem.mutate(
        {
          storyId,
          itemId: clipId,
          data: { start_time_ms: startMs, track },
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
    },
    [moveItem, storyId, toast],
  );

  const handleTrimClip = useCallback(
    (clipId: string, trimStartMs: number, trimEndMs: number) => {
      trimItem.mutate(
        {
          storyId,
          itemId: clipId,
          data: {
            trim_start_ms: trimStartMs,
            trim_end_ms: trimEndMs,
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
    },
    [storyId, toast, trimItem],
  );

  const handleSplitClip = useCallback(
    (clipId: string, splitTimeMs: number) => {
      const item = items.find((candidate) => candidate.id === clipId);
      if (!item) return;
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
          itemId: clipId,
          data: { split_time_ms: splitTimeMs },
        },
        {
          onSuccess: () => setSelectedClipId(null),
          onError: (error) => {
            toast({
              title: 'Failed to split clip',
              description: error instanceof Error ? error.message : String(error),
              variant: 'destructive',
            });
          },
        },
      );
    },
    [items, setSelectedClipId, splitItem, storyId, toast],
  );

  const handleDuplicateClip = useCallback(
    (clipId: string) => {
      duplicateItem.mutate(
        { storyId, itemId: clipId },
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
    },
    [duplicateItem, storyId, toast],
  );

  const handleDeleteClip = useCallback(
    (clipId: string) => {
      removeItem.mutate(
        { storyId, itemId: clipId },
        {
          onSuccess: () => setSelectedClipId(null),
          onError: (error) => {
            toast({
              title: 'Failed to delete clip',
              description: error instanceof Error ? error.message : String(error),
              variant: 'destructive',
            });
          },
        },
      );
    },
    [removeItem, setSelectedClipId, storyId, toast],
  );

  const handleRegenerateClip = useCallback(
    async (clipId: string) => {
      const item = items.find((candidate) => candidate.id === clipId);
      if (!item) return;
      try {
        await apiClient.regenerateGeneration(item.generation_id);
        addPendingGeneration(item.generation_id);
      } catch (error) {
        toast({
          title: 'Failed to regenerate',
          description: error instanceof Error ? error.message : String(error),
          variant: 'destructive',
        });
      }
    },
    [addPendingGeneration, items, toast],
  );

  const handleVolumeChange = useCallback(
    (clipId: string, volume: number) => {
      updateVolume.mutate(
        {
          storyId,
          itemId: clipId,
          data: { volume },
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
      );
    },
    [storyId, toast, updateVolume],
  );

  if (items.length === 0) return null;

  return (
    <AudioTrackEditor
      clips={clips}
      selectedClipId={selectedClipId}
      currentTimeMs={currentTimeMs}
      isPlaying={isCurrentlyPlaying}
      height={editorHeight}
      onHeightChange={setEditorHeight}
      onSelectClip={setSelectedClipId}
      onSeek={seek}
      onPlayPause={handlePlayPause}
      onStop={stop}
      onMoveClip={handleMoveClip}
      onTrimClip={handleTrimClip}
      onSplitClip={handleSplitClip}
      onDuplicateClip={handleDuplicateClip}
      onDeleteClip={handleDeleteClip}
      onRegenerateClip={handleRegenerateClip}
      onVolumeChange={handleVolumeChange}
    />
  );
}
