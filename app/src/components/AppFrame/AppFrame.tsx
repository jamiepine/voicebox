import { useRouterState } from '@tanstack/react-router';
import { AudioKeepAlive, shouldMountAudioKeepAlive } from '@/components/AudioPlayer/AudioKeepAlive';
import { AudioPlayer } from '@/components/AudioPlayer/AudioPlayer';
import { StoryTrackEditor } from '@/components/StoriesTab/StoryTrackEditor';
import { TitleBarDragRegion } from '@/components/TitleBarDragRegion';
import { TOP_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { useStory } from '@/lib/hooks/useStories';
import { cn } from '@/lib/utils/cn';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';
import { useStoryStore } from '@/stores/storyStore';

interface AppFrameProps {
  children: React.ReactNode;
}

export function AppFrame({ children }: AppFrameProps) {
  const routerState = useRouterState();
  const platform = usePlatform();
  const keepAudioSessionAlive = useServerStore((state) => state.keepAudioSessionAlive);
  const isStoriesRoute = routerState.location.pathname === '/stories';
  const mountAudioKeepAlive = shouldMountAudioKeepAlive(
    platform.metadata.isTauri,
    keepAudioSessionAlive,
  );

  const selectedStoryId = useStoryStore((state) => state.selectedStoryId);
  const { data: story } = useStory(selectedStoryId);

  // Show track editor when on stories route with a selected story that has items
  const showTrackEditor = isStoriesRoute && selectedStoryId && story && story.items.length > 0;

  return (
    <div
      className={cn('h-screen bg-background flex flex-col overflow-hidden', TOP_SAFE_AREA_PADDING)}
    >
      <TitleBarDragRegion />
      {mountAudioKeepAlive && <AudioKeepAlive />}
      {children}
      {showTrackEditor ? (
        <StoryTrackEditor storyId={story.id} items={story.items} />
      ) : (
        <AudioPlayer />
      )}
    </div>
  );
}
