import { FloatingGenerateBox } from '@/components/Generation/FloatingGenerateBox';
import { usePlayerStore } from '@/stores/playerStore';
import { StoryContent } from './StoryContent';
import { StoryList } from './StoryList';

export function StoriesTab() {
  const audioUrl = usePlayerStore((state) => state.audioUrl);

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden -mx-8">
      {/* Main content area */}
      <div className="flex-1 min-h-0 flex gap-6 overflow-hidden relative">
        {/* Left Column - Story List */}
        <div className="flex flex-col min-h-0 overflow-hidden w-full max-w-[360px] shrink-0">
          <StoryList />
        </div>

        {/* Right Column - Story Content */}
        <div className="flex flex-col min-h-0 overflow-hidden flex-1 pr-8">
          <StoryContent />
        </div>

        {/* Floating Generate Box - position is managed via storyStore.trackEditorHeight */}
        <FloatingGenerateBox showVoiceSelector isPlayerOpen={!!audioUrl} />
      </div>
    </div>
  );
}
