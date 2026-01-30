import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { GripVertical, Mic, MoreHorizontal, Play, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Textarea } from '@/components/ui/textarea';
import type { StoryItemDetail } from '@/lib/api/types';
import { cn } from '@/lib/utils/cn';
import { useStoryStore } from '@/stores/storyStore';
import { useServerStore } from '@/stores/serverStore';

interface StoryChatItemProps {
  item: StoryItemDetail;
  storyId: string;
  index: number;
  onRemove: () => void;
  currentTimeMs: number;
  isPlaying: boolean;
  dragHandleProps?: React.HTMLAttributes<HTMLButtonElement>;
  isDragging?: boolean;
}

export function StoryChatItem({
  item,
  onRemove,
  currentTimeMs,
  isPlaying,
  dragHandleProps,
  isDragging,
}: StoryChatItemProps) {
  const seek = useStoryStore((state) => state.seek);
  const serverUrl = useServerStore((state) => state.serverUrl);
  const [avatarError, setAvatarError] = useState(false);

  const avatarUrl = `${serverUrl}/profiles/${item.profile_id}/avatar`;

  // Check if this item is currently playing based on timecode
  const itemStartMs = item.start_time_ms;
  const itemEndMs = item.start_time_ms + item.duration * 1000;
  const isCurrentlyPlaying = isPlaying && currentTimeMs >= itemStartMs && currentTimeMs < itemEndMs;

  const handlePlay = () => {
    // Seek to the start of this item
    seek(itemStartMs);
  };

  const formatTime = (ms: number): string => {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const milliseconds = Math.floor((ms % 1000) / 100);
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds}`;
  };

  return (
    <div
      className={cn(
        'flex items-start gap-3 p-4 rounded-lg border transition-colors',
        isCurrentlyPlaying && 'bg-muted/70 border-primary',
        !isCurrentlyPlaying && 'hover:bg-muted/50',
        isDragging && 'opacity-50 shadow-lg',
      )}
    >
      {/* Drag Handle */}
      {dragHandleProps && (
        <button
          type="button"
          className="shrink-0 cursor-grab active:cursor-grabbing touch-none text-muted-foreground hover:text-foreground transition-colors"
          {...dragHandleProps}
        >
          <GripVertical className="h-5 w-5" />
        </button>
      )}

      {/* Voice Avatar */}
      <div className="shrink-0">
        <div className="h-10 w-10 rounded-full bg-muted flex items-center justify-center overflow-hidden">
          {!avatarError ? (
            <img
              src={avatarUrl}
              alt={`${item.profile_name} avatar`}
              className={cn(
                'h-full w-full object-cover transition-all duration-200',
                !isCurrentlyPlaying && 'grayscale'
              )}
              onError={() => setAvatarError(true)}
            />
          ) : (
            <Mic className="h-5 w-5 text-muted-foreground" />
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-2">
          <span className="font-medium text-sm">{item.profile_name}</span>
          <span className="text-xs text-muted-foreground">{item.language}</span>
          <span className="text-xs text-muted-foreground tabular-nums ml-auto">
            {formatTime(itemStartMs)}
          </span>
        </div>
        <Textarea
          value={item.text}
          className="flex-1 resize-none text-sm text-muted-foreground select-text bg-card cursor-text"
          readOnly
          onDoubleClick={handlePlay}
        />
      </div>

      {/* Actions */}
      <div className="shrink-0">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="Actions">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={handlePlay}>
              <Play className="mr-2 h-4 w-4" />
              Play from here
            </DropdownMenuItem>
            <DropdownMenuItem onClick={onRemove} className="text-destructive focus:text-destructive">
              <Trash2 className="mr-2 h-4 w-4" />
              Remove from Story
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}

// Sortable wrapper component
export function SortableStoryChatItem(props: Omit<StoryChatItemProps, 'dragHandleProps' | 'isDragging'>) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: props.item.generation_id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  return (
    <div ref={setNodeRef} style={style} {...attributes}>
      <StoryChatItem
        {...props}
        dragHandleProps={listeners}
        isDragging={isDragging}
      />
    </div>
  );
}
