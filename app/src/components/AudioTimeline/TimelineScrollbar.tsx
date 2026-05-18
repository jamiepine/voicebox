import type { MouseEvent, RefObject } from 'react';
import { cn } from '@/lib/utils/cn';

type TimelineScrollbarMode = 'pan' | 'left' | 'right';

interface TimelineScrollbarProps {
  trackRef: RefObject<HTMLDivElement>;
  labelWidth?: number;
  height?: number;
  thumbWidth: number;
  thumbLeft: number;
  canScrollHorizontally: boolean;
  pixelsPerSecond: number;
  minPixelsPerSecond: number;
  maxPixelsPerSecond: number;
  onMouseDown: (mode: TimelineScrollbarMode) => (event: MouseEvent) => void;
}

export function TimelineScrollbar({
  trackRef,
  labelWidth = 64,
  height = 16,
  thumbWidth,
  thumbLeft,
  canScrollHorizontally,
  pixelsPerSecond,
  minPixelsPerSecond,
  maxPixelsPerSecond,
  onMouseDown,
}: TimelineScrollbarProps) {
  return (
    <div className="flex border-t bg-background/40" style={{ height: `${height}px` }}>
      <div className="shrink-0 border-r" style={{ width: `${labelWidth}px` }} />
      <div ref={trackRef} className="relative flex-1 overflow-hidden select-none px-1">
        <div
          className="absolute top-1 bottom-1 rounded-full bg-foreground/10 transition-colors hover:bg-foreground/15"
          style={{ width: `${thumbWidth}px`, left: `${thumbLeft}px` }}
        >
          <div
            role="slider"
            aria-label="Zoom from left edge"
            aria-valuenow={Math.round(pixelsPerSecond)}
            aria-valuemin={Math.round(minPixelsPerSecond)}
            aria-valuemax={Math.round(maxPixelsPerSecond)}
            className="absolute top-0 bottom-0 left-0 w-1.5 cursor-ew-resize rounded-l-full bg-foreground/25 transition-colors hover:bg-foreground/40"
            onMouseDown={onMouseDown('left')}
          />
          <div
            className={cn(
              'absolute top-0 bottom-0 left-1.5 right-1.5',
              canScrollHorizontally ? 'cursor-grab active:cursor-grabbing' : 'cursor-default',
            )}
            onMouseDown={canScrollHorizontally ? onMouseDown('pan') : undefined}
          />
          <div
            role="slider"
            aria-label="Zoom from right edge"
            aria-valuenow={Math.round(pixelsPerSecond)}
            aria-valuemin={Math.round(minPixelsPerSecond)}
            aria-valuemax={Math.round(maxPixelsPerSecond)}
            className="absolute top-0 bottom-0 right-0 w-1.5 cursor-ew-resize rounded-r-full bg-foreground/25 transition-colors hover:bg-foreground/40"
            onMouseDown={onMouseDown('right')}
          />
        </div>
      </div>
    </div>
  );
}
