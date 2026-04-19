import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils/cn';
import { type LogEntry, useLogStore } from '@/stores/logStore';

function formatTime(timestamp: number): string {
  const d = new Date(timestamp);
  return d.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

function LogLine({ entry }: { entry: LogEntry }) {
  return (
    <div className="flex gap-3 font-mono text-xs leading-5 hover:bg-muted/30">
      <span className="text-muted-foreground/50 select-none shrink-0">
        {formatTime(entry.timestamp)}
      </span>
      <span
        className={cn(
          'whitespace-pre-wrap break-all',
          entry.stream === 'stderr' ? 'text-orange-400/80' : 'text-muted-foreground',
        )}
      >
        {entry.line}
      </span>
    </div>
  );
}

export function LogsPage() {
  const entries = useLogStore((s) => s.entries);
  const clear = useLogStore((s) => s.clear);
  const containerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Auto-scroll to bottom when new entries arrive
  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [entries.length, autoScroll]);

  // Detect manual scroll to disable auto-scroll
  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    setAutoScroll(atBottom);
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-medium">Server Logs</h3>
          <p className="text-sm text-muted-foreground">
            {entries.length} {entries.length === 1 ? 'line' : 'lines'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {!autoScroll && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setAutoScroll(true);
                containerRef.current?.scrollTo({ top: containerRef.current.scrollHeight });
              }}
            >
              Scroll to bottom
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={clear}>
            Clear
          </Button>
        </div>
      </div>

      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 min-h-0 overflow-y-auto rounded-md border bg-black/20 p-3"
      >
        {entries.length === 0 ? (
          <div className="text-sm text-muted-foreground/50 font-mono space-y-1">
            <p>No log output yet.</p>
            {!import.meta.env?.PROD && (
              <p>
                Server logs are only captured when the app manages the server process (production
                builds).
              </p>
            )}
          </div>
        ) : (
          entries.map((entry) => <LogLine key={entry.id} entry={entry} />)
        )}
      </div>
    </div>
  );
}
