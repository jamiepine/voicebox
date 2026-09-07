import { ChevronDown, ChevronRight, PhoneIncoming, PhoneOutgoing } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import type { VoiceCall } from '@/lib/api/types';
import { useCall, useCalls } from '@/lib/hooks/useVoiceAgents';
import { cn } from '@/lib/utils/cn';
import { formatDate } from '@/lib/utils/format';
import { OutcomeBadge, sentimentTone } from './shared';

interface CallsPanelProps {
  agentId: string;
  live: boolean;
  onOpenInConsole: (callId: string) => void;
}

export function CallsPanel({ agentId, live, onOpenInConsole }: CallsPanelProps) {
  const { data, isLoading } = useCalls(agentId, live);
  const [open, setOpen] = useState<string | null>(null);
  const calls = data?.calls ?? [];

  if (isLoading) return <div className="text-sm text-muted-foreground">Loading…</div>;
  if (calls.length === 0) {
    return (
      <div className="text-sm text-muted-foreground py-8 text-center rounded-lg border border-dashed border-border">
        No calls yet.
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {calls.map((c) => (
        <CallRow
          key={c.id}
          call={c}
          expanded={open === c.id}
          onToggle={() => setOpen(open === c.id ? null : c.id)}
          onOpenInConsole={() => onOpenInConsole(c.id)}
        />
      ))}
    </div>
  );
}

function CallRow({
  call,
  expanded,
  onToggle,
  onOpenInConsole,
}: {
  call: VoiceCall;
  expanded: boolean;
  onToggle: () => void;
  onOpenInConsole: () => void;
}) {
  const { data: detail } = useCall(expanded ? call.id : null);
  const Icon = call.direction === 'inbound' ? PhoneIncoming : PhoneOutgoing;
  return (
    <div className={cn('rounded-lg border', expanded ? 'border-accent/40' : 'border-border')}>
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-muted/30 rounded-lg"
      >
        {expanded ? (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-4 w-4 text-muted-foreground" />
        )}
        <Icon className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm">{formatDate(call.started_at)}</span>
        <span className="text-xs text-muted-foreground">{call.turn_count} turns</span>
        <div className="ml-auto flex items-center gap-2">
          {call.status === 'in_progress' && (
            <span className="text-xs text-accent animate-pulse">live</span>
          )}
          <OutcomeBadge outcome={call.outcome} />
        </div>
      </button>
      {expanded && (
        <div className="px-4 pb-3 space-y-2">
          {call.summary && (
            <div className="text-sm rounded-md bg-muted/30 p-2.5">{call.summary}</div>
          )}
          <div className="space-y-1.5">
            {(detail?.turns ?? []).map((t) => (
              <div key={t.id} className="text-sm flex gap-2">
                <span
                  className={cn(
                    'shrink-0 w-16 text-[11px] uppercase tracking-wide pt-0.5',
                    t.role === 'agent' ? 'text-accent' : sentimentTone(t.sentiment),
                  )}
                >
                  {t.role}
                </span>
                <span className="text-foreground/90">{t.text}</span>
              </div>
            ))}
          </div>
          <div className="flex justify-end">
            <Button size="sm" variant="ghost" onClick={onOpenInConsole}>
              Open in console
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
