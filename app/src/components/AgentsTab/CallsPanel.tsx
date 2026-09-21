import {
  ChevronDown,
  ChevronRight,
  Download,
  FlaskConical,
  PhoneIncoming,
  PhoneOutgoing,
} from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { apiClient } from '@/lib/api/client';
import type { VoiceCall } from '@/lib/api/types';
import { useCall, useCalls } from '@/lib/hooks/useVoiceAgents';
import { cn } from '@/lib/utils/cn';
import { formatDate } from '@/lib/utils/format';
import { formatMs, OutcomeBadge, ScoreBadge, sentimentTone } from './shared';

interface CallsPanelProps {
  agentId: string;
  live: boolean;
  onOpenInConsole: (callId: string) => void;
}

export function CallsPanel({ agentId, live, onOpenInConsole }: CallsPanelProps) {
  const { data, isLoading } = useCalls(agentId, live);
  const [open, setOpen] = useState<string | null>(null);
  const [showSims, setShowSims] = useState(true);
  const calls = (data?.calls ?? []).filter((c) => showSims || c.direction !== 'simulation');

  if (isLoading) return <div className="text-sm text-muted-foreground">Loading…</div>;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Checkbox id="show-sims" checked={showSims} onCheckedChange={setShowSims} />
          <label htmlFor="show-sims" className="cursor-pointer">
            Show test calls
          </label>
        </div>
        <Button variant="ghost" size="sm" className="ml-auto" asChild>
          <a href={apiClient.getCallsCsvUrl(agentId)} target="_blank" rel="noreferrer">
            <Download className="h-4 w-4" /> Export CSV
          </a>
        </Button>
      </div>
      {calls.length === 0 && (
        <div className="text-sm text-muted-foreground py-8 text-center rounded-lg border border-dashed border-border">
          No calls yet.
        </div>
      )}
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
  const Icon =
    call.direction === 'inbound'
      ? PhoneIncoming
      : call.direction === 'simulation'
        ? FlaskConical
        : PhoneOutgoing;
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
        <span className="text-xs text-muted-foreground">
          {call.turn_count} turns{call.variant ? ` · ${call.variant}` : ''}
        </span>
        {(call.flags ?? []).length > 0 && (
          <span className="text-[10px] text-muted-foreground">{(call.flags ?? []).join(', ')}</span>
        )}
        <div className="ml-auto flex items-center gap-2">
          {call.status === 'in_progress' && (
            <span className="text-xs text-accent animate-pulse">live</span>
          )}
          <ScoreBadge score={call.score} />
          <OutcomeBadge outcome={call.outcome} />
        </div>
      </button>
      {expanded && (
        <div className="px-4 pb-3 space-y-2">
          {call.summary && (
            <div className="text-sm rounded-md bg-muted/30 p-2.5">{call.summary}</div>
          )}
          {call.score != null && (
            <div className="text-xs text-muted-foreground">
              Score {call.score}/100{call.score_reason ? ` — ${call.score_reason}` : ''}
            </div>
          )}
          {call.analysis && Object.keys(call.analysis).length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(call.analysis).map(([k, v]) => (
                <span key={k} className="rounded-full border border-border px-2 py-0.5 text-[11px]">
                  {k}: <span className="text-foreground">{v == null ? '—' : String(v)}</span>
                </span>
              ))}
            </div>
          )}
          <div className="space-y-1.5">
            {(detail?.turns ?? []).map((t) => (
              <div key={t.id} className="text-sm flex gap-2">
                <span
                  className={cn(
                    'shrink-0 w-16 text-[11px] uppercase tracking-wide pt-0.5',
                    t.role === 'agent'
                      ? 'text-accent'
                      : t.role === 'tool'
                        ? 'text-amber-300'
                        : sentimentTone(t.sentiment),
                  )}
                >
                  {t.role === 'tool'
                    ? t.tool_name
                    : t.role === 'agent' && t.source === 'operator'
                      ? 'operator'
                      : t.role}
                </span>
                <span
                  className={cn(
                    'text-foreground/90',
                    t.role === 'tool' && 'text-xs text-muted-foreground',
                  )}
                >
                  {t.text}
                  {t.interrupted && (
                    <span className="text-[10px] text-muted-foreground ml-1">(interrupted)</span>
                  )}
                  {t.role === 'agent' && (t.stt_ms != null || t.llm_ms != null) && (
                    <span className="text-[10px] text-muted-foreground/70 ml-2">
                      {[
                        t.stt_ms != null ? `stt ${formatMs(t.stt_ms)}` : '',
                        t.llm_ms != null ? `llm ${formatMs(t.llm_ms)}` : '',
                      ]
                        .filter(Boolean)
                        .join(' · ')}
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>
          <div className="flex justify-end gap-1">
            <Button size="sm" variant="ghost" asChild>
              <a href={apiClient.getCallTranscriptUrl(call.id)} target="_blank" rel="noreferrer">
                Transcript
              </a>
            </Button>
            <Button size="sm" variant="ghost" onClick={onOpenInConsole}>
              Open in console
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
