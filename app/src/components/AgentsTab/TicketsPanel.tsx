import { Ticket as TicketIcon } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { Ticket } from '@/lib/api/types';
import { useTickets, useUpdateTicket } from '@/lib/hooks/useVoiceAgents';
import { cn } from '@/lib/utils/cn';
import { formatDate } from '@/lib/utils/format';

const PRIORITY_CLASS: Record<Ticket['priority'], string> = {
  low: 'border-border text-muted-foreground',
  normal: 'border-sky-500/40 text-sky-300',
  high: 'border-amber-500/40 text-amber-300',
  urgent: 'border-red-500/40 text-red-300',
};

const KIND_LABEL: Record<Ticket['kind'], string> = {
  support: 'Support',
  handoff: 'Handoff',
  callback: 'Callback',
  sales_lead: 'Sales lead',
};

export function TicketsPanel({ agentId }: { agentId: string }) {
  const { data, isLoading } = useTickets(agentId);
  const update = useUpdateTicket();
  const tickets = data?.tickets ?? [];

  if (isLoading) return <div className="text-sm text-muted-foreground">Loading…</div>;
  if (tickets.length === 0) {
    return (
      <div className="text-sm text-muted-foreground py-8 text-center rounded-lg border border-dashed border-border">
        <TicketIcon className="h-5 w-5 mx-auto mb-2 opacity-40" />
        No tickets. They appear when the agent hands off, hits a limit, or books a lead.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {tickets.map((t) => (
        <div
          key={t.id}
          className={cn(
            'rounded-lg border border-border p-3',
            (t.status === 'resolved' || t.status === 'closed') && 'opacity-60',
          )}
        >
          <div className="flex items-center gap-2">
            <Badge variant="outline" className={PRIORITY_CLASS[t.priority]}>
              {t.priority}
            </Badge>
            <Badge variant="outline">{KIND_LABEL[t.kind]}</Badge>
            <span className="font-medium truncate">{t.subject}</span>
            <span className="text-xs text-muted-foreground ml-auto shrink-0">
              {formatDate(t.created_at)}
            </span>
            <Select
              value={t.status}
              onValueChange={(v) => update.mutate({ ticketId: t.id, data: { status: v } })}
            >
              <SelectTrigger className="h-8 w-[130px] text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="open">Open</SelectItem>
                <SelectItem value="in_progress">In progress</SelectItem>
                <SelectItem value="resolved">Resolved</SelectItem>
                <SelectItem value="closed">Closed</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {t.description && (
            <details className="mt-2">
              <summary className="text-xs text-muted-foreground cursor-pointer">Transcript</summary>
              <pre className="text-xs text-muted-foreground whitespace-pre-wrap mt-1 font-sans">
                {t.description}
              </pre>
            </details>
          )}
        </div>
      ))}
    </div>
  );
}
