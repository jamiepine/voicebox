import { CalendarCheck, Download, MessageSquare } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { apiClient } from '@/lib/api/client';
import type { VoiceAgent } from '@/lib/api/types';
import {
  useAgentMessages,
  useAppointments,
  useContacts,
  useUpdateAppointment,
} from '@/lib/hooks/useVoiceAgents';
import { cn } from '@/lib/utils/cn';
import { formatAbsoluteDate, formatDate } from '@/lib/utils/format';

export function AppointmentsPanel({ agent }: { agent: VoiceAgent }) {
  const { data: appointments, isLoading } = useAppointments(agent.id);
  const { data: contacts } = useContacts(agent.id);
  const { data: messages } = useAgentMessages(agent.id);
  const update = useUpdateAppointment(agent.id);
  const names = new Map((contacts?.contacts ?? []).map((c) => [c.id, c]));

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Appointments
        </h3>
        {isLoading ? (
          <div className="text-sm text-muted-foreground">Loading…</div>
        ) : (appointments ?? []).length === 0 ? (
          <div className="text-sm text-muted-foreground py-8 text-center rounded-lg border border-dashed border-border">
            <CalendarCheck className="h-5 w-5 mx-auto mb-2 opacity-40" />
            Nothing booked yet. The agent books through the <code>book_appointment</code> tool.
          </div>
        ) : (
          (appointments ?? []).map((a) => {
            const c = names.get(a.contact_id);
            return (
              <div
                key={a.id}
                className={cn(
                  'rounded-lg border border-border p-3 flex items-center gap-3',
                  a.status === 'cancelled' && 'opacity-60',
                )}
              >
                <div className="min-w-0 flex-1">
                  <div className="font-medium">{formatAbsoluteDate(`${a.starts_at}Z`)}</div>
                  <div className="text-xs text-muted-foreground">
                    {c ? `${c.name} · ${c.phone}` : a.contact_id}
                    {a.timezone ? ` · ${a.timezone}` : ''}
                    {a.notes ? ` · ${a.notes}` : ''}
                  </div>
                </div>
                <Select
                  value={a.status}
                  onValueChange={(v) => update.mutate({ appointmentId: a.id, data: { status: v } })}
                >
                  <SelectTrigger className="h-8 w-[130px] text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="booked">Booked</SelectItem>
                    <SelectItem value="confirmed">Confirmed</SelectItem>
                    <SelectItem value="completed">Completed</SelectItem>
                    <SelectItem value="cancelled">Cancelled</SelectItem>
                  </SelectContent>
                </Select>
                <Button variant="ghost" size="sm" asChild title="Download .ics">
                  <a href={apiClient.getAppointmentIcsUrl(a.id)} target="_blank" rel="noreferrer">
                    <Download className="h-4 w-4" />
                  </a>
                </Button>
              </div>
            );
          })
        )}
      </div>

      <div className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Text messages
        </h3>
        {(messages ?? []).length === 0 ? (
          <div className="text-sm text-muted-foreground py-6 text-center rounded-lg border border-dashed border-border">
            <MessageSquare className="h-5 w-5 mx-auto mb-2 opacity-40" />
            No messages. SMS follow-ups and the <code>send_sms</code> tool need the Twilio provider.
          </div>
        ) : (
          (messages ?? []).map((m) => (
            <div key={m.id} className="rounded-lg border border-border p-3 text-sm flex gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-xs text-muted-foreground">
                  {m.to_number} · {formatDate(m.created_at)}
                </div>
                <div>{m.body}</div>
                {m.error && <div className="text-xs text-red-300 mt-1">{m.error}</div>}
              </div>
              <Badge variant="outline" className="h-6 self-start">
                {m.status.replace(/_/g, ' ')}
              </Badge>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
