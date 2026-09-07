import { Headset, type LucideIcon, PhoneOutgoing, Wrench } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import type { CallOutcome, VoiceAgentMode, VoiceAgentStatus } from '@/lib/api/types';
import { cn } from '@/lib/utils/cn';

export const MODE_META: Record<VoiceAgentMode, { label: string; icon: LucideIcon; blurb: string }> =
  {
    outbound_sales: {
      label: 'Outbound sales',
      icon: PhoneOutgoing,
      blurb: 'Dials a contact list, pitches, handles objections, books next steps.',
    },
    customer_service: {
      label: 'Customer service',
      icon: Headset,
      blurb: 'Answers inbound calls from your knowledge base, logs what it cannot solve.',
    },
    support: {
      label: 'Support',
      icon: Wrench,
      blurb: 'Troubleshoots step by step, opens tickets, hands off to a person when needed.',
    },
  };

export const OUTCOME_META: Record<
  CallOutcome,
  { label: string; tone: 'good' | 'bad' | 'neutral' | 'warn' }
> = {
  interested: { label: 'Interested', tone: 'good' },
  resolved: { label: 'Resolved', tone: 'good' },
  callback: { label: 'Callback', tone: 'warn' },
  ticket_created: { label: 'Ticket', tone: 'warn' },
  handoff: { label: 'Handoff', tone: 'warn' },
  max_turns: { label: 'Turn limit', tone: 'warn' },
  not_interested: { label: 'Not interested', tone: 'neutral' },
  unresolved: { label: 'Unresolved', tone: 'neutral' },
  no_answer: { label: 'No answer', tone: 'neutral' },
  voicemail: { label: 'Voicemail', tone: 'neutral' },
  voicemail_left: { label: 'Voicemail left', tone: 'neutral' },
  opt_out: { label: 'Opted out', tone: 'bad' },
  error: { label: 'Error', tone: 'bad' },
};

const TONE_CLASS = {
  good: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
  warn: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
  bad: 'border-red-500/40 bg-red-500/10 text-red-300',
  neutral: 'border-border bg-muted/40 text-muted-foreground',
};

export function OutcomeBadge({
  outcome,
  className,
}: {
  outcome: CallOutcome | null;
  className?: string;
}) {
  if (!outcome) {
    return (
      <Badge variant="outline" className={cn('border-accent/40 text-accent', className)}>
        Live
      </Badge>
    );
  }
  const meta = OUTCOME_META[outcome] ?? { label: outcome, tone: 'neutral' as const };
  return (
    <Badge variant="outline" className={cn(TONE_CLASS[meta.tone], className)}>
      {meta.label}
    </Badge>
  );
}

export function ScoreBadge({ score }: { score: number | null | undefined }) {
  if (score == null) return null;
  const tone = score >= 70 ? 'good' : score >= 40 ? 'warn' : 'bad';
  return (
    <Badge variant="outline" className={TONE_CLASS[tone]} title="Goal-achievement score (0-100)">
      {score}
    </Badge>
  );
}

export function StatusDot({ status, running }: { status: VoiceAgentStatus; running: boolean }) {
  const color = running
    ? 'bg-emerald-400 animate-pulse'
    : status === 'active'
      ? 'bg-emerald-400/60'
      : status === 'paused'
        ? 'bg-amber-400'
        : status === 'completed'
          ? 'bg-sky-400'
          : 'bg-muted-foreground/40';
  return <span className={cn('inline-block h-2 w-2 rounded-full shrink-0', color)} />;
}

export function statusLabel(status: VoiceAgentStatus, running: boolean): string {
  if (running) return 'Dialing';
  return { draft: 'Draft', active: 'Active', paused: 'Paused', completed: 'Completed' }[status];
}

export function contactStatusLabel(status: string): string {
  return status.replace(/_/g, ' ');
}

export function sentimentTone(s: number | null | undefined): string {
  if (s == null) return 'text-muted-foreground';
  if (s <= -0.3) return 'text-red-300';
  if (s >= 0.3) return 'text-emerald-300';
  return 'text-muted-foreground';
}

export function formatMs(ms: number | null | undefined): string {
  if (ms == null) return '';
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

/** Encode Float32 PCM frames as a 16-bit mono WAV blob. */
export function pcmToWav(frames: Float32Array[], sampleRate: number): Blob {
  const length = frames.reduce((n, f) => n + f.length, 0);
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, length * 2, true);
  let offset = 44;
  for (const frame of frames) {
    for (let i = 0; i < frame.length; i++) {
      const s = Math.max(-1, Math.min(1, frame[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([buffer], { type: 'audio/wav' });
}
