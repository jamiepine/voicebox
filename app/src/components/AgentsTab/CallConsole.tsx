import { useQueryClient } from '@tanstack/react-query';
import {
  Bot,
  Loader2,
  Mic,
  PhoneCall,
  PhoneIncoming,
  PhoneOff,
  Send,
  Square,
  User,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { AgentTurnResponse, CallTurn, VoiceAgent, VoiceCall } from '@/lib/api/types';
import { useAudioRecording } from '@/lib/hooks/useAudioRecording';
import { agentKeys, useCall } from '@/lib/hooks/useVoiceAgents';
import { cn } from '@/lib/utils/cn';
import { usePlatform } from '@/platform/PlatformContext';
import { OUTCOME_META, OutcomeBadge, sentimentTone } from './shared';

const END_OUTCOMES = [
  'no_answer',
  'voicemail',
  'callback',
  'not_interested',
  'interested',
  'resolved',
  'unresolved',
] as const;

interface CallConsoleProps {
  agent: VoiceAgent;
  activeCall: VoiceCall | null;
  onCallChange: (callId: string | null) => void;
}

/**
 * Live conversation view for one call. The agent's side is produced by the
 * backend; the customer's side comes from this console (typed or spoken)
 * for `local` agents, or from the phone line for Twilio agents (read-only
 * here, refreshed every 2 s).
 */
export function CallConsole({ agent, activeCall, onCallChange }: CallConsoleProps) {
  const { toast } = useToast();
  const qc = useQueryClient();
  const platform = usePlatform();
  const callId = activeCall?.id ?? null;
  const isLive = activeCall?.status === 'in_progress';
  const isLocal = agent.provider === 'local';
  const { data: call } = useCall(callId, isLive);
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [inboundPhone, setInboundPhone] = useState('');
  const [inboundName, setInboundName] = useState('');
  // The desktop pill plays agent speech itself; in the browser nothing
  // would, so default inline playback to "on" outside Tauri.
  const [playHere, setPlayHere] = useState(() => !platform.metadata.isTauri);
  const playedRef = useRef<Set<string>>(new Set());
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const invalidate = useCallback(() => {
    qc.invalidateQueries({ queryKey: agentKeys.calls(agent.id) });
    qc.invalidateQueries({ queryKey: agentKeys.stats(agent.id) });
    qc.invalidateQueries({ queryKey: ['voice-agents', agent.id, 'contacts'] });
    qc.invalidateQueries({ queryKey: ['voice-tickets'] });
    if (callId) qc.invalidateQueries({ queryKey: agentKeys.call(callId) });
  }, [qc, agent.id, callId]);

  // Auto-scroll on new turns.
  const turnCount = call?.turns.length ?? 0;
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-run whenever a turn is added
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [turnCount]);

  // Inline playback: wait for a turn's generation to finish, then play it.
  useEffect(() => {
    if (!playHere || !call) return;
    const pending = call.turns.filter(
      (t) => t.role === 'agent' && t.generation_id && !playedRef.current.has(t.generation_id),
    );
    if (pending.length === 0) return;
    const turn = pending[0];
    const genId = turn.generation_id as string;
    playedRef.current.add(genId);
    const source = new EventSource(apiClient.getGenerationStatusUrl(genId));
    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as { status: string };
        if (data.status === 'completed') {
          source.close();
          audioRef.current?.pause();
          const audio = new Audio(apiClient.getAudioUrl(genId));
          audioRef.current = audio;
          audio.play().catch(() => undefined);
        } else if (data.status === 'failed' || data.status === 'not_found') {
          source.close();
        }
      } catch {
        source.close();
      }
    };
    source.onerror = () => source.close();
    return () => source.close();
  }, [call, playHere]);

  useEffect(
    () => () => {
      audioRef.current?.pause();
    },
    [],
  );

  const handleTurn = useCallback(
    (result: AgentTurnResponse) => {
      onCallChange(result.call_id);
      invalidate();
      if (result.ended) {
        const meta = result.outcome ? OUTCOME_META[result.outcome] : null;
        toast({
          title: `Call ended — ${meta?.label ?? result.outcome ?? 'done'}`,
          description: result.ticket_id ? 'A ticket was opened.' : undefined,
        });
      }
    },
    [onCallChange, invalidate, toast],
  );

  const run = useCallback(
    async (fn: () => Promise<AgentTurnResponse>) => {
      setBusy(true);
      try {
        handleTurn(await fn());
      } catch (err) {
        toast({
          title: 'Call failed',
          description: err instanceof Error ? err.message : String(err),
          variant: 'destructive',
        });
      } finally {
        setBusy(false);
      }
    },
    [handleTurn, toast],
  );

  const sendText = () => {
    const t = text.trim();
    if (!t || !callId) return;
    setText('');
    run(() => apiClient.sendCustomerTurn(callId, t));
  };

  const recording = useAudioRecording({
    maxDurationSeconds: 30,
    onRecordingComplete: (blob) => {
      if (!callId) return;
      run(() => apiClient.sendCustomerTurnAudio(callId, blob, agent.language));
    },
  });

  const endCall = async (outcome: string) => {
    if (!callId) return;
    setBusy(true);
    try {
      await apiClient.endCall(callId, outcome);
      invalidate();
      onCallChange(callId);
    } catch (err) {
      toast({
        title: 'Could not end call',
        description: err instanceof Error ? err.message : String(err),
        variant: 'destructive',
      });
    } finally {
      setBusy(false);
    }
  };

  const turns: CallTurn[] = call?.turns ?? activeCall?.turns ?? [];
  const isOutbound = agent.mode === 'outbound_sales';

  return (
    <div className="rounded-xl border border-border bg-card/40 flex flex-col min-h-[420px]">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2 text-sm font-medium">
          <PhoneCall className="h-4 w-4 text-accent" />
          Call console
        </div>
        {call && <OutcomeBadge outcome={call.outcome} />}
        {call && (
          <span className="text-xs text-muted-foreground">
            {call.direction} · {call.turn_count} turns
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPlayHere((v) => !v)}
            title={playHere ? 'Playing agent audio here' : 'Agent audio muted here'}
          >
            {playHere ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
          </Button>
          {isLive && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="destructive" size="sm" disabled={busy}>
                  <PhoneOff className="h-4 w-4" />
                  End call
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {END_OUTCOMES.map((o) => (
                  <DropdownMenuItem key={o} onClick={() => endCall(o)}>
                    {OUTCOME_META[o].label}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </div>

      {/* Transcript */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3 max-h-[480px]">
        {turns.length === 0 && (
          <div className="h-full min-h-[240px] flex flex-col items-center justify-center text-center text-muted-foreground gap-3">
            <Bot className="h-8 w-8 opacity-40" />
            <div className="text-sm max-w-sm">
              {isOutbound
                ? 'Place the next call from your contact list. The agent speaks first; type or hold the mic to answer as the customer.'
                : 'Start an inbound conversation. The agent greets the caller; type or hold the mic to speak as the caller.'}
            </div>
            <div className="flex items-center gap-2">
              {isOutbound ? (
                <Button
                  size="sm"
                  disabled={busy}
                  onClick={() => run(() => apiClient.startNextCall(agent.id))}
                >
                  {busy ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <PhoneCall className="h-4 w-4" />
                  )}
                  Call next contact
                </Button>
              ) : (
                <div className="flex items-center gap-2">
                  <Input
                    className="h-9 w-40"
                    placeholder="+44 7700 900123"
                    value={inboundPhone}
                    onChange={(e) => setInboundPhone(e.target.value)}
                  />
                  <Input
                    className="h-9 w-32"
                    placeholder="Caller name"
                    value={inboundName}
                    onChange={(e) => setInboundName(e.target.value)}
                  />
                  <Button
                    size="sm"
                    disabled={busy || !inboundPhone.trim()}
                    onClick={() =>
                      run(() =>
                        apiClient.startInboundCall(agent.id, {
                          phone: inboundPhone.trim(),
                          name: inboundName.trim() || null,
                        }),
                      )
                    }
                  >
                    {busy ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <PhoneIncoming className="h-4 w-4" />
                    )}
                    Answer call
                  </Button>
                </div>
              )}
            </div>
          </div>
        )}
        {turns.map((t) => (
          <TurnBubble key={t.id} turn={t} />
        ))}
        {busy && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground pl-9">
            <Loader2 className="h-3 w-3 animate-spin" /> agent is thinking…
          </div>
        )}
        {call?.summary && call.status !== 'in_progress' && (
          <div className="mt-4 rounded-lg border border-border bg-muted/30 p-3 text-sm">
            <div className="text-[11px] uppercase tracking-wider text-muted-foreground mb-1">
              Summary
            </div>
            {call.summary}
          </div>
        )}
      </div>

      {/* Composer */}
      {isLive && isLocal && (
        <div className="border-t border-border px-4 py-3 flex items-center gap-2">
          <Button
            variant={recording.isRecording ? 'destructive' : 'outline'}
            size="icon"
            title="Hold to speak as the customer"
            disabled={busy}
            onMouseDown={() => recording.startRecording()}
            onMouseUp={() => recording.stopRecording()}
            onMouseLeave={() => recording.isRecording && recording.stopRecording()}
            onTouchStart={() => recording.startRecording()}
            onTouchEnd={() => recording.stopRecording()}
          >
            {recording.isRecording ? <Square className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
          </Button>
          <Input
            placeholder="What the customer says…"
            value={text}
            disabled={busy}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendText();
              }
            }}
          />
          <Button size="icon" onClick={sendText} disabled={busy || !text.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      )}
      {isLive && !isLocal && (
        <div className="border-t border-border px-4 py-3 text-xs text-muted-foreground">
          Live on the phone line — the transcript updates as the caller speaks.
        </div>
      )}
      {!isLive && turns.length > 0 && (
        <div className="border-t border-border px-4 py-3 flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => onCallChange(null)}>
            Clear
          </Button>
          {isOutbound && (
            <Button
              size="sm"
              disabled={busy}
              onClick={() => run(() => apiClient.startNextCall(agent.id))}
            >
              <PhoneCall className="h-4 w-4" />
              Call next contact
            </Button>
          )}
        </div>
      )}
    </div>
  );
}

function TurnBubble({ turn }: { turn: CallTurn }) {
  const isAgent = turn.role === 'agent';
  return (
    <div className={cn('flex gap-2', isAgent ? 'justify-start' : 'justify-end')}>
      {isAgent && (
        <div className="h-7 w-7 rounded-full bg-accent/15 flex items-center justify-center shrink-0 mt-0.5">
          <Bot className="h-3.5 w-3.5 text-accent" />
        </div>
      )}
      <div
        className={cn(
          'max-w-[75%] rounded-2xl px-3.5 py-2 text-sm leading-relaxed',
          isAgent ? 'bg-muted/50 rounded-tl-sm' : 'bg-accent/15 rounded-tr-sm',
        )}
      >
        {turn.text}
        {!isAgent && turn.sentiment != null && Math.abs(turn.sentiment) >= 0.3 && (
          <div className={cn('text-[10px] mt-1', sentimentTone(turn.sentiment))}>
            {turn.sentiment <= -0.3 ? 'sounds upset' : 'sounds positive'}
          </div>
        )}
      </div>
      {!isAgent && (
        <div className="h-7 w-7 rounded-full bg-muted flex items-center justify-center shrink-0 mt-0.5">
          <User className="h-3.5 w-3.5 text-muted-foreground" />
        </div>
      )}
    </div>
  );
}
