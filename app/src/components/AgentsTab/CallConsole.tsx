import { useQueryClient } from '@tanstack/react-query';
import {
  Bot,
  Hand,
  Headphones,
  Loader2,
  Mic,
  PhoneCall,
  PhoneIncoming,
  PhoneOff,
  Radio,
  Send,
  Square,
  User,
  Volume2,
  VolumeX,
  Wrench,
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
import { formatMs, OUTCOME_META, OutcomeBadge, pcmToWav, sentimentTone } from './shared';

const END_OUTCOMES = [
  'no_answer',
  'voicemail',
  'callback',
  'not_interested',
  'interested',
  'resolved',
  'unresolved',
] as const;

// Client-side endpointing. Tuned for a headset on a quiet desk; the
// threshold doubles while the agent is talking so speaker bleed doesn't
// trigger a false barge-in (browser echo cancellation does the rest).
const VAD_THRESHOLD = 0.012;
const VAD_PREROLL_FRAMES = 6; // ~550 ms at 4096-sample frames / 44.1 kHz
const SPEECH_START_MS = 200;
const SPEECH_END_MS = 750;
const BARGE_IN_MS = 300;
const MAX_UTTERANCE_MS = 30000;

interface CallConsoleProps {
  agent: VoiceAgent;
  activeCall: VoiceCall | null;
  onCallChange: (callId: string | null) => void;
}

type QueueItem = { generationId: string; label: string };

/**
 * Live conversation view for one call. The agent's side is produced by the
 * backend; the customer's side comes from this console (typed, push-to-talk,
 * or hands-free live mode) for `local` agents, or from the phone line for
 * Twilio agents (read-only here, refreshed every 2 s).
 */
export function CallConsole({ agent, activeCall, onCallChange }: CallConsoleProps) {
  const { toast } = useToast();
  const qc = useQueryClient();
  const platform = usePlatform();
  const callId = activeCall?.id ?? null;
  const isLive = activeCall?.status === 'in_progress';
  const isLocal = agent.provider === 'local';
  const { data: call, refetch: refetchCall } = useCall(callId, isLive);
  const [text, setText] = useState('');
  const [operatorText, setOperatorText] = useState('');
  const [busy, setBusy] = useState(false);
  const [inboundPhone, setInboundPhone] = useState('');
  const [inboundName, setInboundName] = useState('');
  const [liveMode, setLiveMode] = useState(false);
  const [liveState, setLiveState] = useState<
    'idle' | 'listening' | 'speaking' | 'thinking' | 'playing'
  >('idle');
  const [micLevel, setMicLevel] = useState(0);
  const [takeover, setTakeover] = useState(false);
  // The desktop pill plays agent speech itself; in the browser nothing
  // would, so default inline playback to "on" outside Tauri. Live mode
  // always plays here (it needs to know when playback ends).
  const [playHere, setPlayHere] = useState(() => !platform.metadata.isTauri);
  const clientPlays = playHere || liveMode;

  const playedRef = useRef<Set<string>>(new Set());
  const queueRef = useRef<QueueItem[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playingRef = useRef<QueueItem | null>(null);
  const currentTurnRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const eventsRef = useRef<EventSource | null>(null);

  const invalidate = useCallback(() => {
    qc.invalidateQueries({ queryKey: agentKeys.calls(agent.id) });
    qc.invalidateQueries({ queryKey: agentKeys.stats(agent.id) });
    qc.invalidateQueries({ queryKey: ['voice-agents', agent.id, 'contacts'] });
    qc.invalidateQueries({ queryKey: ['voice-agents', agent.id, 'appointments'] });
    qc.invalidateQueries({ queryKey: ['voice-tickets'] });
    if (callId) qc.invalidateQueries({ queryKey: agentKeys.call(callId) });
  }, [qc, agent.id, callId]);

  // ── Playback queue ────────────────────────────────────────────────
  const stopPlayback = useCallback(() => {
    queueRef.current = [];
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
      audioRef.current = null;
    }
    playingRef.current = null;
  }, []);

  const pumpQueue = useCallback(() => {
    if (playingRef.current || queueRef.current.length === 0) return;
    const item = queueRef.current.shift() as QueueItem;
    playingRef.current = item;
    setLiveState((s) => (liveMode ? 'playing' : s));
    const finish = () => {
      if (playingRef.current === item) playingRef.current = null;
      if (queueRef.current.length === 0) setLiveState((s) => (s === 'playing' ? 'listening' : s));
      pumpQueue();
    };
    const source = new EventSource(apiClient.getGenerationStatusUrl(item.generationId));
    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as { status: string };
        if (data.status === 'completed') {
          source.close();
          if (playingRef.current !== item) return; // interrupted meanwhile
          const audio = new Audio(apiClient.getAudioUrl(item.generationId));
          audioRef.current = audio;
          audio.onended = finish;
          audio.onerror = finish;
          audio.play().catch(finish);
        } else if (data.status === 'failed' || data.status === 'not_found') {
          source.close();
          finish();
        }
      } catch {
        source.close();
        finish();
      }
    };
    source.onerror = () => {
      source.close();
      finish();
    };
  }, [liveMode]);

  const enqueue = useCallback(
    (ids: string[], label: string, front = false) => {
      if (!clientPlays) return;
      const items = ids
        .filter((id) => !playedRef.current.has(id))
        .map((generationId) => {
          playedRef.current.add(generationId);
          return { generationId, label };
        });
      if (items.length === 0) return;
      queueRef.current = front ? [...items, ...queueRef.current] : [...queueRef.current, ...items];
      pumpQueue();
    },
    [clientPlays, pumpQueue],
  );

  // ── SSE events for the live call ──────────────────────────────────
  useEffect(() => {
    if (!callId || !isLive) return;
    const source = new EventSource(apiClient.getCallEventsUrl(callId));
    eventsRef.current = source;
    const refresh = () => {
      refetchCall();
    };
    source.addEventListener('customer_turn', refresh);
    source.addEventListener('filler', (e) => {
      const data = JSON.parse((e as MessageEvent).data);
      enqueue([data.generation_id], 'filler', true);
    });
    source.addEventListener('agent_turn', (e) => {
      const data = JSON.parse((e as MessageEvent).data);
      currentTurnRef.current = data.turn_id;
      enqueue(data.generation_ids ?? [], 'agent');
      setLiveState((s) =>
        liveMode && !data.ended ? (queueRef.current.length ? 'playing' : 'listening') : s,
      );
      refresh();
    });
    source.addEventListener('tool_call', refresh);
    source.addEventListener('ai_paused', refresh);
    source.addEventListener('awaiting_operator', () => {
      setLiveState((s) => (liveMode ? 'listening' : s));
      refresh();
    });
    source.addEventListener('ended', () => {
      refresh();
      invalidate();
    });
    return () => {
      source.close();
      eventsRef.current = null;
    };
  }, [callId, isLive, enqueue, refetchCall, invalidate, liveMode]);

  // Non-live playback (typed turns without SSE race): enqueue from turn data.
  useEffect(() => {
    if (!clientPlays || !call || liveMode) return;
    for (const t of call.turns) {
      if (t.role === 'agent' && t.generation_ids?.length) enqueue(t.generation_ids, 'agent');
      else if (t.role === 'agent' && t.generation_id) enqueue([t.generation_id], 'agent');
    }
  }, [call, clientPlays, liveMode, enqueue]);

  // Auto-scroll on new turns.
  const turnCount = call?.turns.length ?? 0;
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-run whenever a turn is added
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [turnCount]);

  useEffect(() => () => stopPlayback(), [stopPlayback]);

  // ── Turn plumbing ─────────────────────────────────────────────────
  const handleTurn = useCallback(
    (result: AgentTurnResponse) => {
      onCallChange(result.call_id);
      invalidate();
      if (result.ended) {
        const meta = result.outcome ? OUTCOME_META[result.outcome] : null;
        toast({
          title: `Call ended — ${meta?.label ?? result.outcome ?? 'done'}`,
          description: result.ticket_id
            ? 'A ticket was opened.'
            : result.appointment_id
              ? 'An appointment was booked.'
              : undefined,
        });
      }
    },
    [onCallChange, invalidate, toast],
  );

  const run = useCallback(
    async (fn: () => Promise<AgentTurnResponse>) => {
      setBusy(true);
      setLiveState((s) => (liveMode ? 'thinking' : s));
      try {
        handleTurn(await fn());
      } catch (err) {
        toast({
          title: 'Call failed',
          description: err instanceof Error ? err.message : String(err),
          variant: 'destructive',
        });
        setLiveState((s) => (liveMode ? 'listening' : s));
      } finally {
        setBusy(false);
      }
    },
    [handleTurn, toast, liveMode],
  );

  const sendText = () => {
    const t = text.trim();
    if (!t || !callId) return;
    setText('');
    run(() => apiClient.sendCustomerTurn(callId, t, clientPlays));
  };

  const sendOperator = () => {
    const t = operatorText.trim();
    if (!t || !callId) return;
    setOperatorText('');
    run(() => apiClient.agentSay(callId, t, clientPlays));
  };

  const recording = useAudioRecording({
    maxDurationSeconds: 30,
    onRecordingComplete: (blob) => {
      if (!callId) return;
      run(() => apiClient.sendCustomerTurnAudio(callId, blob, agent.language, clientPlays));
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

  const toggleTakeover = async () => {
    if (!callId) return;
    try {
      await apiClient.setCallAiPaused(callId, !takeover);
      setTakeover(!takeover);
      refetchCall();
    } catch (err) {
      toast({
        title: 'Could not change take-over',
        description: String(err),
        variant: 'destructive',
      });
    }
  };

  // ── Live mode: mic + VAD + barge-in ───────────────────────────────
  const liveRef = useRef<{
    ctx: AudioContext;
    stream: MediaStream;
    node: ScriptProcessorNode;
  } | null>(null);
  const bargeRef = useRef<() => void>(() => undefined);
  bargeRef.current = () => {
    const turnId = currentTurnRef.current;
    stopPlayback();
    if (callId && turnId) apiClient.interruptCall(callId, turnId).catch(() => undefined);
  };
  const sendUtteranceRef = useRef<(blob: Blob) => void>(() => undefined);
  sendUtteranceRef.current = (blob: Blob) => {
    if (!callId) return;
    run(() => apiClient.sendCustomerTurnAudio(callId, blob, agent.language, true));
  };

  const stopLive = useCallback(() => {
    const live = liveRef.current;
    if (live) {
      live.node.disconnect();
      live.ctx.close().catch(() => undefined);
      for (const track of live.stream.getTracks()) track.stop();
      liveRef.current = null;
    }
    setLiveState('idle');
    setMicLevel(0);
  }, []);

  const startLive = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      const ctx = new AudioContext();
      const source = ctx.createMediaStreamSource(stream);
      const node = ctx.createScriptProcessor(4096, 1, 1);
      const preroll: Float32Array[] = [];
      let frames: Float32Array[] = [];
      let speechMs = 0;
      let silenceMs = 0;
      let inSpeech = false;
      let utteranceMs = 0;
      const frameMs = (4096 / ctx.sampleRate) * 1000;
      node.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        let sum = 0;
        for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
        const rms = Math.sqrt(sum / input.length);
        setMicLevel(rms);
        const playing = playingRef.current !== null;
        const threshold = playing ? VAD_THRESHOLD * 2.2 : VAD_THRESHOLD;
        const voiced = rms > threshold;
        if (!inSpeech) {
          preroll.push(new Float32Array(input));
          if (preroll.length > VAD_PREROLL_FRAMES) preroll.shift();
          if (voiced) {
            speechMs += frameMs;
            if (speechMs >= (playing ? BARGE_IN_MS : SPEECH_START_MS)) {
              if (playing) bargeRef.current();
              inSpeech = true;
              frames = [...preroll];
              utteranceMs = speechMs;
              silenceMs = 0;
              setLiveState('speaking');
            }
          } else {
            speechMs = 0;
          }
          return;
        }
        frames.push(new Float32Array(input));
        utteranceMs += frameMs;
        if (voiced) silenceMs = 0;
        else silenceMs += frameMs;
        if (silenceMs >= SPEECH_END_MS || utteranceMs >= MAX_UTTERANCE_MS) {
          const blob = pcmToWav(frames, ctx.sampleRate);
          frames = [];
          inSpeech = false;
          speechMs = 0;
          silenceMs = 0;
          setLiveState('thinking');
          sendUtteranceRef.current(blob);
        }
      };
      source.connect(node);
      node.connect(ctx.destination);
      liveRef.current = { ctx, stream, node };
      setLiveState('listening');
    } catch (err) {
      toast({
        title: 'Microphone unavailable',
        description: err instanceof Error ? err.message : String(err),
        variant: 'destructive',
      });
      setLiveMode(false);
    }
  }, [toast]);

  useEffect(() => {
    if (liveMode && isLive) startLive();
    else stopLive();
    return () => stopLive();
  }, [liveMode, isLive, startLive, stopLive]);

  useEffect(() => {
    if (!isLive) setTakeover(false);
  }, [isLive]);

  const turns: CallTurn[] = call?.turns ?? activeCall?.turns ?? [];
  const isOutbound = agent.mode === 'outbound_sales';
  const aiPaused = call?.ai_paused ?? false;

  return (
    <div className="rounded-xl border border-border bg-card/40 flex flex-col min-h-[440px]">
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
            {call.variant ? ` · variant ${call.variant}` : ''}
          </span>
        )}
        {liveMode && isLive && <LiveIndicator state={liveState} level={micLevel} />}
        <div className="ml-auto flex items-center gap-2">
          {isLive && isLocal && (
            <Button
              variant={liveMode ? 'default' : 'outline'}
              size="sm"
              onClick={() => setLiveMode((v) => !v)}
              title="Hands-free: mic stays open, the agent is interrupted when you speak"
            >
              <Radio className="h-4 w-4" />
              {liveMode ? 'Live' : 'Go live'}
            </Button>
          )}
          {isLive && (
            <Button
              variant={takeover || aiPaused ? 'default' : 'ghost'}
              size="sm"
              onClick={toggleTakeover}
              title={
                aiPaused ? 'Hand the call back to the AI' : 'Pause the AI and speak as the agent'
              }
            >
              <Hand className="h-4 w-4" />
              {aiPaused ? 'Hand back' : 'Take over'}
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPlayHere((v) => !v)}
            disabled={liveMode}
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
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3 max-h-[520px]">
        {turns.length === 0 && (
          <div className="h-full min-h-[240px] flex flex-col items-center justify-center text-center text-muted-foreground gap-3">
            <Bot className="h-8 w-8 opacity-40" />
            <div className="text-sm max-w-sm">
              {isOutbound
                ? 'Place the next call from your contact list. The agent speaks first; answer as the customer by typing, holding the mic, or going live.'
                : 'Start an inbound conversation. The agent greets the caller; answer as the caller by typing, holding the mic, or going live.'}
            </div>
            <div className="flex items-center gap-2">
              {isOutbound ? (
                <Button
                  size="sm"
                  disabled={busy}
                  onClick={() => run(() => apiClient.startNextCall(agent.id, { clientPlays }))}
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
                        apiClient.startInboundCall(
                          agent.id,
                          { phone: inboundPhone.trim(), name: inboundName.trim() || null },
                          clientPlays,
                        ),
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
        {call && call.status !== 'in_progress' && (call.summary || call.score != null) && (
          <div className="mt-4 rounded-lg border border-border bg-muted/30 p-3 text-sm space-y-2">
            {call.summary && (
              <div>
                <div className="text-[11px] uppercase tracking-wider text-muted-foreground mb-1">
                  Summary
                </div>
                {call.summary}
              </div>
            )}
            {call.score != null && (
              <div className="text-xs text-muted-foreground">
                Score <span className="text-foreground font-medium">{call.score}/100</span>
                {call.score_reason ? ` — ${call.score_reason}` : ''}
              </div>
            )}
            {call.analysis && Object.keys(call.analysis).length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(call.analysis).map(([k, v]) => (
                  <span
                    key={k}
                    className="rounded-full border border-border px-2 py-0.5 text-[11px]"
                  >
                    {k}: <span className="text-foreground">{v == null ? '—' : String(v)}</span>
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Composer */}
      {isLive && isLocal && (aiPaused || takeover) && (
        <div className="border-t border-amber-500/30 bg-amber-500/5 px-4 py-3 flex items-center gap-2">
          <Headphones className="h-4 w-4 text-amber-300 shrink-0" />
          <Input
            placeholder="Speak as the agent…"
            value={operatorText}
            disabled={busy}
            onChange={(e) => setOperatorText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendOperator();
              }
            }}
          />
          <Button size="icon" onClick={sendOperator} disabled={busy || !operatorText.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      )}
      {isLive && isLocal && (
        <div className="border-t border-border px-4 py-3 flex items-center gap-2">
          <Button
            variant={recording.isRecording ? 'destructive' : 'outline'}
            size="icon"
            title="Hold to speak as the customer"
            disabled={busy || liveMode}
            onMouseDown={() => recording.startRecording()}
            onMouseUp={() => recording.stopRecording()}
            onMouseLeave={() => recording.isRecording && recording.stopRecording()}
            onTouchStart={() => recording.startRecording()}
            onTouchEnd={() => recording.stopRecording()}
          >
            {recording.isRecording ? <Square className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
          </Button>
          <Input
            placeholder={liveMode ? 'Live — just talk. Or type here.' : 'What the customer says…'}
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
          Live on the phone line — the transcript updates as the caller speaks. Take over to speak
          as the agent.
        </div>
      )}
      {!isLive && turns.length > 0 && (
        <div className="border-t border-border px-4 py-3 flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => onCallChange(null)}>
            Clear
          </Button>
          {call && (
            <Button variant="ghost" size="sm" asChild>
              <a href={apiClient.getCallTranscriptUrl(call.id)} target="_blank" rel="noreferrer">
                Transcript
              </a>
            </Button>
          )}
          {isOutbound && (
            <Button
              size="sm"
              disabled={busy}
              onClick={() => run(() => apiClient.startNextCall(agent.id, { clientPlays }))}
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

function LiveIndicator({ state, level }: { state: string; level: number }) {
  const label =
    state === 'listening'
      ? 'Listening'
      : state === 'speaking'
        ? 'Hearing you'
        : state === 'thinking'
          ? 'Thinking'
          : state === 'playing'
            ? 'Agent speaking'
            : 'Live';
  const bars = Math.min(5, Math.round(level / 0.01));
  return (
    <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
      <span
        className={cn(
          'h-2 w-2 rounded-full',
          state === 'speaking'
            ? 'bg-emerald-400 animate-pulse'
            : state === 'playing'
              ? 'bg-accent animate-pulse'
              : 'bg-muted-foreground/60',
        )}
      />
      {label}
      <span className="flex items-end gap-0.5 h-3 ml-1">
        {[0, 1, 2, 3, 4].map((i) => (
          <span
            key={i}
            className={cn(
              'w-0.5 rounded-sm',
              i < bars ? 'bg-emerald-400' : 'bg-muted-foreground/30',
            )}
            style={{ height: `${4 + i * 2}px` }}
          />
        ))}
      </span>
    </div>
  );
}

function TurnBubble({ turn }: { turn: CallTurn }) {
  if (turn.role === 'tool') {
    const meta = (turn.meta ?? {}) as { ok?: boolean; args?: Record<string, unknown>; ms?: number };
    return (
      <div className="flex justify-start pl-9">
        <div
          className={cn(
            'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px]',
            meta.ok === false
              ? 'border-amber-500/40 text-amber-300'
              : 'border-border text-muted-foreground',
          )}
          title={turn.text}
        >
          <Wrench className="h-3 w-3" />
          {turn.tool_name}
          {meta.args && Object.keys(meta.args).length > 0 && (
            <span className="opacity-70">({Object.values(meta.args).map(String).join(', ')})</span>
          )}
          {meta.ms != null && <span className="opacity-60">· {formatMs(meta.ms)}</span>}
        </div>
      </div>
    );
  }
  const isAgent = turn.role === 'agent';
  const latency = [
    turn.stt_ms != null ? `stt ${formatMs(turn.stt_ms)}` : '',
    turn.llm_ms != null ? `llm ${formatMs(turn.llm_ms)}` : '',
  ]
    .filter(Boolean)
    .join(' · ');
  return (
    <div className={cn('flex gap-2', isAgent ? 'justify-start' : 'justify-end')}>
      {isAgent && (
        <div
          className={cn(
            'h-7 w-7 rounded-full flex items-center justify-center shrink-0 mt-0.5',
            turn.source === 'operator' ? 'bg-amber-500/15' : 'bg-accent/15',
          )}
          title={
            turn.source === 'operator'
              ? 'Supervisor'
              : turn.source === 'system'
                ? 'Scripted'
                : 'Model'
          }
        >
          {turn.source === 'operator' ? (
            <Headphones className="h-3.5 w-3.5 text-amber-300" />
          ) : (
            <Bot className="h-3.5 w-3.5 text-accent" />
          )}
        </div>
      )}
      <div
        className={cn(
          'max-w-[75%] rounded-2xl px-3.5 py-2 text-sm leading-relaxed',
          isAgent ? 'bg-muted/50 rounded-tl-sm' : 'bg-accent/15 rounded-tr-sm',
          turn.interrupted && 'opacity-70',
        )}
      >
        {turn.text}
        {turn.interrupted && (
          <span className="text-[10px] text-muted-foreground ml-1">(interrupted)</span>
        )}
        {!isAgent && turn.sentiment != null && Math.abs(turn.sentiment) >= 0.3 && (
          <div className={cn('text-[10px] mt-1', sentimentTone(turn.sentiment))}>
            {turn.sentiment <= -0.3 ? 'sounds upset' : 'sounds positive'}
          </div>
        )}
        {isAgent && latency && (
          <div className="text-[10px] text-muted-foreground/70 mt-1">{latency}</div>
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
