'use client';

import { AnimatePresence, motion } from 'framer-motion';
import {
  AudioLines,
  Box,
  ChevronDown,
  CircleDot,
  Copy,
  FileAudio,
  Mic,
  Download,
  Play,
  Settings,
  Sparkles,
  Subtitles,
  Users,
  Volume2,
  Wand2,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

// ─── Sidebar (matches ControlUI exactly) ───────────────────────────────────

const SIDEBAR_ITEMS = [
  { icon: Volume2, label: 'Generate' },
  { icon: AudioLines, label: 'Stories' },
  { icon: Mic, label: 'Captures', active: true },
  { icon: Users, label: 'Voices' },
  { icon: Wand2, label: 'Effects' },
  { icon: Box, label: 'Models' },
  { icon: Settings, label: 'Settings' },
];

function Sidebar() {
  return (
    <div className="hidden md:flex w-16 shrink-0 border-r border-app-line bg-sidebar flex-col items-center py-4 gap-4">
      {/* Logo */}
      <div className="mb-1">
        <div
          className="w-9 h-9 rounded-lg overflow-hidden"
          style={{
            filter:
              'drop-shadow(0 0 6px hsl(43 50% 45% / 0.5)) drop-shadow(0 0 14px hsl(43 50% 45% / 0.35))',
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/voicebox-logo-app.webp"
            alt=""
            className="w-full h-full object-contain"
          />
        </div>
      </div>

      {/* Nav items */}
      <div className="flex flex-col gap-2">
        {SIDEBAR_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <div
              key={item.label}
              className={`w-9 h-9 rounded-full flex items-center justify-center transition-all duration-200 ${
                item.active
                  ? 'bg-white/[0.07] text-foreground shadow-lg backdrop-blur-sm border border-white/[0.08]'
                  : 'text-muted-foreground/60'
              }`}
            >
              <Icon className="h-4 w-4" />
            </div>
          );
        })}
      </div>

      {/* Version */}
      <div className="mt-auto text-[8px] text-muted-foreground/40">v0.5.0</div>
    </div>
  );
}

// ─── FakeWaveform (ported from CapturesTab.tsx) ────────────────────────────

function FakeWaveform({
  seed,
  active,
  className,
}: {
  seed: number;
  active?: boolean;
  className?: string;
}) {
  const bars = useMemo(() => {
    return Array.from({ length: 72 }).map((_, i) => {
      const h =
        28 +
        Math.sin(i * 0.35 + seed) * 22 +
        Math.cos(i * 0.81 + seed * 2) * 14 +
        Math.sin(i * 1.7 + seed * 3) * 8;
      return Math.max(6, Math.min(96, h));
    });
  }, [seed]);

  return (
    <div className={`flex items-center gap-[2px] h-10 ${className ?? ''}`}>
      {bars.map((h, i) => (
        <div
          key={i}
          className="w-[3px] rounded-full"
          style={{
            height: `${h}%`,
            backgroundColor: active ? 'hsl(43 50% 50% / 0.85)' : 'hsl(var(--foreground) / 0.25)',
          }}
        />
      ))}
    </div>
  );
}

// ─── Data ───────────────────────────────────────────────────────────────────

type Capture = {
  id: string;
  seed: number;
  transcriptRaw: string;
  transcriptRefined: string;
  durationMs: number;
  ago: string;
  createdAtLabel: string;
  source: 'dictation' | 'recording' | 'file';
  sttModel: string;
  language?: string;
};

const CAPTURES: Capture[] = [
  {
    id: 'c1',
    seed: 11,
    transcriptRaw:
      "okay so the pitch for voicebox is basically this it's a local first voice studio everything runs on your machine you clone voices from a few seconds of audio generate speech across seven TTS engines and now with the captures tab you can dictate into any app no cloud no API keys no per character fees your voice data never leaves your device privacy isn't a feature here it's the architecture",
    transcriptRefined:
      "Okay, so the pitch for Voicebox is basically this: it's a local-first voice studio. Everything runs on your machine. You clone voices from a few seconds of audio, generate speech across seven TTS engines, and now with the Captures tab, you can dictate into any app. No cloud, no API keys, no per-character fees. Your voice data never leaves your device. Privacy isn't a feature here — it's the architecture.",
    durationMs: 38000,
    ago: '4 min ago',
    createdAtLabel: 'Apr 22, 3:47 PM',
    source: 'dictation',
    sttModel: 'turbo',
    language: 'en',
  },
  {
    id: 'c2',
    seed: 23,
    transcriptRaw:
      "draft an update for the blog about the agent voice feature the key point is one MCP tool call and any agent on your machine gets a voice claude code finishes a long task calls voicebox dot speak and you hear it in a voice you've cloned morgan scarlett whatever you set up same pill that shows when you're dictating also shows when an agent is speaking so you always know what's coming out of your machine closes the whole voice IO loop for agents",
    transcriptRefined:
      "Draft an update for the blog about the agent voice feature. The key point: one MCP tool call, and any agent on your machine gets a voice. Claude Code finishes a long task, calls voicebox.speak, and you hear it in a voice you've cloned — Morgan, Scarlett, whatever you've set up. The same pill that shows when you're dictating also shows when an agent is speaking, so you always know what's coming out of your machine. It closes the full voice I/O loop for agents.",
    durationMs: 41000,
    ago: '22 min ago',
    createdAtLabel: 'Apr 22, 3:29 PM',
    source: 'dictation',
    sttModel: 'turbo',
    language: 'en',
  },
  {
    id: 'c3',
    seed: 37,
    transcriptRaw:
      "tech overview for the readme seven TTS engines qwen3 kokoro chatterbox luxtts customvoice tada and chatterbox turbo whisper for STT in five sizes from base up to large and a turbo variant one local LLM qwen 3.5 shared runtime across all of them one model directory one GPU story no fragmented caches pick the right model per job speed on CPU laptops quality on an M series mac all switchable per generation",
    transcriptRefined:
      "Tech overview for the README: seven TTS engines — Qwen3, Kokoro, Chatterbox, LuxTTS, CustomVoice, TADA, and Chatterbox Turbo. Whisper for STT, in five sizes from Base up to Large, plus a Turbo variant. One local LLM, Qwen 3.5, with a shared runtime across all of them. One model directory, one GPU story, no fragmented caches. Pick the right model per job — speed on CPU laptops, quality on an M-series Mac, switchable per-generation.",
    durationMs: 34000,
    ago: '1 hr ago',
    createdAtLabel: 'Apr 22, 2:51 PM',
    source: 'dictation',
    sttModel: 'turbo',
    language: 'en',
  },
  {
    id: 'c4',
    seed: 53,
    transcriptRaw:
      "okay the real magic is this you speak to voicebox your transcript gets cleaned up by a local LLM it pastes into whatever you're focused on then the agent you're talking to responds and it replies with voice in a voice you cloned through the same pill that's the loop elevenlabs has TTS wisprflow has dictation but neither runs locally and neither does both halves voicebox is full voice IO for humans and AI agents entirely on your machine",
    transcriptRefined:
      "Okay, the real magic: you speak to Voicebox, your transcript gets cleaned up by a local LLM, and it pastes into whatever you're focused on. Then the agent you're talking to responds — and it replies with voice, in a voice you've cloned, through the same pill. That's the loop. ElevenLabs has TTS, WisprFlow has dictation, but neither runs locally and neither does both halves. Voicebox is full voice I/O for humans and AI agents, entirely on your machine.",
    durationMs: 42000,
    ago: 'Yesterday',
    createdAtLabel: 'Apr 21, 11:14 PM',
    source: 'dictation',
    sttModel: 'large',
    language: 'en',
  },
];

const PROFILES = [
  { id: 'p1', name: 'Morgan', description: 'Warm, measured', gradient: 'from-blue-400 to-indigo-500' },
  { id: 'p2', name: 'Scarlett', description: 'Bright, conversational', gradient: 'from-emerald-400 to-teal-500' },
  { id: 'p3', name: 'Jarvis', description: 'Dry, composed', gradient: 'from-purple-500 to-fuchsia-500' },
];

function formatDuration(ms: number): string {
  const total = Math.round(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function SourceBadge({ source }: { source: Capture['source'] }) {
  const Icon = source === 'dictation' ? Mic : source === 'recording' ? CircleDot : FileAudio;
  const label =
    source === 'dictation' ? 'Dictation' : source === 'recording' ? 'Recording' : 'File';
  return (
    <span className="inline-flex items-center h-5 px-1.5 gap-1 rounded-md text-[10px] font-medium bg-muted/60 text-muted-foreground border border-transparent">
      <Icon className="h-2.5 w-2.5" />
      {label}
    </span>
  );
}

function RefinedBadge() {
  return (
    <span className="inline-flex items-center h-5 px-1.5 gap-1 rounded-md text-[10px] font-medium bg-accent/10 text-accent border border-accent/20">
      <Sparkles className="h-2.5 w-2.5" />
      Refined
    </span>
  );
}

function BetaBadge() {
  return (
    <span className="inline-flex items-center h-5 px-1.5 rounded-md text-[10px] font-medium text-accent bg-accent/10 border border-accent/20">
      Beta
    </span>
  );
}

// ─── Capture list row ───────────────────────────────────────────────────────

function CaptureRow({
  capture,
  selected,
  onSelect,
}: {
  capture: Capture;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full text-left p-3 rounded-lg transition-colors block ${
        selected
          ? 'bg-muted/70 border border-border'
          : 'border border-transparent hover:bg-muted/30'
      }`}
    >
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-[11px] text-muted-foreground font-medium">{capture.ago}</span>
        <div className="flex-1" />
        <span className="text-[10px] text-muted-foreground/70 tabular-nums">
          {formatDuration(capture.durationMs)}
        </span>
      </div>
      <div className="text-[13px] text-foreground/90 line-clamp-2 leading-snug mb-2">
        {capture.transcriptRefined}
      </div>
      <div className="flex items-center gap-1.5 flex-wrap">
        <SourceBadge source={capture.source} />
        <RefinedBadge />
      </div>
    </button>
  );
}

// ─── Detail view ────────────────────────────────────────────────────────────

function DetailView({ capture }: { capture: Capture }) {
  const [showRefined, setShowRefined] = useState(true);
  const [profileIdx, setProfileIdx] = useState(0);

  useEffect(() => {
    setShowRefined(true);
  }, [capture.id]);

  useEffect(() => {
    const iv = window.setInterval(() => {
      setProfileIdx((i) => (i + 1) % PROFILES.length);
    }, 2600);
    return () => window.clearInterval(iv);
  }, []);

  const playAs = PROFILES[profileIdx];
  const transcript = showRefined ? capture.transcriptRefined : capture.transcriptRaw;

  return (
    <div className="h-full flex flex-col px-8 pt-4 pb-5 overflow-hidden">
      {/* Compact top row — date + language + source, inline */}
      <div className="flex items-center gap-2 text-[11px] text-muted-foreground/80 mb-4 shrink-0">
        <span>{capture.createdAtLabel}</span>
        {capture.language && (
          <>
            <span className="text-muted-foreground/30">·</span>
            <span>{capture.language.toUpperCase()}</span>
          </>
        )}
        <span className="text-muted-foreground/30">·</span>
        <SourceBadge source={capture.source} />
      </div>

      {/* Audio player card */}
      <div className="rounded-xl border border-border bg-muted/20 p-4 mb-5 shrink-0">
        <div className="flex items-center gap-4">
          <div className="h-10 w-10 rounded-full border border-border bg-background flex items-center justify-center shrink-0">
            <Play className="h-4 w-4 ml-0.5 fill-current text-foreground" />
          </div>
          <FakeWaveform seed={capture.seed} active className="flex-1" />
          <span className="text-xs tabular-nums text-muted-foreground font-medium">
            {formatDuration(capture.durationMs)}
          </span>
        </div>
      </div>

      {/* Transcript header */}
      <div className="flex items-center gap-3 mb-3 shrink-0">
        <div className="inline-flex rounded-md bg-muted/40 p-0.5 border border-border">
          <button
            type="button"
            onClick={() => setShowRefined(true)}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              showRefined
                ? 'bg-background shadow-sm text-foreground'
                : 'text-muted-foreground'
            }`}
          >
            <Sparkles className="h-3 w-3 inline-block mr-1 -translate-y-px" />
            Refined
          </button>
          <button
            type="button"
            onClick={() => setShowRefined(false)}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              !showRefined
                ? 'bg-background shadow-sm text-foreground'
                : 'text-muted-foreground'
            }`}
          >
            <Subtitles className="h-3 w-3 inline-block mr-1 -translate-y-px" />
            Raw
          </button>
        </div>
        <div className="flex-1" />
        <span className="text-xs text-muted-foreground whitespace-nowrap">
          {showRefined
            ? 'Refined with Qwen3 · 1.7B'
            : `Whisper ${capture.sttModel}`}
        </span>
      </div>

      {/* Transcript body — focal point, fills remaining height */}
      <div className="flex-1 min-h-0 rounded-xl border border-border bg-muted/10 p-6 overflow-y-auto mb-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={`${capture.id}-${showRefined}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="text-[15px] leading-relaxed text-foreground/90"
          >
            {transcript}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Action row — matches CapturesTab bottom row */}
      <div className="flex items-center gap-2 shrink-0 flex-wrap">
        <div className="inline-flex">
          <div className="inline-flex items-center justify-center gap-2 whitespace-nowrap h-9 rounded-full rounded-tr-none rounded-br-none border border-r-0 border-input bg-background pl-2 pr-3 text-sm font-medium transition-colors">
            <div
              className={`h-5 w-5 rounded-full bg-gradient-to-br shrink-0 ring-1 ring-white/10 ${playAs.gradient}`}
            />
            <Volume2 className="h-4 w-4 shrink-0" />
            Play as {playAs.name}
          </div>
          <div className="inline-flex items-center justify-center gap-2 whitespace-nowrap h-9 px-2 rounded-full rounded-tl-none rounded-bl-none border border-input bg-background text-sm font-medium transition-colors">
            <ChevronDown className="h-4 w-4 shrink-0 opacity-70" />
          </div>
        </div>
        <div className="inline-flex items-center gap-2 h-9 px-3 rounded-full border border-input bg-background text-sm font-medium text-foreground whitespace-nowrap">
          <Copy className="h-3.5 w-3.5" />
          Copy
        </div>
        <div className="inline-flex items-center gap-2 h-9 px-3 rounded-full border border-input bg-background text-sm font-medium text-foreground whitespace-nowrap">
          <Sparkles className="h-3.5 w-3.5" />
          Re-refine
        </div>
        <div className="inline-flex items-center gap-2 h-9 px-3 rounded-full border border-input bg-background text-sm font-medium text-foreground whitespace-nowrap">
          <Download className="h-3.5 w-3.5" />
          Export
        </div>
      </div>
    </div>
  );
}

// ─── Main mockup ────────────────────────────────────────────────────────────

export function CapturesMockup() {
  const [selectedId, setSelectedId] = useState<string>(CAPTURES[0].id);

  useEffect(() => {
    const iv = window.setInterval(() => {
      setSelectedId((current) => {
        const idx = CAPTURES.findIndex((c) => c.id === current);
        return CAPTURES[(idx + 1) % CAPTURES.length].id;
      });
    }, 4200);
    return () => window.clearInterval(iv);
  }, []);

  const selected = CAPTURES.find((c) => c.id === selectedId) ?? CAPTURES[0];

  return (
    <div className="relative z-20 mx-auto w-full max-w-5xl px-6">
      <div className="overflow-hidden rounded-2xl border border-app-line bg-app-box shadow-[0_25px_60px_rgba(0,0,0,0.5),0_8px_20px_rgba(0,0,0,0.3)] md:h-[640px] pointer-events-none select-none">
        <div className="flex flex-col md:flex-row h-full">
          <Sidebar />

          {/* ── Main area: two-panel Captures tab ─────────────────── */}
          <div className="flex-1 flex flex-col md:flex-row min-w-0 relative">
            {/* ── Left: capture list (w-[340px]) ──────────────────── */}
            <div
              style={{ width: 300, flex: '0 0 300px' }}
              className="flex flex-col overflow-hidden border-r border-app-line"
            >
              {/* Header — normal flow */}
              <div className="shrink-0 pl-4 pr-4 pt-4 pb-2">
                <div className="flex items-center gap-2 mb-5">
                  <h1 className="text-2xl px-4 font-bold">Captures</h1>
                  <BetaBadge />
                </div>
                <div className="h-9 flex items-center rounded-full border border-input bg-background px-4 text-sm text-muted-foreground">
                  Search transcripts…
                </div>
              </div>

              {/* Scroll area */}
              <div className="flex-1 overflow-hidden">
                <div className="px-4 pt-2 pb-6 space-y-1">
                  {CAPTURES.map((capture) => (
                    <CaptureRow
                      key={capture.id}
                      capture={capture}
                      selected={selectedId === capture.id}
                      onSelect={() => setSelectedId(capture.id)}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* ── Right: capture detail (flex-1) ───────────────────── */}
            <div className="flex-1 flex flex-col overflow-hidden min-w-0">
              <AnimatePresence mode="wait">
                <motion.div
                  key={selectedId}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="flex-1 overflow-hidden"
                >
                  <DetailView capture={selected} />
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
