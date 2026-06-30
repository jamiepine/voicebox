'use client';

import { motion } from 'framer-motion';
import { Bot, Mic2, Sparkles } from 'lucide-react';
import { useEffect, useState } from 'react';

// ─── Hero: Hotkey Pill ──────────────────────────────────────────────────────
// Ported from app/src/components/ServerTab/CapturesPage.tsx HotkeyPillPreview.
// Scaled up and retuned for the landing page — larger grid field, stretched
// aspect, longer rest phase so the loop reads as intentional.

type PillState = 'recording' | 'transcribing' | 'refining' | 'rest';

const PILL_SEQUENCE: PillState[] = ['recording', 'transcribing', 'refining', 'rest'];
const PILL_DURATIONS: Record<PillState, number> = {
  recording: 2800,
  transcribing: 1600,
  refining: 1600,
  rest: 1400,
};
const PILL_LABELS: Record<Exclude<PillState, 'rest'>, string> = {
  recording: 'Recording',
  transcribing: 'Transcribing',
  refining: 'Refining',
};

function PillAudioBars({ mode }: { mode: 'live' | 'thinking' }) {
  return (
    <div className="flex items-center gap-[3px] h-6 shrink-0">
      {[0, 1, 2, 3, 4, 5, 6].map((i) => (
        <motion.div
          key={`${mode}-${i}`}
          className="w-[3.5px] rounded-full bg-accent"
          animate={
            mode === 'live'
              ? { height: ['10px', '18px', '6px', '16px', '10px'] }
              : { height: ['8px', '20px', '8px'] }
          }
          transition={
            mode === 'live'
              ? { duration: 1.1, repeat: Infinity, delay: i * 0.12, ease: 'easeInOut' }
              : { duration: 0.7, repeat: Infinity, delay: i * 0.09, ease: 'easeInOut' }
          }
        />
      ))}
    </div>
  );
}

function KbdKey({ children }: { children: string }) {
  return (
    <kbd className="inline-flex items-center justify-center h-7 min-w-[1.75rem] px-2 rounded-md border border-app-line bg-app-darkBox/80 font-mono text-[12px] font-medium text-foreground shadow-[inset_0_-2px_0_rgba(0,0,0,0.2)]">
      {children}
    </kbd>
  );
}

export function DictationHero() {
  const [state, setState] = useState<PillState>('recording');
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const t = window.setTimeout(() => {
      const next = PILL_SEQUENCE[(PILL_SEQUENCE.indexOf(state) + 1) % PILL_SEQUENCE.length];
      setState(next);
    }, PILL_DURATIONS[state]);
    return () => window.clearTimeout(t);
  }, [state]);

  useEffect(() => {
    if (state !== 'recording') return;
    setTick(0);
    const iv = window.setInterval(() => setTick((n) => n + 1), 90);
    return () => window.clearInterval(iv);
  }, [state]);

  const elapsedSec = Math.floor((tick * 90) / 1000);
  const elapsedLabel = `0:${String(elapsedSec).padStart(2, '0')}`;
  const pillVisible = state !== 'rest';
  const barMode: 'live' | 'thinking' = state === 'recording' ? 'live' : 'thinking';
  const labelText = state === 'rest' ? PILL_LABELS.recording : PILL_LABELS[state];

  return (
    <div className="mx-auto w-full max-w-4xl">
      {/* Shortcut hint above the field */}
      <div className="mt-10 mb-4 flex flex-wrap items-center justify-center gap-x-3 gap-y-1.5 text-[13px] text-muted-foreground">
        <span>Hold</span>
        <div className="flex items-center gap-2">
          <KbdKey>⌘</KbdKey>
          <KbdKey>⌥</KbdKey>
        </div>
        <span>on macOS,</span>
        <div className="flex items-center gap-2">
          <KbdKey>Ctrl</KbdKey>
          <KbdKey>Alt</KbdKey>
        </div>
        <span>on Windows — from anywhere on your machine.</span>
      </div>

      {/* The stage — gridded field with the pill floating in the middle */}
      <div
        className="relative rounded-2xl border border-app-line bg-app-darkerBox/60 overflow-hidden aspect-[5/1]"
        style={{
          backgroundImage: `
            linear-gradient(to right, hsl(30 10% 94% / 0.04) 1px, transparent 1px),
            linear-gradient(to bottom, hsl(30 10% 94% / 0.04) 1px, transparent 1px)
          `,
          backgroundSize: '32px 32px',
        }}
      >
        {/* Soft accent glow behind the pill */}
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="w-[420px] h-[160px] rounded-full bg-accent/10 blur-[80px]" />
        </div>

        {/* Floating pill */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div
            className={`inline-flex items-center gap-4 px-6 h-14 rounded-full bg-black/55 backdrop-blur-md text-accent shadow-[0_12px_40px_rgba(0,0,0,0.45)] transition-opacity duration-500 ease-out ${
              pillVisible ? 'opacity-100' : 'opacity-0'
            }`}
          >
            {/* Gold dot — pings during recording */}
            <span className="relative flex h-2.5 w-2.5 shrink-0">
              {state === 'recording' && (
                <span className="absolute inset-0 rounded-full bg-accent animate-ping opacity-70" />
              )}
              <span className="relative rounded-full h-2.5 w-2.5 bg-accent" />
            </span>

            <span
              className="text-[15px] font-medium shrink-0"
              style={{ minWidth: '120px' }}
            >
              {labelText}
            </span>

            <PillAudioBars mode={barMode} />

            <span className="text-[13px] tabular-nums text-accent/70 font-medium shrink-0 -ml-1">
              {elapsedLabel}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Card: Whisper, sized for every machine ────────────────────────────────

type EngineRow = { name: string; size: string; langs: string };

const STT_ENGINES: EngineRow[] = [
  { name: 'Whisper Base', size: '74M', langs: '99 langs' },
  { name: 'Whisper Small', size: '244M', langs: '99 langs' },
  { name: 'Whisper Medium', size: '769M', langs: '99 langs' },
  { name: 'Whisper Large', size: '1.5B', langs: '99 langs' },
  { name: 'Whisper Turbo', size: '809M', langs: '99 langs' },
];

function MultiEngineSTTAnimation() {
  const [activeIdx, setActiveIdx] = useState(0);

  useEffect(() => {
    const iv = window.setInterval(() => {
      setActiveIdx((i) => (i + 1) % STT_ENGINES.length);
    }, 1600);
    return () => window.clearInterval(iv);
  }, []);

  return (
    <div className="h-40 w-full flex items-center justify-center overflow-hidden rounded-md bg-app-darkerBox/50 p-4">
      <div className="w-full max-w-[240px] space-y-1.5">
        {STT_ENGINES.map((engine, i) => {
          const active = i === activeIdx;
          return (
            <motion.div
              key={engine.name}
              className="flex items-center gap-2 px-2.5 py-1.5 rounded-md border"
              animate={{
                borderColor: active ? 'hsl(43 50% 45% / 0.5)' : 'rgba(255,255,255,0.06)',
                backgroundColor: active ? 'hsl(43 50% 45% / 0.08)' : 'rgba(255,255,255,0.02)',
              }}
              transition={{ duration: 0.3 }}
            >
              <motion.div
                className="w-1.5 h-1.5 rounded-full shrink-0"
                animate={{
                  backgroundColor: active ? 'hsl(43 50% 50%)' : 'rgba(255,255,255,0.15)',
                  boxShadow: active ? '0 0 8px hsl(43 50% 50%)' : '0 0 0 transparent',
                }}
                transition={{ duration: 0.3 }}
              />
              <span
                className="text-[10px] font-medium flex-1 truncate"
                style={{ color: active ? 'hsl(43 50% 55%)' : 'rgba(255,255,255,0.55)' }}
              >
                {engine.name}
              </span>
              <span className="text-[9px] font-mono text-ink-faint/70 tabular-nums">
                {engine.size}
              </span>
              <span className="text-[9px] text-ink-faint/60">{engine.langs}</span>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Card: LLM Refinement ───────────────────────────────────────────────────

const REFINEMENT_PAIRS = [
  {
    raw: 'um so like i think we should ship it on friday, actually no wait, tuesday',
    clean: 'I think we should ship it on Tuesday.',
  },
  {
    raw: 'could you uh run the migration real quick, and then, yeah, check the logs',
    clean: 'Could you run the migration, then check the logs?',
  },
];

function RefinementAnimation() {
  const [pairIdx, setPairIdx] = useState(0);
  const [showClean, setShowClean] = useState(false);

  useEffect(() => {
    let mounted = true;
    const step = () => {
      if (!mounted) return;
      setShowClean(false);
      window.setTimeout(() => mounted && setShowClean(true), 1400);
      window.setTimeout(() => {
        if (!mounted) return;
        setPairIdx((i) => (i + 1) % REFINEMENT_PAIRS.length);
      }, 4000);
    };
    step();
    const iv = window.setInterval(step, 4000);
    return () => {
      mounted = false;
      window.clearInterval(iv);
    };
  }, []);

  const pair = REFINEMENT_PAIRS[pairIdx];

  return (
    <div className="h-40 w-full flex flex-col items-center justify-center overflow-hidden rounded-md bg-app-darkerBox/50 p-4 gap-2.5">
      <div className="w-full max-w-[260px] space-y-2">
        {/* Raw line — always visible, dims when refined */}
        <motion.div
          key={`raw-${pairIdx}`}
          className="text-[10px] font-mono leading-relaxed"
          initial={{ opacity: 0, y: 2 }}
          animate={{
            opacity: showClean ? 0.35 : 1,
            y: 0,
            color: showClean ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.6)',
          }}
          transition={{ duration: 0.4 }}
        >
          <span className="text-ink-faint/50 mr-1.5">raw</span>
          {pair.raw}
        </motion.div>

        {/* Refined line — fades in */}
        <motion.div
          key={`clean-${pairIdx}`}
          className="text-[10px] leading-relaxed"
          initial={{ opacity: 0, y: 4 }}
          animate={{
            opacity: showClean ? 1 : 0,
            y: showClean ? 0 : 4,
          }}
          transition={{ duration: 0.5 }}
        >
          <span className="text-accent/70 mr-1.5 font-mono">clean</span>
          <span className="text-foreground">{pair.clean}</span>
        </motion.div>
      </div>

      {/* Activity indicator */}
      <div className="flex items-center gap-1.5 text-[9px] font-mono text-ink-faint mt-1">
        <Sparkles className="h-2.5 w-2.5 text-accent" />
        <span>{showClean ? 'refined' : 'Qwen3 · refining...'}</span>
      </div>
    </div>
  );
}

// ─── Card: Agent voice output ───────────────────────────────────────────────

type AgentSpeaker = {
  agent: string;
  voice: string;
  gradient: [string, string];
  message: string;
};

const AGENT_SPEAKERS: AgentSpeaker[] = [
  {
    agent: 'Claude Code',
    voice: 'Morgan',
    gradient: ['#60a5fa', '#6366f1'],
    message: 'Tests passing. Ready to merge.',
  },
  {
    agent: 'Cursor',
    voice: 'Scarlett',
    gradient: ['#34d399', '#14b8a6'],
    message: 'Build finished in 42s.',
  },
  {
    agent: 'Cline',
    voice: 'Jarvis',
    gradient: ['#a855f7', '#ec4899'],
    message: 'Deploy complete.',
  },
];

function AgentVoiceAnimation() {
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    const iv = window.setInterval(() => {
      setIdx((i) => (i + 1) % AGENT_SPEAKERS.length);
    }, 2600);
    return () => window.clearInterval(iv);
  }, []);

  const current = AGENT_SPEAKERS[idx];

  return (
    <div className="h-40 w-full flex flex-col items-center justify-center overflow-hidden rounded-md bg-app-darkerBox/50 p-4 gap-2.5">
      {/* Which agent called speak() */}
      <motion.div
        key={`agent-${idx}`}
        className="text-[9px] font-mono"
        initial={{ opacity: 0, y: 2 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <span className="text-ink-faint/50">via MCP</span>
        <span className="mx-1.5 text-ink-faint/30">·</span>
        <span className="text-ink-dull">{current.agent}</span>
      </motion.div>

      {/* Pill in speaking state */}
      <motion.div
        key={`pill-${idx}`}
        className="inline-flex items-center gap-2.5 px-3 h-8 rounded-full bg-black/55 backdrop-blur-sm shadow-[0_6px_20px_rgba(0,0,0,0.35)]"
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div
          className="h-4 w-4 rounded-full shrink-0 ring-1 ring-white/10"
          style={{
            background: `linear-gradient(135deg, ${current.gradient[0]}, ${current.gradient[1]})`,
          }}
        />
        <span className="text-[10px] font-medium text-foreground/90">
          Speaking · <span className="text-accent">{current.voice}</span>
        </span>
        <div className="flex items-center gap-[2px] h-3.5">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <motion.div
              key={`${current.voice}-${i}`}
              className="w-[2px] rounded-full bg-accent"
              animate={{ height: ['4px', '11px', '5px', '9px', '4px'] }}
              transition={{
                duration: 0.9,
                repeat: Infinity,
                delay: i * 0.08,
                ease: 'easeInOut',
              }}
            />
          ))}
        </div>
      </motion.div>

      {/* The line the agent is saying */}
      <motion.div
        key={`msg-${idx}`}
        className="text-[10px] font-mono text-ink-dull max-w-[220px] text-center leading-relaxed"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4, delay: 0.15 }}
      >
        &ldquo;{current.message}&rdquo;
      </motion.div>
    </div>
  );
}

// ─── Feature data + card ────────────────────────────────────────────────────

const CAPTURE_FEATURES = [
  {
    title: 'Whisper, sized for every machine',
    description:
      'Base, Small, Medium, Large, and Turbo. Pick the size that fits your hardware and quality bar — 99 languages across every tier, all running locally.',
    icon: Mic2,
    animation: MultiEngineSTTAnimation,
  },
  {
    title: 'Refined transcripts',
    description:
      'A local LLM cleans ums, self-corrections, and punctuation without rephrasing. Optional, toggleable, and never leaves your machine.',
    icon: Sparkles,
    animation: RefinementAnimation,
  },
  {
    title: 'Agents speak in voices you own',
    description:
      'Any MCP-aware agent — Claude Code, Cursor, Cline — gets a voice with one tool call. The pill surfaces when an agent is speaking, so you always see what’s coming out of your machine.',
    icon: Bot,
    animation: AgentVoiceAnimation,
  },
];

function CaptureCard({ feature }: { feature: (typeof CAPTURE_FEATURES)[number] }) {
  const Icon = feature.icon;
  const Animation = feature.animation;
  return (
    <div className="rounded-lg border border-app-line bg-app-darkBox overflow-hidden">
      <div className="pointer-events-none select-none">
        <Animation />
      </div>
      <div className="p-5">
        <div className="flex items-center gap-2 mb-2">
          <Icon className="h-4 w-4 text-accent" />
          <h3 className="text-[15px] font-medium text-foreground">{feature.title}</h3>
        </div>
        <p className="text-sm leading-relaxed text-muted-foreground">{feature.description}</p>
      </div>
    </div>
  );
}

// ─── Section ────────────────────────────────────────────────────────────────

export function CaptureSection() {
  return (
    <section id="capture" className="border-t border-border py-24">
      <div className="mx-auto max-w-7xl px-6">
        {/* Kicker + headline */}
        <div className="text-center mb-14">
          <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-4">
            Capture
          </div>
          <h2 className="text-4xl font-semibold tracking-tight text-foreground md:text-5xl mb-5">
            Dictate anywhere. Paste into any app.
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto text-base md:text-lg leading-relaxed">
            Hold a shortcut anywhere on your machine, speak, release.
            The transcript lands in a focused text field in any app, or your clipboard. Agents speak
            back through the same pill in any cloned voice.
          </p>
        </div>

        {/* Hero pill animation */}
        <div className="mb-16">
          <DictationHero />
        </div>

        {/* Feature cards */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {CAPTURE_FEATURES.map((f) => (
            <CaptureCard key={f.title} feature={f} />
          ))}
        </div>
      </div>
    </section>
  );
}
