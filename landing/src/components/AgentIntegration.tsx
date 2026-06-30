'use client';

import { motion } from 'framer-motion';
import { Eye, Sliders, Waypoints } from 'lucide-react';
import { useEffect, useState } from 'react';

// ─── Scenarios (the agent console cycles through these) ────────────────────

type Scenario = {
  agent: string;
  voice: string;
  voiceGradient: [string, string];
  log: { prefix: string; text: string; tone: 'accent' | 'success' | 'dim' }[];
  utterance: string;
};

const SCENARIOS: Scenario[] = [
  {
    agent: 'Claude Code',
    voice: 'Morgan',
    voiceGradient: ['#60a5fa', '#6366f1'],
    log: [
      { prefix: '$', text: 'claude run', tone: 'accent' },
      { prefix: '✓', text: 'Tests passing (42 files)', tone: 'success' },
      { prefix: '✓', text: 'Build succeeded in 12.4s', tone: 'success' },
      { prefix: '→', text: 'voicebox.speak({ profile: "Morgan" })', tone: 'dim' },
    ],
    utterance: 'Tests passing. Ready to merge.',
  },
  {
    agent: 'Cursor',
    voice: 'Scarlett',
    voiceGradient: ['#34d399', '#14b8a6'],
    log: [
      { prefix: '$', text: 'cursor agent:deploy', tone: 'accent' },
      { prefix: '✓', text: 'Migration applied (4 tables)', tone: 'success' },
      { prefix: '✓', text: 'Deploy complete', tone: 'success' },
      { prefix: '→', text: 'voicebox.speak({ profile: "Scarlett" })', tone: 'dim' },
    ],
    utterance: 'Deploy shipped. Prod is green.',
  },
  {
    agent: 'Cline',
    voice: 'Jarvis',
    voiceGradient: ['#a855f7', '#ec4899'],
    log: [
      { prefix: '$', text: 'cline task:review', tone: 'accent' },
      { prefix: '!', text: '3 files need attention', tone: 'dim' },
      { prefix: '→', text: 'voicebox.speak({ profile: "Jarvis" })', tone: 'dim' },
    ],
    utterance: 'Review ready. Three files to look at.',
  },
];

const TONE_CLASSES: Record<Scenario['log'][number]['tone'], string> = {
  accent: 'text-accent',
  success: 'text-emerald-400/80',
  dim: 'text-ink-faint/70',
};

// ─── Console mockup ─────────────────────────────────────────────────────────

function AgentConsole({ scenario, cycleKey }: { scenario: Scenario; cycleKey: number }) {
  return (
    <div className="rounded-xl border border-app-line bg-app-darkerBox overflow-hidden shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
      {/* Titlebar */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-app-line bg-app-darkBox/60">
        <div className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500/50" />
          <span className="h-2.5 w-2.5 rounded-full bg-yellow-500/50" />
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/50" />
        </div>
        <div className="flex-1 text-center">
          <span className="text-[10px] font-mono text-ink-faint/60">{scenario.agent}</span>
        </div>
        <div className="w-12" />
      </div>

      {/* Body */}
      <div className="p-5 font-mono text-[12px] leading-relaxed min-h-[220px] flex flex-col">
        <div className="space-y-1.5">
          {scenario.log.map((line, i) => (
            <motion.div
              key={`${cycleKey}-line-${i}`}
              className="flex items-start gap-2"
              initial={{ opacity: 0, y: 2 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, delay: i * 0.15 }}
            >
              <span className={`shrink-0 ${TONE_CLASSES[line.tone]}`}>{line.prefix}</span>
              <span
                className={
                  line.tone === 'dim' ? 'text-ink-faint/70' : 'text-ink-dull'
                }
              >
                {line.text}
              </span>
            </motion.div>
          ))}
        </div>

        {/* Idle cursor so the terminal doesn't feel empty */}
        <div className="mt-auto flex items-center gap-2 pt-4">
          <span className="text-ink-faint/50">$</span>
          <span className="inline-block h-3.5 w-[7px] bg-ink-faint/40 animate-pulse" />
        </div>
      </div>
    </div>
  );
}

// ─── Desktop-floating pill stage ────────────────────────────────────────────

function AgentSpeakStage({ scenario, cycleKey }: { scenario: Scenario; cycleKey: number }) {
  return (
    <div
      className="relative rounded-xl border border-app-line bg-app-darkerBox/60 overflow-hidden min-h-[180px] flex-1"
      style={{
        backgroundImage: `
          linear-gradient(to right, hsl(30 10% 94% / 0.04) 1px, transparent 1px),
          linear-gradient(to bottom, hsl(30 10% 94% / 0.04) 1px, transparent 1px)
        `,
        backgroundSize: '28px 28px',
      }}
    >
      {/* Caption in the corner — "this is on the desktop, not in a terminal" */}
      <div className="absolute top-3 left-4 text-[9px] font-mono uppercase tracking-[0.22em] text-ink-faint/50">
        On your desktop
      </div>

      {/* Voice-tinted glow behind the pill */}
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <motion.div
          key={`glow-${cycleKey}`}
          className="w-[320px] h-[140px] rounded-full blur-[70px]"
          style={{
            background: `linear-gradient(135deg, ${scenario.voiceGradient[0]}, ${scenario.voiceGradient[1]})`,
          }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.3 }}
          transition={{ duration: 0.5 }}
        />
      </div>

      {/* Pill + utterance caption */}
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6">
        <motion.div
          key={`pill-${cycleKey}`}
          className="inline-flex items-center gap-3 px-4 h-11 rounded-full bg-black/55 backdrop-blur-md shadow-[0_12px_40px_rgba(0,0,0,0.45)]"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.6 }}
        >
          <div
            className="h-4 w-4 rounded-full shrink-0 ring-1 ring-white/10"
            style={{
              background: `linear-gradient(135deg, ${scenario.voiceGradient[0]}, ${scenario.voiceGradient[1]})`,
            }}
          />
          <span className="text-[12px] font-medium text-foreground/90 shrink-0">
            Speaking · <span className="text-accent">{scenario.voice}</span>
          </span>
          <div className="flex items-center gap-[2.5px] h-5 shrink-0">
            {[0, 1, 2, 3, 4, 5].map((i) => (
              <motion.div
                key={`bar-${scenario.voice}-${i}`}
                className="w-[2.5px] rounded-full bg-accent"
                animate={{ height: ['5px', '14px', '7px', '12px', '5px'] }}
                transition={{
                  duration: 1.0,
                  repeat: Infinity,
                  delay: i * 0.09,
                  ease: 'easeInOut',
                }}
              />
            ))}
          </div>
        </motion.div>

        <motion.div
          key={`utter-${cycleKey}`}
          className="text-[12px] text-ink-dull/80 italic text-center max-w-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4, delay: 0.9 }}
        >
          &ldquo;{scenario.utterance}&rdquo;
        </motion.div>
      </div>
    </div>
  );
}

// ─── Code panel ─────────────────────────────────────────────────────────────

const MCP_CONFIG = `{
  "mcpServers": {
    "voicebox": {
      "url": "http://127.0.0.1:17493/mcp"
    }
  }
}`;

const SPEAK_EXAMPLE = `// In any MCP-aware agent:
await voicebox.speak({
  text: "Deploy complete.",
  profile: "Morgan",
})`;

function CodePanel() {
  return (
    <div className="rounded-xl border border-app-line bg-app-darkBox overflow-hidden flex flex-col">
      {/* MCP config */}
      <div className="p-5 border-b border-app-line">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-[9px] font-mono text-accent font-semibold tabular-nums">
            01
          </span>
          <span className="text-[10px] font-mono text-ink-faint/70 uppercase tracking-wider">
            Add Voicebox to your MCP config
          </span>
        </div>
        <pre className="text-[11px] font-mono text-ink-dull leading-relaxed overflow-x-auto">
          {MCP_CONFIG}
        </pre>
      </div>

      {/* Tool call */}
      <div className="p-5 flex-1">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-[9px] font-mono text-accent font-semibold tabular-nums">
            02
          </span>
          <span className="text-[10px] font-mono text-ink-faint/70 uppercase tracking-wider">
            The tool is now available
          </span>
        </div>
        <pre className="text-[11px] font-mono text-ink-dull leading-relaxed overflow-x-auto">
          {SPEAK_EXAMPLE}
        </pre>

        {/* Hint line */}
        <div className="mt-4 text-[10px] text-ink-faint/60 leading-relaxed">
          Also exposed as{' '}
          <code className="text-accent/80">POST /speak</code> for anything that
          doesn&rsquo;t speak MCP — ACP, A2A, shell scripts, or custom harnesses.
        </div>
      </div>
    </div>
  );
}

// ─── Support bullets ────────────────────────────────────────────────────────

const BULLETS = [
  {
    icon: Sliders,
    title: 'Per-agent voice',
    description:
      'Bind each MCP client to a voice profile. Claude Code in Morgan, Cursor in Scarlett — you know which agent is talking without looking.',
  },
  {
    icon: Eye,
    title: 'Always visible',
    description:
      'Every agent-initiated speech surfaces the pill. No silent background TTS — you always see what’s coming out of your machine.',
  },
  {
    icon: Waypoints,
    title: 'Open protocols',
    description:
      'MCP ships day one. ACP, A2A, and anything else built on a tool-call primitive slots into the same endpoint.',
  },
];

// ─── Section ────────────────────────────────────────────────────────────────

export function AgentIntegration() {
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    const iv = window.setInterval(() => {
      setIdx((i) => (i + 1) % SCENARIOS.length);
    }, 4200);
    return () => window.clearInterval(iv);
  }, []);

  const scenario = SCENARIOS[idx];

  return (
    <section id="mcp" className="border-t border-border py-24">
      <div className="mx-auto max-w-6xl px-6">
        {/* Header */}
        <div className="max-w-3xl mx-auto text-center mb-14">
          <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-4">
            MCP
          </div>
          <h2 className="text-4xl md:text-5xl font-semibold tracking-tight text-foreground mb-5">
            Every agent gets a voice.
          </h2>
          <p className="text-muted-foreground text-base md:text-lg leading-relaxed">
            One tool call —{' '}
            <code className="text-accent font-mono text-[0.9em]">voicebox.speak</code> —
            and any MCP-aware agent can talk to you in a voice you&rsquo;ve cloned. Claude Code,
            Cursor, Cline, or anything that speaks MCP.
          </p>
        </div>

        {/* Code (left) + console with pill stage stacked underneath (right) */}
        <div className="grid md:grid-cols-2 gap-6 mb-12 items-stretch">
          <CodePanel />
          <div className="flex flex-col gap-4">
            <AgentConsole scenario={scenario} cycleKey={idx} />
            <AgentSpeakStage scenario={scenario} cycleKey={idx} />
          </div>
        </div>

        {/* Bullets */}
        <div className="grid md:grid-cols-3 gap-6">
          {BULLETS.map((bullet) => {
            const Icon = bullet.icon;
            return (
              <div
                key={bullet.title}
                className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-5"
              >
                <div className="flex items-center gap-2 mb-2">
                  <Icon className="h-4 w-4 text-accent" />
                  <h3 className="text-[14px] font-semibold text-foreground">
                    {bullet.title}
                  </h3>
                </div>
                <p className="text-[13px] leading-relaxed text-muted-foreground">
                  {bullet.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
