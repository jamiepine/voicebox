'use client';

import { AnimatePresence, motion } from 'framer-motion';
import { ArrowRight, Dices, Wand2 } from 'lucide-react';
import { useEffect, useState } from 'react';

// ─── Modes ──────────────────────────────────────────────────────────────────

type Mode = {
  id: 'compose' | 'rewrite';
  label: string;
  icon: typeof Dices;
  outputLabel: string;
  output: string;
} & (
  | { inputLabel: string; input: string }
  | { inputLabel?: undefined; input?: undefined }
);

const MODES: Mode[] = [
  {
    id: 'rewrite',
    label: 'Rewrite',
    icon: Wand2,
    inputLabel: 'Your text',
    outputLabel: "Marlowe, in character",
    input: 'the build is done and we shipped to production',
    output:
      "Build's wrapped, ship's left the dock. Another stack of code makes its way into prod, another row of green checks lining the wall.",
  },
  {
    id: 'compose',
    label: 'Compose',
    icon: Dices,
    outputLabel: "Marlowe, in character",
    output:
      "She came through clean. Not a single test casting a shadow. In this town, that's usually when you start worrying.",
  },
];

const PERSONA_DESCRIPTION =
  "1940s noir detective. World-weary, cynical, every situation a metaphor for the city's underbelly. Talks like he's seen one stack trace too many.";

// ─── Persona card ───────────────────────────────────────────────────────────

function PersonaCard() {
  return (
    <div className="rounded-xl border border-app-line bg-app-darkBox p-5 shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
      <div className="flex items-center gap-3 mb-4">
        <div
          className="h-12 w-12 rounded-full shrink-0 ring-1 ring-white/10"
          style={{ background: 'linear-gradient(135deg, #dc2626, #7f1d1d)' }}
        />
        <div className="min-w-0">
          <div className="text-[15px] font-semibold text-foreground leading-tight">Marlowe</div>
          <div className="text-[11px] text-muted-foreground/80 leading-tight mt-0.5">
            Voice profile · cloned from a 12s sample
          </div>
        </div>
      </div>

      <div className="mb-1.5 text-[10px] font-mono uppercase tracking-[0.2em] text-ink-faint/70">
        Personality
      </div>
      <p className="text-[13px] leading-relaxed text-ink-dull italic">&ldquo;{PERSONA_DESCRIPTION}&rdquo;</p>
    </div>
  );
}

// ─── Mode demo ──────────────────────────────────────────────────────────────

function ModeDemo({ mode, cycleKey }: { mode: Mode; cycleKey: number }) {
  return (
    <div className="rounded-xl border border-app-line bg-app-darkerBox overflow-hidden flex flex-col flex-1">
      {/* Mode tabs */}
      <div className="flex items-center gap-1 p-1.5 border-b border-app-line bg-app-darkBox/40">
        {MODES.map((m) => {
          const Icon = m.icon;
          const active = m.id === mode.id;
          return (
            <div
              key={m.id}
              className={`flex items-center gap-1.5 h-8 px-3 rounded-md text-[12px] font-medium transition-colors ${
                active
                  ? 'bg-white/[0.07] text-foreground border border-white/[0.08]'
                  : 'text-muted-foreground/60'
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              {m.label}
            </div>
          );
        })}
      </div>

      {/* Input → Output */}
      <div className="p-5 flex-1 flex flex-col">
        <AnimatePresence mode="wait">
          <motion.div
            key={`${cycleKey}-${mode.id}`}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.25 }}
            className="flex flex-col gap-4 flex-1"
          >
            {/* Input */}
            {mode.input ? (
              <div>
                <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-ink-faint/70 mb-1.5">
                  {mode.inputLabel}
                </div>
                <div className="text-[13px] leading-relaxed text-ink-dull/90 font-mono bg-black/20 rounded-md border border-app-line/60 px-3 py-2.5">
                  {mode.input}
                </div>
              </div>
            ) : (
              <div>
                <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-ink-faint/70 mb-1.5">
                  No input
                </div>
                <div className="flex items-center gap-2.5 text-[13px] text-ink-dull/90 bg-black/20 rounded-md border border-app-line/60 px-3 py-2.5">
                  <Dices className="h-4 w-4 text-accent shrink-0" />
                  <span>Click Compose — the character improvises a fresh line.</span>
                </div>
              </div>
            )}

            {/* Arrow */}
            <div className="flex items-center justify-center gap-2 text-[10px] font-mono uppercase tracking-[0.2em] text-ink-faint/50">
              <span>In character</span>
              <ArrowRight className="h-3 w-3" />
            </div>

            {/* Output */}
            <div>
              <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-accent mb-1.5">
                {mode.outputLabel}
              </div>
              <div className="text-[14px] leading-relaxed text-foreground bg-accent/[0.06] rounded-md border border-accent/20 px-3 py-2.5 italic">
                &ldquo;{mode.output}&rdquo;
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

// ─── Bullets ────────────────────────────────────────────────────────────────

const BULLETS = [
  {
    icon: Wand2,
    title: 'Rewrite',
    description:
      'Restate your text in their voice while preserving every idea. Same content, their delivery — for scripts, dubs, and consistent character voice across long-form work.',
  },
  {
    icon: Dices,
    title: 'Compose',
    description:
      'No input needed — hit the button and the character improvises a fresh line of their own. Roll again for another take. Useful for game dialogue, narration cues, or character barks.',
  },
];

// ─── Section ────────────────────────────────────────────────────────────────

export function Personalities() {
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    const iv = window.setInterval(() => {
      setIdx((i) => (i + 1) % MODES.length);
    }, 4500);
    return () => window.clearInterval(iv);
  }, []);

  const mode = MODES[idx];

  return (
    <section id="personalities" className="border-t border-border py-24">
      <div className="mx-auto max-w-6xl px-6">
        {/* Header */}
        <div className="max-w-3xl mx-auto text-center mb-14">
          <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-4">
            Personalities
          </div>
          <h2 className="text-4xl md:text-5xl font-semibold tracking-tight text-foreground mb-5">
            Voices with a personality.
          </h2>
          <p className="text-muted-foreground text-base md:text-lg leading-relaxed">
            Give any voice profile a free-form personality. Then{' '}
            <b className="text-foreground/90">Rewrite</b> your text in their voice, or let them{' '}
            <b className="text-foreground/90">Compose</b> a fresh line of their own — your cloned voice, in full character.
          </p>
        </div>

        {/* Mockup: persona card (left) + mode demo (right) */}
        <div className="grid md:grid-cols-[340px_1fr] gap-6 mb-12 items-stretch">
          <PersonaCard />
          <ModeDemo mode={mode} cycleKey={idx} />
        </div>

        {/* Bullets */}
        <div className="grid md:grid-cols-2 gap-6">
          {BULLETS.map((bullet) => {
            const Icon = bullet.icon;
            return (
              <div
                key={bullet.title}
                className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-5"
              >
                <div className="flex items-center gap-2 mb-2">
                  <Icon className="h-4 w-4 text-accent" />
                  <h3 className="text-[14px] font-semibold text-foreground">{bullet.title}</h3>
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
