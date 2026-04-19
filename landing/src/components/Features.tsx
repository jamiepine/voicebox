'use client';

import { motion } from 'framer-motion';
import { AudioLines, Cloud, MessageSquareText, Mic, Sparkles, TextCursorInput } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';

// ─── Lazy load wrapper ──────────────────────────────────────────────────────

function LazyLoad({
  children,
  className,
  rootMargin = '200px',
}: {
  children: React.ReactNode;
  className?: string;
  rootMargin?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [rootMargin]);

  return (
    <div ref={ref} className={className}>
      {visible ? children : null}
    </div>
  );
}

// ─── Animation: Voice Cloning ───────────────────────────────────────────────

function VoiceCloningAnimation() {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setPhase((p) => (p + 1) % 3);
    }, 2400);
    return () => clearInterval(interval);
  }, []);

  const samples = ['Sample 1', 'Sample 2', 'Sample 3'];
  const bars = [0.4, 0.7, 0.5, 0.9, 0.3, 0.6, 0.8, 0.4, 0.7, 0.5, 0.3, 0.6];

  return (
    <div className="h-40 w-full flex items-center justify-center overflow-hidden rounded-md bg-app-darkerBox/50 p-4">
      <div className="flex flex-col items-center gap-3 w-full max-w-[200px]">
        {/* Sample pills */}
        <div className="flex gap-1.5">
          {samples.map((s, i) => (
            <motion.div
              key={s}
              className="text-[9px] px-2 py-1 rounded-full border font-medium"
              animate={{
                borderColor: i === phase ? 'hsl(43 50% 45% / 0.5)' : 'rgba(255,255,255,0.06)',
                backgroundColor: i === phase ? 'hsl(43 50% 45% / 0.08)' : 'rgba(255,255,255,0.02)',
                color: i === phase ? 'hsl(43 50% 45%)' : 'rgba(255,255,255,0.4)',
              }}
              transition={{ duration: 0.3 }}
            >
              {s}
            </motion.div>
          ))}
        </div>

        {/* Waveform visualization */}
        <div className="flex items-center gap-[2px] h-10 w-full justify-center">
          {bars.map((h, i) => (
            <motion.div
              key={i}
              className="w-[4px] rounded-full"
              animate={{
                height: `${h * 100}%`,
                backgroundColor: phase === 2 ? 'hsl(43 50% 45%)' : 'rgba(255,255,255,0.15)',
              }}
              transition={{
                height: { duration: 0.6, delay: i * 0.04, ease: 'easeInOut' },
                backgroundColor: { duration: 0.3 },
              }}
            />
          ))}
        </div>

        {/* Result label */}
        <motion.div
          className="text-[9px] font-mono"
          animate={{
            opacity: phase === 2 ? 1 : 0.3,
            color: phase === 2 ? 'hsl(43 50% 45%)' : 'rgba(255,255,255,0.3)',
          }}
          transition={{ duration: 0.3 }}
        >
          voice profile ready
        </motion.div>
      </div>
    </div>
  );
}

// ─── Mini waveform for clips ────────────────────────────────────────────────
// Fixed-width dense waveform that overflows — the clip container clips it.
// This way resizing a clip just reveals/hides bars instead of re-rendering.

const WAVEFORM_BAR_COUNT = 60;

function MiniWaveform({ seed, color }: { seed: number; color: string }) {
  // Deterministic pseudo-random waveform that looks like real speech audio.
  // Uses layered noise at different frequencies for natural envelope + detail.
  const bars = useMemo(() => {
    // Seeded pseudo-random number generator (deterministic per seed)
    let s = seed * 9301 + 49297;
    const rand = () => {
      s = (s * 16807 + 0) % 2147483647;
      return s / 2147483647;
    };

    // Pre-generate random values
    const r = Array.from({ length: WAVEFORM_BAR_COUNT }, () => rand());

    return Array.from({ length: WAVEFORM_BAR_COUNT }, (_, i) => {
      const t = i / WAVEFORM_BAR_COUNT;

      // Slow envelope — broad amplitude shape (words / phrases)
      const envelope =
        0.3 +
        0.35 *
          Math.sin(t * Math.PI * (2 + (seed % 3))) *
          Math.sin(t * Math.PI * (1.3 + seed * 0.7)) +
        0.2 * Math.sin(t * Math.PI * (4.7 + seed * 1.3));

      // Medium variation — syllable-level bumps
      const mid = 0.15 * Math.sin(i * 0.8 + seed * 3.1) * Math.cos(i * 1.3 + seed);

      // High-frequency noise — individual sample jitter
      const noise = (r[i] - 0.5) * 0.25;

      // Combine and clamp
      const raw = envelope + mid + noise;
      return Math.max(0.06, Math.min(1, raw));
    });
  }, [seed]);

  return (
    <div className="flex items-center h-full overflow-hidden">
      {bars.map((h, i) => (
        <div
          key={`w-${seed}-${i}`}
          className="shrink-0 rounded-full opacity-50"
          style={{
            width: 2,
            marginRight: 1,
            height: `${h * 100}%`,
            backgroundColor: color,
          }}
        />
      ))}
    </div>
  );
}

// ─── Animation: Stories Editor ───────────────────────────────────────────────

// Clip shape: id, profile, track, left (px out of 220), width (px), waveform seed
type DemoClip = { id: string; profile: string; track: number; x: number; w: number; seed: number };

const INITIAL_CLIPS: DemoClip[] = [
  { id: 'n1', profile: 'Morgan', track: 0, x: 4, w: 70, seed: 1 },
  { id: 'n2', profile: 'Morgan', track: 0, x: 135, w: 35, seed: 2 },
  { id: 'a1', profile: 'Scarlett', track: 1, x: 25, w: 40, seed: 3 },
  { id: 'a2', profile: 'Scarlett', track: 1, x: 120, w: 35, seed: 4 },
  { id: 'b1', profile: 'Jarvis', track: 2, x: 70, w: 45, seed: 5 },
];

// Timeline width the clips live inside
const TL_W = 220;
// Each action returns a new clips array (or modifies in place)
type Action = { label: string; apply: (clips: DemoClip[]) => DemoClip[] };

const ACTIONS: Action[] = [
  // 0 — move Jarvis clip earlier
  { label: 'Move clip', apply: (c) => c.map((cl) => (cl.id === 'b1' ? { ...cl, x: 55 } : cl)) },
  // 1 — split Morgan's first clip into two with visible gap
  {
    label: 'Split clip',
    apply: (c) => {
      // Idempotent: if n1b already exists, the split already happened
      if (c.some((cl) => cl.id === 'n1b')) return c;
      const clip = c.find((cl) => cl.id === 'n1');
      if (!clip) return c;
      const leftW = 25;
      const gap = 8;
      const rightW = clip.w - leftW - gap;
      return [
        ...c.filter((cl) => cl.id !== 'n1'),
        { ...clip, w: leftW, id: 'n1' },
        {
          id: 'n1b',
          profile: clip.profile,
          track: clip.track,
          x: clip.x + leftW + gap,
          w: rightW,
          seed: 6,
        },
      ];
    },
  },
  // 2 — trim Scarlett's second clip shorter
  { label: 'Trim clip', apply: (c) => c.map((cl) => (cl.id === 'a2' ? { ...cl, w: 25 } : cl)) },
  // 3 — duplicate Jarvis to track 0
  {
    label: 'Duplicate',
    apply: (c) => {
      // Idempotent: if b1d already exists, the duplicate already happened
      if (c.some((cl) => cl.id === 'b1d')) return c;
      const clip = c.find((cl) => cl.id === 'b1');
      if (!clip) return c;
      return [...c, { ...clip, id: 'b1d', track: 0, x: 180, w: 35, seed: 7 }];
    },
  },
  // 4 — reset
  { label: '', apply: () => INITIAL_CLIPS },
];

function StoriesAnimation() {
  const [clips, setClips] = useState<DemoClip[]>(INITIAL_CLIPS);
  const [actionIndex, setActionIndex] = useState(-1);
  const [playheadX, setPlayheadX] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const playheadRef = useRef<ReturnType<typeof requestAnimationFrame>>(0);

  // Animate the playhead continuously
  useEffect(() => {
    let start: number | null = null;
    const speed = 12; // px per second
    const animate = (ts: number) => {
      if (start === null) start = ts;
      const elapsed = (ts - start) / 1000;
      setPlayheadX((elapsed * speed) % TL_W);
      playheadRef.current = requestAnimationFrame(animate);
    };
    playheadRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(playheadRef.current);
  }, []);

  // Step through actions
  useEffect(() => {
    const interval = setInterval(() => {
      setActionIndex((prev) => {
        const next = (prev + 1) % ACTIONS.length;
        setClips((current) => ACTIONS[next].apply(current));
        // Highlight the clip being acted on
        if (next === 0) setSelectedId('b1');
        else if (next === 1) setSelectedId('n1');
        else if (next === 2) setSelectedId('a2');
        else if (next === 3) setSelectedId('b1');
        else setSelectedId(null);
        return next;
      });
    }, 2600);
    return () => clearInterval(interval);
  }, []);

  const trackLabels = ['1', '0', '-1'];
  const timeMarkers = [0, 2, 4, 6, 8];
  const accentColor = 'hsl(43 50% 45%)';
  const accentFg = 'hsl(30 10% 94%)';

  return (
    <div className="h-40 w-full flex flex-col overflow-hidden rounded-md bg-app-darkerBox/50">
      {/* Toolbar */}
      <div className="flex items-center gap-1.5 px-2 py-1 border-b border-app-line bg-app-darkBox/60 shrink-0">
        <div className="w-1.5 h-1.5 rounded-full bg-ink-faint/40" />
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded flex items-center justify-center bg-app-button">
            <div className="border-l-[4px] border-l-ink-faint border-t-[3px] border-t-transparent border-b-[3px] border-b-transparent ml-0.5" />
          </div>
          <div className="w-4 h-4 rounded flex items-center justify-center bg-app-button">
            <div className="w-2 h-2 rounded-sm bg-ink-faint/60" />
          </div>
        </div>
        <span className="text-[8px] text-ink-faint font-mono ml-1 tabular-nums">0:03 / 0:10</span>
        <div className="flex-1" />
        {actionIndex >= 0 && actionIndex < ACTIONS.length - 1 && (
          <motion.span
            key={actionIndex}
            className="text-[7px] font-medium px-1.5 py-0.5 rounded-full"
            style={{
              backgroundColor: `${accentColor.replace(')', ' / 0.15)')}`,
              color: accentColor,
            }}
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            {ACTIONS[actionIndex].label}
          </motion.span>
        )}
        <div className="flex items-center gap-0.5">
          <span className="text-[7px] text-ink-faint">Zoom</span>
          <div className="w-3 h-3 rounded flex items-center justify-center bg-app-button text-[8px] text-ink-faint">
            -
          </div>
          <div className="w-3 h-3 rounded flex items-center justify-center bg-app-button text-[8px] text-ink-faint">
            +
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="flex flex-1 min-h-0">
        {/* Track labels sidebar */}
        <div className="w-7 shrink-0 border-r border-app-line bg-app-darkBox/30 flex flex-col">
          <div className="h-5 border-b border-app-line" />
          {trackLabels.map((label) => (
            <div
              key={label}
              className="flex-1 flex items-center justify-center border-b border-app-line"
            >
              <span className="text-[7px] text-ink-faint select-none">{label}</span>
            </div>
          ))}
        </div>

        {/* Tracks area */}
        <div className="flex-1 relative overflow-hidden flex flex-col">
          {/* Time ruler */}
          <div className="h-5 shrink-0 border-b border-app-line bg-app-darkBox/20 relative">
            {timeMarkers.map((t) => (
              <div
                key={`tm-${t}`}
                className="absolute top-0 h-full flex flex-col justify-end pb-0.5"
                style={{ left: `${(t / 10) * 100}%` }}
              >
                <div className="h-1.5 w-px bg-app-line" />
                <span className="text-[7px] text-ink-faint ml-0.5 select-none">{`0:0${t}`}</span>
              </div>
            ))}
          </div>

          {/* Track rows + clips — same parent so percentages match */}
          <div className="flex-1 relative min-h-0">
            {/* Track rows background */}
            {trackLabels.map((label, i) => (
              <div
                key={`bg-${label}`}
                className="border-b border-app-line absolute left-0 right-0"
                style={{
                  height: `${100 / 3}%`,
                  top: `${(i * 100) / 3}%`,
                  backgroundColor: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)',
                }}
              />
            ))}

            {/* Clips */}
            {clips.map((clip) => {
              const trackIdx = clip.track;
              const isSelected = clip.id === selectedId;
              const clipTop = `calc(${(trackIdx * 100) / 3}% + 2px)`;
              const clipHeight = `calc(${100 / 3}% - 4px)`;
              return (
                <motion.div
                  key={clip.id}
                  className="absolute rounded overflow-hidden"
                  initial={false}
                  style={{
                    height: clipHeight,
                    left: `${(clip.x / TL_W) * 100}%`,
                    width: `${(clip.w / TL_W) * 100}%`,
                    top: clipTop,
                  }}
                  animate={{
                    left: `${(clip.x / TL_W) * 100}%`,
                    width: `${(clip.w / TL_W) * 100}%`,
                    top: clipTop,
                  }}
                  transition={{ type: 'spring', stiffness: 200, damping: 25 }}
                >
                  <div
                    className="w-full h-full rounded overflow-hidden flex flex-col"
                    style={{
                      backgroundColor: isSelected ? 'hsl(43 50% 45%)' : 'hsl(43 45% 40%)',
                      boxShadow: isSelected
                        ? 'inset 0 0 0 1px hsl(43 50% 55%), 0 0 0 1px hsl(30 10% 94% / 0.4)'
                        : 'inset 0 0 0 1px hsl(30 10% 94% / 0.1)',
                    }}
                  >
                    {/* Profile label — scaled to bypass browser min font size */}
                    <div className="shrink-0 relative" style={{ height: 9 }}>
                      <span
                        className="text-[10px] font-medium leading-none absolute top-0 left-0.5 origin-top-left opacity-80 whitespace-nowrap"
                        style={{ color: accentFg, transform: 'scale(0.75)' }}
                      >
                        {clip.profile}
                      </span>
                    </div>
                    {/* Waveform — absolutely positioned so it never affects clip width */}
                    <div className="absolute left-0 right-0 bottom-0" style={{ top: 9 }}>
                      <MiniWaveform seed={clip.seed} color={accentFg} />
                    </div>
                  </div>
                  {/* Trim handles on selected */}
                  {isSelected && (
                    <>
                      <div
                        className="absolute left-0 top-0 bottom-0 w-1 rounded-l"
                        style={{ backgroundColor: 'hsl(30 10% 94% / 0.25)' }}
                      />
                      <div
                        className="absolute right-0 top-0 bottom-0 w-1 rounded-r"
                        style={{ backgroundColor: 'hsl(30 10% 94% / 0.25)' }}
                      />
                    </>
                  )}
                </motion.div>
              );
            })}

            {/* Playhead */}
            <motion.div
              className="absolute top-0 bottom-0 w-[2px] rounded-full z-20 pointer-events-none"
              style={{ backgroundColor: accentColor }}
              animate={{ left: `${(playheadX / TL_W) * 100}%` }}
              transition={{ duration: 0.05, ease: 'linear' }}
            >
              <div
                className="absolute -top-0.5 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full"
                style={{ backgroundColor: accentColor }}
              />
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Animation: Effects Pipeline ────────────────────────────────────────────

function EffectsAnimation() {
  const [activeEffect, setActiveEffect] = useState(0);
  const effects = [
    { name: 'Pitch Shift', param: '-3 semitones', color: '#3b82f6' },
    { name: 'Reverb', param: 'Room 0.7', color: '#8b5cf6' },
    { name: 'Compressor', param: '-15 dB', color: '#ec4899' },
    { name: 'Low-Pass', param: '6000 Hz', color: '#14b8a6' },
  ];

  // Waveform bars — original shape
  const rawBars = [0.3, 0.6, 0.8, 0.5, 0.9, 0.4, 0.7, 0.3, 0.6, 0.5, 0.8, 0.4, 0.7, 0.9, 0.3];

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveEffect((p) => (p + 1) % effects.length);
    }, 2200);
    return () => clearInterval(interval);
  }, [effects.length]);

  return (
    <div className="h-40 w-full flex flex-col items-center justify-center overflow-hidden rounded-md bg-app-darkerBox/50 p-4 gap-3">
      {/* Effects chain */}
      <div className="flex items-center gap-1">
        {effects.map((fx, i) => (
          <div key={fx.name} className="flex items-center gap-1">
            <motion.div
              className="text-[8px] px-2 py-0.5 rounded-full border font-medium"
              animate={{
                borderColor: i <= activeEffect ? `${fx.color}60` : 'rgba(255,255,255,0.06)',
                backgroundColor: i <= activeEffect ? `${fx.color}15` : 'rgba(255,255,255,0.02)',
                color: i <= activeEffect ? fx.color : 'rgba(255,255,255,0.3)',
              }}
              transition={{ duration: 0.3 }}
            >
              {fx.name}
            </motion.div>
            {i < effects.length - 1 && (
              <motion.span
                className="text-[8px]"
                animate={{
                  color: i < activeEffect ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.08)',
                }}
                transition={{ duration: 0.3 }}
              >
                &rarr;
              </motion.span>
            )}
          </div>
        ))}
      </div>

      {/* Waveform that morphs as effects are applied */}
      <div className="flex items-center gap-[2px] h-10 w-full max-w-[200px] justify-center">
        {rawBars.map((h, i) => {
          // Each effect stage progressively transforms the shape
          const shifted = activeEffect >= 0 ? h * (0.7 + 0.3 * Math.sin(i * 0.8)) : h;
          const dampened = activeEffect >= 1 ? shifted * (0.6 + 0.4 * Math.cos(i * 0.3)) : shifted;
          const compressed = activeEffect >= 2 ? 0.3 + dampened * 0.5 : dampened;
          const filtered = activeEffect >= 3 ? compressed * (1 - i * 0.03) : compressed;
          const finalH = Math.max(0.08, Math.min(1, filtered));

          return (
            <motion.div
              key={`bar-${i}`}
              className="w-[3px] rounded-full"
              animate={{
                height: `${finalH * 100}%`,
                backgroundColor: effects[activeEffect].color,
              }}
              transition={{
                height: { duration: 0.5, delay: i * 0.02, ease: 'easeInOut' },
                backgroundColor: { duration: 0.4 },
              }}
            />
          );
        })}
      </div>

      {/* Active effect detail */}
      <motion.div
        className="text-[9px] font-mono text-ink-faint"
        key={activeEffect}
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {effects[activeEffect].name}: {effects[activeEffect].param}
      </motion.div>
    </div>
  );
}

// ─── Animation: Local or Remote ─────────────────────────────────────────────

function LocalRemoteAnimation() {
  const [mode, setMode] = useState(0);
  const modes = ['Local GPU', 'Remote Server'];

  useEffect(() => {
    const interval = setInterval(() => {
      setMode((p) => (p + 1) % 2);
    }, 2800);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-40 w-full flex items-center justify-center overflow-hidden rounded-md bg-app-darkerBox/50 p-4">
      <div className="flex flex-col items-center gap-4 w-full max-w-[180px]">
        {/* Toggle */}
        <div className="flex gap-1 p-0.5 rounded-full border border-app-line bg-app-darkerBox">
          {modes.map((m, i) => (
            <motion.div
              key={m}
              className="text-[9px] px-3 py-1 rounded-full font-medium"
              animate={{
                backgroundColor: i === mode ? 'hsl(43 50% 45%)' : 'transparent',
                color: i === mode ? 'hsl(30 10% 94%)' : 'rgba(255,255,255,0.35)',
              }}
              transition={{ duration: 0.25 }}
            >
              {m}
            </motion.div>
          ))}
        </div>

        {/* Status */}
        <div className="flex flex-col items-center gap-2">
          <motion.div
            className="w-2 h-2 rounded-full"
            animate={{
              backgroundColor: mode === 0 ? '#4ade80' : '#3b82f6',
              boxShadow: mode === 0 ? '0 0 8px #4ade80' : '0 0 8px #3b82f6',
            }}
            transition={{ duration: 0.3 }}
          />
          <span className="text-[9px] text-ink-faint font-mono">
            {mode === 0 ? 'Metal acceleration active' : 'Connected to 192.168.1.50'}
          </span>
          <span className="text-[8px] text-ink-faint/60 font-mono">
            {mode === 0 ? 'VRAM: 8.2 / 16.0 GB' : 'Latency: 12ms | CUDA'}
          </span>
        </div>
      </div>
    </div>
  );
}

// ─── Animation: Transcription ───────────────────────────────────────────────

function TranscriptionAnimation() {
  const [charIndex, setCharIndex] = useState(0);
  const text = 'The quick brown fox jumps over the lazy dog near the riverbank.';

  useEffect(() => {
    const interval = setInterval(() => {
      setCharIndex((p) => {
        if (p >= text.length) return 0;
        return p + 1;
      });
    }, 80);
    return () => clearInterval(interval);
  }, [text.length]);

  return (
    <div className="h-40 w-full flex flex-col items-center justify-center overflow-hidden rounded-md bg-app-darkerBox/50 p-4 gap-3">
      {/* Fake waveform */}
      <div className="flex items-center gap-[1px] h-6 w-full max-w-[180px] justify-center">
        {Array.from({ length: 30 }, (_, i) => {
          const h = 0.2 + 0.8 * Math.abs(Math.sin(i * 0.5 + charIndex * 0.1));
          const active = i < (charIndex / text.length) * 30;
          return (
            <div
              key={i}
              className={`w-[3px] rounded-full transition-colors duration-100 ${
                active ? 'bg-accent' : 'bg-app-line'
              }`}
              style={{ height: `${h * 100}%` }}
            />
          );
        })}
      </div>

      {/* Transcribed text */}
      <div className="text-[10px] text-ink-dull font-mono max-w-[200px] text-center leading-relaxed min-h-[32px]">
        {text.slice(0, charIndex)}
        {charIndex < text.length && (
          <span className="inline-block w-[2px] h-3 bg-accent animate-pulse ml-[1px] align-middle" />
        )}
      </div>
    </div>
  );
}

// ─── Animation: Unlimited Length ─────────────────────────────────────────────

function UnlimitedLengthAnimation() {
  const [phase, setPhase] = useState(0);

  const chunks = [
    'The morning sun crept over the mountains, casting long shadows across the valley below.',
    'Birds stirred in the canopy, their songs weaving through the cool air like threads of gold.',
    'Far below, a river wound its way through ancient stones, carrying whispers of the night.',
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setPhase((p) => (p + 1) % 4); // 0-2 = processing chunks, 3 = crossfade/done
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-40 w-full flex flex-col items-center justify-center overflow-hidden rounded-md bg-app-darkerBox/50 p-4 gap-2.5">
      {/* Chunk pills */}
      <div className="flex flex-col gap-1 w-full max-w-[220px]">
        {chunks.map((chunk, i) => (
          <motion.div
            key={`chunk-${i}`}
            className="flex items-center gap-1.5 px-2 py-1 rounded border text-[8px]"
            animate={{
              borderColor:
                phase === 3
                  ? 'hsl(43 50% 45% / 0.3)'
                  : i === phase
                    ? 'hsl(43 50% 45% / 0.5)'
                    : i < phase
                      ? 'rgba(255,255,255,0.12)'
                      : 'rgba(255,255,255,0.06)',
              backgroundColor:
                phase === 3
                  ? 'hsl(43 50% 45% / 0.04)'
                  : i === phase
                    ? 'hsl(43 50% 45% / 0.08)'
                    : i < phase
                      ? 'rgba(255,255,255,0.04)'
                      : 'rgba(255,255,255,0.02)',
            }}
            transition={{ duration: 0.4 }}
          >
            {/* Status indicator */}
            <motion.div
              className="w-1.5 h-1.5 rounded-full shrink-0"
              animate={{
                backgroundColor:
                  phase === 3
                    ? 'hsl(43 50% 50%)'
                    : i === phase
                      ? 'hsl(43 50% 50%)'
                      : i < phase
                        ? 'rgba(255,255,255,0.3)'
                        : 'rgba(255,255,255,0.1)',
                boxShadow:
                  i === phase && phase < 3 ? '0 0 6px hsl(43 50% 50%)' : '0 0 0px transparent',
              }}
              transition={{ duration: 0.3 }}
            />
            <span
              className={`truncate font-mono ${
                phase === 3 || i <= phase ? 'text-ink-dull' : 'text-ink-faint/50'
              }`}
            >
              {chunk}
            </span>
          </motion.div>
        ))}
      </div>

      {/* Crossfade / result bar */}
      <div className="flex items-center gap-1 w-full max-w-[220px]">
        {chunks.map((_, i) => (
          <motion.div
            key={`seg-${i}`}
            className="h-1.5 flex-1 rounded-full"
            animate={{
              backgroundColor:
                phase === 3
                  ? 'hsl(43 50% 45%)'
                  : i < phase
                    ? 'rgba(255,255,255,0.2)'
                    : i === phase
                      ? 'hsl(43 50% 45% / 0.5)'
                      : 'rgba(255,255,255,0.06)',
            }}
            transition={{ duration: 0.4 }}
          />
        ))}
      </div>

      {/* Status text */}
      <motion.div
        className="text-[9px] font-mono"
        key={phase}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <span className={phase === 3 ? 'text-accent' : 'text-ink-faint'}>
          {phase < 3
            ? `generating chunk ${phase + 1} of ${chunks.length}...`
            : 'crossfaded & ready'}
        </span>
      </motion.div>
    </div>
  );
}

// ─── Feature data ───────────────────────────────────────────────────────────

const FEATURES = [
  {
    title: 'Near-Perfect Voice Cloning',
    description:
      'Multiple TTS engines for exceptional voice quality. Clone any voice from a few seconds of audio with natural intonation and emotion.',
    icon: Mic,
    animation: VoiceCloningAnimation,
  },
  {
    title: 'Stories Editor',
    description:
      'Create multi-voice narratives with a timeline-based editor. Arrange tracks, trim clips, and mix conversations between characters.',
    icon: AudioLines,
    animation: StoriesAnimation,
  },
  {
    title: 'Audio Effects Pipeline',
    description:
      'Apply pitch shift, reverb, delay, compression, and more — then save as presets. Preview effects live and set defaults per voice profile.',
    icon: Sparkles,
    animation: EffectsAnimation,
  },
  {
    title: 'Local or Remote',
    description:
      'Run GPU inference locally with Metal, CUDA, ROCm, Intel Arc, or DirectML — or connect to a remote machine. One-click server setup with automatic discovery.',
    icon: Cloud,
    animation: LocalRemoteAnimation,
  },
  {
    title: 'Audio Transcription',
    description:
      'Powered by Whisper for accurate speech-to-text. Automatically extract reference text from voice samples.',
    icon: MessageSquareText,
    animation: TranscriptionAnimation,
  },
  {
    title: 'Unlimited Generation Length',
    description:
      'Generate up to 50,000 characters in one go. Text is auto-split at sentence boundaries, generated per-chunk, and crossfaded seamlessly.',
    icon: TextCursorInput,
    animation: UnlimitedLengthAnimation,
  },
];

// ─── Feature Card ───────────────────────────────────────────────────────────

function FeatureCard({ feature }: { feature: (typeof FEATURES)[number] }) {
  const Icon = feature.icon;
  const Animation = feature.animation;

  return (
    <div className="rounded-lg border border-app-line bg-app-darkBox overflow-hidden">
      <LazyLoad>
        <div className="pointer-events-none select-none">
          <Animation />
        </div>
      </LazyLoad>
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

// ─── Features Section ───────────────────────────────────────────────────────

export function Features() {
  return (
    <section id="features" className="border-t border-border py-24">
      <div className="mx-auto max-w-7xl px-6">
        <div className="mb-16 text-center">
          <h2 className="text-3xl font-semibold tracking-tight text-foreground md:text-4xl mb-4">
            Professional voice tools, zero compromise
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Everything you need to clone voices, generate speech, and produce multi-voice content —
            running entirely on your machine.
          </p>
        </div>
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((feature) => (
            <FeatureCard key={feature.title} feature={feature} />
          ))}
        </div>
      </div>
    </section>
  );
}
