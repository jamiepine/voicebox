'use client';

import { motion } from 'framer-motion';
import {
  AudioLines,
  Box,
  Download,
  Mic,
  MoreHorizontal,
  Pencil,
  Server,
  Sparkles,
  Speaker,
  Star,
  Trash2,
  Volume2,
  Wand2,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { LandingAudioPlayer, unlockAudioContext } from './LandingAudioPlayer';

// ─── Data ───────────────────────────────────────────────────────────────────
// Edit this section to customise all the content shown in the ControlUI demo.

interface VoiceProfile {
  name: string;
  description: string;
  language: string;
  hasEffects: boolean;
}

/** Voice profiles shown in the grid / scroll strip. Index matters — DemoScript references profiles by index. */
const PROFILES: VoiceProfile[] = [
  {
    name: 'Jarvis',
    description: 'Dry wit, composed British AI assistant',
    language: 'en',
    hasEffects: true,
  },
  {
    name: 'Samuel L. Jackson',
    description: 'Commanding intensity with sharp, punchy delivery',
    language: 'en',
    hasEffects: true,
  },
  {
    name: 'Bob Ross',
    description: 'Gentle, soothing voice full of quiet encouragement',
    language: 'en',
    hasEffects: false,
  },
  {
    name: 'Sam Altman',
    description: 'Measured, thoughtful Silicon Valley cadence',
    language: 'en',
    hasEffects: false,
  },
  {
    name: 'Morgan Freeman',
    description: 'Rich, warm baritone with gravitas and calm authority',
    language: 'en',
    hasEffects: false,
  },
  {
    name: 'Linus Tech Tips',
    description: 'Enthusiastic, fast-paced tech explainer energy',
    language: 'en',
    hasEffects: false,
  },
  {
    name: 'Fireship',
    description: 'Rapid-fire, deadpan tech humor with zero filler',
    language: 'en',
    hasEffects: false,
  },
  {
    name: 'Scarlett Johansson',
    description: 'Smooth, low alto with understated warmth',
    language: 'en',
    hasEffects: false,
  },
  {
    name: 'Dario Amodei',
    description: 'Calm, precise articulation with academic depth',
    language: 'en',
    hasEffects: false,
  },
  {
    name: 'David Attenborough',
    description: 'Warm, reverent narration with wonder and precision',
    language: 'en',
    hasEffects: false,
  },
  {
    name: 'Zendaya',
    description: 'Relaxed, modern delivery with effortless cool',
    language: 'en',
    hasEffects: false,
  },
  {
    name: 'Barack Obama',
    description: 'Measured cadence with rhythmic pauses and gravitas',
    language: 'en',
    hasEffects: false,
  },
];

/** Each entry is one cycle of the demo animation: select a profile → type text → generate → play audio. */
interface DemoStep {
  profileIndex: number;
  text: string;
  audioUrl: string;
  engine: string;
  duration: string;
  effect?: string;
}

const DEMO_SCRIPT: DemoStep[] = [
  {
    profileIndex: 0,
    text: 'Sir, I have completed the analysis. Your code has twelve critical vulnerabilities, your coffee is cold, and frankly your commit messages could use some work.',
    audioUrl: '/audio/jarvis.webm',
    engine: 'Qwen 1.7B',
    duration: '0:10',
    effect: 'Robot',
  },
  {
    profileIndex: 4,
    text: "I've narrated penguins, galaxies, and the entire history of mankind. But nothing prepared me for the moment a computer learned to do my job from a five second audio clip.",
    audioUrl: '/audio/morganfreeman.webm',
    engine: 'Qwen 1.7B',
    duration: '0:11',
    effect: 'Radio',
  },
  {
    profileIndex: 3,
    text: "Open source? [laugh] What's that?",
    audioUrl: '/audio/samaltman.webm',
    engine: 'Chatterbox',
    duration: '0:03',
  },
  {
    profileIndex: 1,
    text: "So let me get this straight. You downloaded an app, pressed a button, and now there's two of me? The world was not ready for one",
    audioUrl: '/audio/samjackson.webm',
    engine: 'Qwen 1.7B',
    duration: '0:10',
  },
  {
    profileIndex: 5,
    text: "So we got this voice cloning software and honestly it's kind of terrifying. Like, my wife could not tell the difference. Voicebox dot s h, link in the description!",
    audioUrl: '/audio/linus.webm',
    engine: 'Qwen 1.7B',
    duration: '0:11',
  },
  {
    profileIndex: 6,
    text: 'This is Voicebox in one hundred seconds. It clones voices locally, it runs on your GPU, and no, OpenAI cannot hear you. Lets go.',
    audioUrl: '/audio/fireship.webm',
    engine: 'Qwen 0.6B',
    duration: '0:09',
  },
];

/** History rows pre-populated on first load. Oldest first visually (array index 0 = top row). */
interface Generation {
  id: number;
  profileName: string;
  text: string;
  language: string;
  engine: string;
  duration: string;
  timeAgo: string;
  favorited: boolean;
  versions: number;
}

const INITIAL_GENERATIONS: Generation[] = [
  {
    id: 1,
    profileName: 'Morgan Freeman',
    text: 'The neural pathways of human speech contain more complexity than any language model can fully capture, yet we keep pushing the boundaries of what is possible.',
    language: 'en',
    engine: 'Qwen 1.7B',
    duration: '0:08',
    timeAgo: '2 minutes ago',
    favorited: true,
    versions: 3,
  },
  {
    id: 2,
    profileName: 'Samuel L. Jackson',
    text: 'In a world increasingly shaped by artificial intelligence, the human voice remains our most powerful tool for connection and storytelling.',
    language: 'en',
    engine: 'Qwen 1.7B',
    duration: '0:07',
    timeAgo: '15 minutes ago',
    favorited: false,
    versions: 1,
  },
  {
    id: 3,
    profileName: 'Jarvis',
    text: 'The architecture of modern text-to-speech systems reveals an elegant interplay between transformer models and acoustic feature prediction.',
    language: 'en',
    engine: 'Qwen 0.6B',
    duration: '0:09',
    timeAgo: '1 hour ago',
    favorited: false,
    versions: 2,
  },
  {
    id: 4,
    profileName: 'Bob Ross',
    text: 'Welcome to the next chapter. Every great story begins with a single voice, and today that voice can be yours.',
    language: 'en',
    engine: 'Chatterbox',
    duration: '0:06',
    timeAgo: '3 hours ago',
    favorited: true,
    versions: 1,
  },
  {
    id: 5,
    profileName: 'Linus Tech Tips',
    text: 'Local inference gives you complete control over your voice data. No cloud, no subscriptions, no compromises.',
    language: 'en',
    engine: 'Qwen 1.7B',
    duration: '0:05',
    timeAgo: '5 hours ago',
    favorited: false,
    versions: 1,
  },
];

const SIDEBAR_ITEMS = [
  { icon: Volume2, label: 'Generate' },
  { icon: AudioLines, label: 'Stories' },
  { icon: Mic, label: 'Voices' },
  { icon: Wand2, label: 'Effects' },
  { icon: Speaker, label: 'Audio' },
  { icon: Box, label: 'Models' },
  { icon: Server, label: 'Server' },
];

// ─── Phase system ───────────────────────────────────────────────────────────

type Phase = 'idle' | 'selecting' | 'typing' | 'generating' | 'complete' | 'playing';

const PHASE_DURATIONS: Record<Phase, number> = {
  idle: 2500,
  selecting: 800,
  typing: 6000,
  generating: 2800,
  complete: 1200,
  playing: 4000,
};

// ─── Typewriter ─────────────────────────────────────────────────────────────

function TypewriterText({ text, speed }: { text: string; speed?: number }) {
  // Default: fill the typing phase duration, leaving 500ms buffer at the end
  const resolvedSpeed =
    speed ?? Math.max(20, Math.floor((PHASE_DURATIONS.typing - 500) / text.length));
  const [displayed, setDisplayed] = useState('');
  const indexRef = useRef(0);

  useEffect(() => {
    indexRef.current = 0;
    setDisplayed('');
    const interval = setInterval(() => {
      indexRef.current += 1;
      if (indexRef.current <= text.length) {
        setDisplayed(text.slice(0, indexRef.current));
      } else {
        clearInterval(interval);
      }
    }, resolvedSpeed);
    return () => clearInterval(interval);
  }, [text, resolvedSpeed]);

  return (
    <>
      {displayed}
      <span className="inline-block h-3.5 w-[2px] animate-pulse bg-foreground/70 ml-[1px] align-middle" />
    </>
  );
}

// ─── Loading bars (simplified react-loaders replacement) ────────────────────

function LoadingBars({ mode }: { mode: 'idle' | 'generating' | 'playing' }) {
  const barColor = mode !== 'idle' ? 'bg-accent' : 'bg-muted-foreground/40';
  return (
    <div className="flex items-center gap-[2px] h-5">
      {[0, 1, 2, 3, 4].map((i) => (
        <motion.div
          key={`${i}-${mode}`}
          className={`w-[3px] rounded-full ${barColor}`}
          animate={
            mode === 'generating'
              ? { height: ['6px', '16px', '6px'] }
              : mode === 'playing'
                ? { height: ['8px', '14px', '4px', '12px', '8px'] }
                : { height: '8px' }
          }
          transition={
            mode === 'generating'
              ? { duration: 0.6, repeat: Infinity, delay: i * 0.08, ease: 'easeInOut' }
              : mode === 'playing'
                ? { duration: 1.2, repeat: Infinity, delay: i * 0.15, ease: 'easeInOut' }
                : {}
          }
        />
      ))}
    </div>
  );
}

// ─── Profile Card ───────────────────────────────────────────────────────────

const ProfileCard = ({
  profile,
  selected,
  selecting,
  cardRef,
}: {
  profile: VoiceProfile;
  selected: boolean;
  selecting: boolean;
  cardRef?: React.Ref<HTMLDivElement>;
}) => {
  return (
    <motion.div
      ref={cardRef}
      className={`rounded-xl border-2 bg-card p-3.5 flex flex-col h-[143px] transition-all duration-200 ${
        selected ? 'border-accent shadow-md' : 'border-border/50 hover:shadow-sm'
      } ${selecting && !selected ? 'opacity-60' : ''}`}
      animate={selecting && selected ? { scale: [1, 1.02, 1] } : {}}
      transition={{ duration: 0.3 }}
    >
      <div className="text-[15px] font-bold leading-tight line-clamp-2">{profile.name}</div>
      <div className="text-[10px] text-muted-foreground line-clamp-2 leading-relaxed mt-1">
        {profile.description}
      </div>
      <div className="flex items-center gap-1.5 mt-2">
        <span className="text-[10px] px-1.5 py-0.5 rounded-md border border-border text-muted-foreground">
          {profile.language}
        </span>
        {profile.hasEffects && <Sparkles className="h-3 w-3 text-accent fill-accent" />}
      </div>
      <div className="flex items-center gap-1 mt-auto justify-end">
        <Download className="h-3.5 w-3.5 text-muted-foreground/40" />
        <Pencil className="h-3.5 w-3.5 text-muted-foreground/40" />
        <Trash2 className="h-3.5 w-3.5 text-muted-foreground/40" />
      </div>
    </motion.div>
  );
};

// ─── History Row ────────────────────────────────────────────────────────────

function HistoryRow({
  gen,
  mode,
  isNew,
}: {
  gen: Generation;
  mode: 'idle' | 'generating' | 'playing';
  isNew: boolean;
}) {
  return (
    <motion.div
      className={`border rounded-md transition-colors text-left w-full ${
        mode === 'playing' ? 'bg-muted/70' : 'bg-card'
      }`}
      initial={isNew ? { opacity: 0, y: -8 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      <div className="flex items-stretch gap-3 h-[80px] p-2.5">
        {/* Status icon */}
        <div className="w-8 flex items-center justify-center shrink-0">
          <LoadingBars mode={mode} />
        </div>

        {/* Meta info */}
        <div className="flex flex-col gap-1 w-36 shrink-0 justify-center">
          <div className="text-[12px] font-medium truncate">{gen.profileName}</div>
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
            <span>{gen.language}</span>
            <span>{gen.engine}</span>
            {mode !== 'generating' && <span>{gen.duration}</span>}
          </div>
          <div className="text-[10px] text-muted-foreground">
            {mode === 'generating' ? (
              <span className="text-accent">Generating...</span>
            ) : (
              gen.timeAgo
            )}
          </div>
        </div>

        {/* Transcript */}
        <div className="flex-1 min-w-0 flex items-center">
          <div className="text-[11px] text-muted-foreground line-clamp-3 leading-relaxed">
            {gen.text}
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex flex-col justify-center items-center gap-0.5 shrink-0">
          <button className="h-5 w-5 flex items-center justify-center rounded-sm hover:bg-muted">
            <Star
              className={`h-2.5 w-2.5 ${
                gen.favorited ? 'text-accent fill-accent' : 'text-muted-foreground/50'
              }`}
            />
          </button>
          {gen.versions > 1 && (
            <button className="h-5 w-5 flex items-center justify-center rounded-sm hover:bg-muted">
              <AudioLines className="h-2.5 w-2.5 text-muted-foreground/50" />
            </button>
          )}
          <button className="h-5 w-5 flex items-center justify-center rounded-sm hover:bg-muted">
            <MoreHorizontal className="h-2.5 w-2.5 text-muted-foreground/50" />
          </button>
        </div>
      </div>
    </motion.div>
  );
}

// ─── Floating Generate Box ──────────────────────────────────────────────────

function FloatingGenerateBox({
  phase,
  typingText,
  selectedProfile,
  engine,
  effect,
}: {
  phase: Phase;
  typingText: string;
  selectedProfile: VoiceProfile | null;
  engine: string;
  effect?: string;
}) {
  const isFocused = phase === 'typing' || phase === 'generating';
  const isGenerating = phase === 'generating';

  return (
    <motion.div
      className="bg-background/30 backdrop-blur-2xl border border-accent/20 rounded-[1.5rem] shadow-2xl p-2.5"
      animate={{
        borderColor: isGenerating
          ? 'hsl(43 50% 45% / 0.35)'
          : isFocused
            ? 'hsl(43 50% 45% / 0.25)'
            : 'hsl(43 50% 45% / 0.15)',
      }}
      transition={{ duration: 0.3 }}
    >
      {/* Text area + generate button */}
      <div className="flex items-start gap-2">
        <div className="flex-1 min-w-0">
          <motion.div
            className="overflow-hidden"
            animate={{ height: isFocused ? 100 : 32 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
          >
            <div
              className="text-[12.5px] text-muted-foreground/60 px-2 py-1 leading-relaxed"
              style={{ minHeight: isFocused ? 100 : 32 }}
            >
              {phase === 'typing' ? (
                <span className="text-foreground">
                  <TypewriterText text={typingText} />
                </span>
              ) : phase === 'generating' ? (
                <span className="text-muted-foreground/40">{typingText}</span>
              ) : (
                <span>
                  {selectedProfile
                    ? `Generate speech using ${selectedProfile.name}...`
                    : 'Select a voice profile above...'}
                </span>
              )}
            </div>
          </motion.div>
        </div>

        {/* Generate button */}
        <button className="h-8 w-8 rounded-full bg-accent flex items-center justify-center shrink-0 shadow-lg">
          <Sparkles className="h-3.5 w-3.5 text-white fill-white" />
        </button>
      </div>

      {/* Bottom selectors */}
      <div className="flex items-center gap-1.5 mt-2">
        <span className="text-[10px] px-2 py-1 rounded-full border border-border bg-card text-muted-foreground">
          English
        </span>
        <span className="text-[10px] px-2 py-1 rounded-full border border-border bg-card text-muted-foreground">
          {engine}
        </span>
        <span
          className={`text-[10px] px-2 py-1 rounded-full border flex items-center gap-1 ${
            effect
              ? 'border-accent/30 bg-accent/10 text-accent'
              : 'border-border bg-card text-muted-foreground'
          }`}
        >
          <Sparkles className={`h-2.5 w-2.5 ${effect ? 'fill-accent' : ''}`} />
          {effect || 'Effect'}
        </span>
      </div>
    </motion.div>
  );
}

// ─── Main ControlUI ─────────────────────────────────────────────────────────

export function ControlUI() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [selectedIndex, setSelectedIndex] = useState(DEMO_SCRIPT[0].profileIndex);
  const [cycle, setCycle] = useState(0);
  const [newGenId, setNewGenId] = useState<number | null>(null);
  const [generations, setGenerations] = useState<Generation[]>([...INITIAL_GENERATIONS]);
  const [isMuted, setIsMuted] = useState(true);
  const [isVisible, setIsVisible] = useState(true);
  const [pageHidden, setPageHidden] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const phaseRef = useRef(phase);
  const mobileCardRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const desktopCardRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const profileGridRef = useRef<HTMLDivElement>(null);
  const [scrollLeft, setScrollLeft] = useState(0);
  phaseRef.current = phase;

  const step = DEMO_SCRIPT[cycle % DEMO_SCRIPT.length];
  const selectedProfile = PROFILES[selectedIndex];

  // Scroll to selected profile card — accounts for generate box overlay on desktop
  useEffect(() => {
    const isMobile = window.innerWidth < 768;

    if (isMobile) {
      const el = mobileCardRefs.current.get(selectedIndex);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
      return;
    }

    // Desktop
    const el = desktopCardRefs.current.get(selectedIndex);
    const scrollContainer = profileGridRef.current;
    if (!el || !scrollContainer) return;

    const containerTop = scrollContainer.getBoundingClientRect().top;
    const elTop = el.getBoundingClientRect().top;
    const elRelTop = elTop - containerTop + scrollContainer.scrollTop;

    const rowHeight = 145;
    const generateBoxHeight = 200;
    const visibleTop = scrollContainer.scrollTop;
    const visibleBottom = visibleTop + scrollContainer.clientHeight - generateBoxHeight;
    const elRelBottom = elRelTop + el.offsetHeight;

    if (elRelTop >= visibleTop && elRelBottom <= visibleBottom) {
      return;
    }

    const target = elRelTop - rowHeight;
    scrollContainer.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
  }, [selectedIndex]);

  // Visibility detection
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => setIsVisible(entry.isIntersecting), {
      threshold: 0,
    });
    if (containerRef.current) observer.observe(containerRef.current);

    const handleVisibility = () => setPageHidden(document.visibilityState !== 'visible');
    document.addEventListener('visibilitychange', handleVisibility);

    return () => {
      observer.disconnect();
      document.removeEventListener('visibilitychange', handleVisibility);
    };
  }, []);

  const paused = !isVisible || pageHidden;

  // Phase cycling — `playing` phase is driven by audio finish, not a timeout
  useEffect(() => {
    if (paused || phase === 'playing') return;

    const duration = PHASE_DURATIONS[phase];
    const timer = setTimeout(() => {
      console.log(
        '[ControlUI] phase transition',
        phase,
        '→ next, cycle:',
        cycle,
        'step profile:',
        PROFILES[step.profileIndex].name,
      );
      switch (phase) {
        case 'idle': {
          setSelectedIndex(step.profileIndex);
          setPhase('selecting');
          break;
        }
        case 'selecting':
          setPhase('typing');
          break;
        case 'typing': {
          const profile = PROFILES[step.profileIndex];
          const newGen: Generation = {
            id: Date.now(),
            profileName: profile.name,
            text: step.text,
            language: profile.language,
            engine: step.engine,
            duration: step.duration,
            timeAgo: 'just now',
            favorited: false,
            versions: 1,
          };
          setGenerations((prev) => [newGen, ...prev.slice(0, 5)]);
          setNewGenId(newGen.id);
          setPhase('generating');
          break;
        }
        case 'generating':
          setPhase('playing');
          break;
      }
    }, duration);

    return () => clearTimeout(timer);
  }, [phase, paused, step, cycle]);

  const handleAudioFinish = useCallback(() => {
    if (phaseRef.current !== 'playing') return;
    setPhase('idle');
    setCycle((c) => c + 1);
    setNewGenId(null);
  }, []);

  const isGenerating = phase === 'generating';

  return (
    <div ref={containerRef} className="relative z-20 mx-auto w-full max-w-6xl px-6">
      {/* Unmute button with handwritten hint */}
      <div className="flex justify-end mb-3">
        <div className="relative">
          {/* Handwritten hint — absolutely positioned above the button */}
          {isMuted && (
            <motion.div
              className="absolute select-none pointer-events-none"
              style={{ top: -30, right: 100 }}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 2, duration: 0.6, ease: 'easeOut' }}
            >
              <span
                className="text-xl text-accent/80 whitespace-nowrap"
                style={{
                  fontFamily: "'Caveat', 'Segoe Script', 'Comic Sans MS', cursive",
                  letterSpacing: '0.02em',
                }}
              >
                try me!
              </span>
              {/* Curved arrow from text down-right toward the button */}
              <svg
                width="22"
                height="11"
                viewBox="0 0 80 40"
                fill="none"
                className="text-accent/70 absolute"
                style={{ top: 14, left: 60 }}
                aria-hidden="true"
              >
                <title>Arrow</title>
                <path
                  d="M4 4 C20 4, 40 8, 55 20 C62 26, 66 32, 70 36"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeLinecap="round"
                  fill="none"
                />
                <path
                  d="M58 42 L70 36 L64 22"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  transform="rotate(35, 70, 36)"
                  fill="none"
                />
              </svg>
            </motion.div>
          )}
          <button
            onClick={() => {
              unlockAudioContext();
              setIsMuted(!isMuted);
            }}
            className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-border bg-card/50 backdrop-blur text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {isMuted ? (
              <>
                <Volume2 className="h-3.5 w-3.5" />
                <span>Unmute</span>
              </>
            ) : (
              <>
                <Volume2 className="h-3.5 w-3.5 text-accent" />
                <span>Mute</span>
              </>
            )}
          </button>
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-app-line bg-app-box shadow-[0_25px_60px_rgba(0,0,0,0.5),0_8px_20px_rgba(0,0,0,0.3)] md:h-[640px] pointer-events-none select-none">
        <div className="flex flex-col md:flex-row h-full">
          {/* ── Sidebar (hidden on mobile) ─────────────────────────── */}
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
                <img
                  src="/voicebox-logo-app.webp"
                  alt=""
                  className="w-full h-full object-contain"
                />
              </div>
            </div>

            {/* Nav items */}
            <div className="flex flex-col gap-2">
              {SIDEBAR_ITEMS.map((item, i) => {
                const Icon = item.icon;
                const active = i === 0;
                return (
                  <div
                    key={item.label}
                    className={`w-9 h-9 rounded-full flex items-center justify-center transition-all duration-200 ${
                      active
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
            <div className="mt-auto text-[8px] text-muted-foreground/40">v0.2.0</div>
          </div>

          {/* ── Main content ──────────────────────────────────────── */}
          <div className="flex-1 flex flex-col md:flex-row min-w-0 relative">
            {/* Left: Profiles + Generate box */}
            <div className="flex flex-col min-w-0 relative md:flex-1 md:overflow-hidden">
              {/* Gradient fade overlay — sits between header and scroll content */}
              <div className="hidden md:block absolute top-0 left-0 right-0 h-16 bg-gradient-to-b from-app-box to-transparent z-[1] pointer-events-none" />

              {/* Header — floats above everything */}
              <div className="absolute top-0 left-0 right-0 z-10 px-4 pt-4 md:pt-6 pb-2 flex items-center justify-between">
                <h2 className="text-base font-bold">Voicebox</h2>
                <div className="flex items-center gap-1.5">
                  <button className="h-6 text-[10px] px-2.5 rounded-full border border-border bg-card text-muted-foreground flex items-center gap-1">
                    Import Voice
                  </button>
                  <button className="h-6 text-[10px] px-2.5 rounded-full bg-accent text-accent-foreground flex items-center">
                    Create Voice
                  </button>
                </div>
              </div>

              {/* Scrollable profile cards — scrolls behind header + gradient */}
              <div
                ref={profileGridRef}
                className="flex-1 min-h-0 md:overflow-y-auto md:pt-14 pt-12"
              >
                <div className="px-4">
                  {/* Mobile: horizontal scroll strip with edge fade */}
                  <div className="relative md:hidden">
                    {scrollLeft > 0 && (
                      <div className="absolute left-0 top-0 bottom-0 w-6 bg-gradient-to-r from-app-box to-transparent z-10" />
                    )}
                    <div className="absolute right-0 top-0 bottom-0 w-6 bg-gradient-to-l from-app-box to-transparent z-10" />
                    <div
                      className="flex gap-2 overflow-x-auto pb-2"
                      onScroll={(e) => setScrollLeft(e.currentTarget.scrollLeft)}
                    >
                      {PROFILES.map((profile, i) => (
                        <div
                          key={profile.name}
                          className="shrink-0 w-[140px]"
                          ref={(el) => {
                            if (el) mobileCardRefs.current.set(i, el);
                          }}
                        >
                          <ProfileCard
                            profile={profile}
                            selected={i === selectedIndex}
                            selecting={phase === 'selecting'}
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Desktop: 3-col grid */}
                  <div className="hidden md:grid grid-cols-3 gap-2 mt-1 pb-44">
                    {PROFILES.map((profile, i) => (
                      <ProfileCard
                        key={profile.name}
                        profile={profile}
                        selected={i === selectedIndex}
                        selecting={phase === 'selecting'}
                        cardRef={(el: HTMLDivElement | null) => {
                          if (el) desktopCardRefs.current.set(i, el);
                        }}
                      />
                    ))}
                  </div>
                </div>
              </div>

              {/* Floating generate box — desktop: absolute overlay, mobile: inline */}
              <div className="px-3 pt-2 pb-3 md:pt-0 md:absolute md:left-4 md:right-4 md:bottom-[117px] md:z-20 md:pb-0 md:px-0">
                <FloatingGenerateBox
                  phase={phase}
                  typingText={step.text}
                  selectedProfile={selectedProfile}
                  engine={step.engine}
                  effect={step.effect}
                />
              </div>
            </div>

            {/* Right/Below: History */}
            <div className="md:w-[48%] shrink-0 flex flex-col min-w-0 border-t md:border-t-0 border-app-line">
              <div className="max-h-[360px] md:max-h-none flex-1 overflow-hidden px-3 pt-3 md:pt-6 pb-3">
                <div className="flex flex-col gap-2">
                  {generations.map((gen) => {
                    const isThisNew = gen.id === newGenId;
                    const rowMode: 'idle' | 'generating' | 'playing' =
                      isThisNew && isGenerating
                        ? 'generating'
                        : isThisNew && phase === 'playing'
                          ? 'playing'
                          : 'idle';
                    return <HistoryRow key={gen.id} gen={gen} mode={rowMode} isNew={isThisNew} />;
                  })}
                </div>
              </div>
            </div>

            {/* Audio player */}
            <LandingAudioPlayer
              audioUrl={step.audioUrl}
              title={selectedProfile.name}
              playing={phase === 'playing'}
              muted={isMuted}
              onFinish={handleAudioFinish}
              onClose={() => {}}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
