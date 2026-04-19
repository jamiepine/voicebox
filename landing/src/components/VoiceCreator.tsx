'use client';

import { AnimatePresence, motion } from 'framer-motion';
import { Mic, Monitor, Upload } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

// ─── Waveform bars generator ────────────────────────────────────────────────

function generateWaveformBars(count: number, seed: number): number[] {
  const bars: number[] = [];
  for (let i = 0; i < count; i++) {
    const x = i / count;
    // Speech-like envelope: ramp up, sustain, taper
    const envelope = Math.sin(x * Math.PI) * 0.8 + 0.2;
    // Layered pseudo-random noise
    const n1 = Math.sin(seed * 127.1 + i * 43.7) * 0.5 + 0.5;
    const n2 = Math.sin(seed * 269.5 + i * 17.3) * 0.3 + 0.5;
    const n3 = Math.sin(seed * 53.9 + i * 97.1) * 0.2 + 0.5;
    const noise = (n1 + n2 + n3) / 3;
    bars.push(envelope * noise);
  }
  return bars;
}

// ─── Animated waveform background ───────────────────────────────────────────

function WaveformBackground({ active }: { active: boolean }) {
  const bars = useMemo(() => generateWaveformBars(60, 42), []);

  return (
    <div className="absolute inset-0 pointer-events-none flex items-end justify-center overflow-hidden">
      <div className="flex items-end gap-[2px] w-full h-full px-4 pb-4">
        {bars.map((h, i) => {
          const maxH = 120; // max bar height in px
          const baseH = 4;
          const activeH = baseH + h * maxH;
          const idleH = baseH + h * maxH * 0.25;
          return (
            <motion.div
              key={i}
              className="flex-1 rounded-full bg-accent"
              animate={{
                opacity: active ? 0.35 : 0.1,
                height: active ? [idleH, activeH, idleH * 1.5, activeH * 0.7, idleH] : idleH,
              }}
              transition={
                active
                  ? {
                      duration: 1.0 + (i % 5) * 0.12,
                      repeat: Infinity,
                      repeatType: 'mirror',
                      delay: (i % 7) * 0.04,
                      ease: 'easeInOut',
                    }
                  : { duration: 0.6 }
              }
            />
          );
        })}
      </div>
    </div>
  );
}

// ─── Tab content panels ─────────────────────────────────────────────────────

function UploadPanel() {
  const [hasFile, setHasFile] = useState(false);

  useEffect(() => {
    // Simulate file drop after 2s
    const t1 = setTimeout(() => setHasFile(true), 2000);
    const t2 = setTimeout(() => setHasFile(false), 5000);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, []);

  return (
    <div
      className={`relative flex flex-col items-center justify-center gap-3 p-6 border-2 rounded-lg min-h-[180px] transition-colors duration-300 ${
        hasFile ? 'border-accent bg-accent/5' : 'border-dashed border-muted-foreground/25'
      }`}
    >
      <AnimatePresence mode="wait">
        {!hasFile ? (
          <motion.div
            key="idle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center gap-3"
          >
            <div className="h-10 px-5 rounded-md bg-accent text-accent-foreground flex items-center gap-2 text-sm font-medium">
              <Upload className="h-4 w-4" />
              Choose File
            </div>
            <p className="text-xs text-muted-foreground text-center">
              Drag and drop an audio file, or click to browse.
              <br />
              Maximum duration: 30 seconds.
            </p>
          </motion.div>
        ) : (
          <motion.div
            key="file"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center gap-3"
          >
            <div className="flex items-center gap-2">
              <Upload className="h-4 w-4 text-accent" />
              <span className="text-sm font-medium">sample-voice-clip.wav</span>
            </div>
            <div className="flex gap-2">
              <div className="h-8 px-3 rounded-md border border-border flex items-center gap-1.5 text-xs text-muted-foreground">
                <span>0:04</span>
              </div>
              <div className="h-8 px-3 rounded-md border border-border flex items-center gap-1.5 text-xs text-muted-foreground">
                <Mic className="h-3 w-3" />
                Transcribe
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function RecordPanel() {
  const [state, setState] = useState<'idle' | 'recording' | 'done'>('idle');
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const t1 = setTimeout(() => setState('recording'), 1500);
    const t2 = setTimeout(() => setState('done'), 5500);
    const t3 = setTimeout(() => {
      setState('idle');
      setElapsed(0);
    }, 8000);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, []);

  // Timer
  useEffect(() => {
    if (state !== 'recording') return;
    setElapsed(0);
    const interval = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(interval);
  }, [state]);

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;

  return (
    <div
      className={`relative flex flex-col items-center justify-center gap-3 p-6 border-2 rounded-lg min-h-[180px] overflow-hidden transition-colors duration-300 ${
        state === 'recording'
          ? 'border-accent bg-accent/5'
          : state === 'done'
            ? 'border-accent bg-accent/5'
            : 'border-dashed border-muted-foreground/25'
      }`}
    >
      <WaveformBackground active={state === 'recording'} />

      <AnimatePresence mode="wait">
        {state === 'idle' && (
          <motion.div
            key="idle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="relative z-10 flex flex-col items-center gap-3"
          >
            <div className="h-10 px-5 rounded-md bg-accent text-accent-foreground flex items-center gap-2 text-sm font-medium">
              <Mic className="h-4 w-4" />
              Start Recording
            </div>
            <p className="text-xs text-muted-foreground text-center">
              Click to record from your microphone.
              <br />
              Maximum duration: 30 seconds.
            </p>
          </motion.div>
        )}

        {state === 'recording' && (
          <motion.div
            key="recording"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="relative z-10 flex flex-col items-center gap-3"
          >
            <div className="flex items-center gap-3">
              <div className="h-3 w-3 rounded-full bg-accent animate-pulse" />
              <span className="text-lg font-mono font-semibold">{formatTime(elapsed)}</span>
            </div>
            <div className="h-9 px-4 rounded-md bg-accent text-accent-foreground flex items-center gap-2 text-sm font-medium">
              <div className="h-3 w-3 rounded-sm bg-accent-foreground" />
              Stop Recording
            </div>
            <p className="text-xs text-muted-foreground">{formatTime(30 - elapsed)} remaining</p>
          </motion.div>
        )}

        {state === 'done' && (
          <motion.div
            key="done"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="relative z-10 flex flex-col items-center gap-3"
          >
            <div className="flex items-center gap-2">
              <Mic className="h-4 w-4 text-accent" />
              <span className="text-sm font-medium">Recording complete</span>
            </div>
            <div className="flex gap-2">
              <div className="h-8 px-3 rounded-md border border-border flex items-center gap-1.5 text-xs text-muted-foreground">
                <span>0:04</span>
              </div>
              <div className="h-8 px-3 rounded-md border border-border flex items-center gap-1.5 text-xs text-muted-foreground">
                <Mic className="h-3 w-3" />
                Transcribe
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function SystemPanel() {
  const [state, setState] = useState<'idle' | 'capturing' | 'done'>('idle');
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const t1 = setTimeout(() => setState('capturing'), 1500);
    const t2 = setTimeout(() => setState('done'), 5500);
    const t3 = setTimeout(() => {
      setState('idle');
      setElapsed(0);
    }, 8000);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, []);

  useEffect(() => {
    if (state !== 'capturing') return;
    setElapsed(0);
    const interval = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(interval);
  }, [state]);

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;

  return (
    <div
      className={`relative flex flex-col items-center justify-center gap-3 p-6 border-2 rounded-lg min-h-[180px] overflow-hidden transition-colors duration-300 ${
        state === 'capturing'
          ? 'border-accent bg-accent/5'
          : state === 'done'
            ? 'border-accent bg-accent/5'
            : 'border-dashed border-muted-foreground/25'
      }`}
    >
      <WaveformBackground active={state === 'capturing'} />

      <AnimatePresence mode="wait">
        {state === 'idle' && (
          <motion.div
            key="idle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="relative z-10 flex flex-col items-center gap-3"
          >
            <div className="h-10 px-5 rounded-md bg-accent text-accent-foreground flex items-center gap-2 text-sm font-medium">
              <Monitor className="h-4 w-4" />
              Start Capture
            </div>
            <p className="text-xs text-muted-foreground text-center">
              Capture audio playing on your system.
              <br />
              Maximum duration: 30 seconds.
            </p>
          </motion.div>
        )}

        {state === 'capturing' && (
          <motion.div
            key="capturing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="relative z-10 flex flex-col items-center gap-3"
          >
            <div className="flex items-center gap-3">
              <div className="h-3 w-3 rounded-full bg-accent animate-pulse" />
              <span className="text-lg font-mono font-semibold">{formatTime(elapsed)}</span>
            </div>
            <div className="h-9 px-4 rounded-md bg-accent text-accent-foreground flex items-center gap-2 text-sm font-medium">
              <div className="h-3 w-3 rounded-sm bg-accent-foreground" />
              Stop Capture
            </div>
            <p className="text-xs text-muted-foreground">{formatTime(30 - elapsed)} remaining</p>
          </motion.div>
        )}

        {state === 'done' && (
          <motion.div
            key="done"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="relative z-10 flex flex-col items-center gap-3"
          >
            <div className="flex items-center gap-2">
              <Monitor className="h-4 w-4 text-accent" />
              <span className="text-sm font-medium">Capture complete</span>
            </div>
            <div className="flex gap-2">
              <div className="h-8 px-3 rounded-md border border-border flex items-center gap-1.5 text-xs text-muted-foreground">
                <span>0:04</span>
              </div>
              <div className="h-8 px-3 rounded-md border border-border flex items-center gap-1.5 text-xs text-muted-foreground">
                <Mic className="h-3 w-3" />
                Transcribe
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Tab selector ───────────────────────────────────────────────────────────

const TABS = [
  { id: 'upload' as const, label: 'Upload', icon: Upload },
  { id: 'record' as const, label: 'Microphone', icon: Mic },
  { id: 'system' as const, label: 'System Audio', icon: Monitor },
];

type TabId = (typeof TABS)[number]['id'];

// ─── Main section ───────────────────────────────────────────────────────────

export function VoiceCreator() {
  const [activeTab, setActiveTab] = useState<TabId>('record');
  const [cycleKey, setCycleKey] = useState(0);

  // Auto-cycle tabs
  useEffect(() => {
    const tabOrder: TabId[] = ['record', 'upload', 'system'];
    let idx = tabOrder.indexOf(activeTab);

    const interval = setInterval(() => {
      idx = (idx + 1) % tabOrder.length;
      setActiveTab(tabOrder[idx]);
      setCycleKey((k) => k + 1);
    }, 9000);

    return () => clearInterval(interval);
  }, [activeTab]);

  return (
    <section className="border-t border-border py-24">
      <div className="mx-auto max-w-5xl px-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 md:gap-16 items-center">
          {/* Left: Copy */}
          <div>
            <h2 className="text-3xl font-semibold tracking-tight text-foreground md:text-4xl mb-4">
              Clone any voice in seconds
            </h2>
            <p className="text-muted-foreground mb-6">
              Three ways to capture a voice sample. Upload a clip, record from your microphone, or
              capture audio playing on your system. Voicebox clones the voice from as little as 3
              seconds of audio.
            </p>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="h-8 w-8 rounded-lg bg-accent/10 flex items-center justify-center shrink-0 mt-0.5">
                  <Upload className="h-4 w-4 text-accent" />
                </div>
                <div>
                  <div className="text-sm font-medium">Upload a clip</div>
                  <div className="text-xs text-muted-foreground">
                    Drag and drop any audio file — WAV, MP3, FLAC, or WebM.
                  </div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="h-8 w-8 rounded-lg bg-accent/10 flex items-center justify-center shrink-0 mt-0.5">
                  <Mic className="h-4 w-4 text-accent" />
                </div>
                <div>
                  <div className="text-sm font-medium">Record from microphone</div>
                  <div className="text-xs text-muted-foreground">
                    Live waveform preview while you record. Up to 30 seconds.
                  </div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="h-8 w-8 rounded-lg bg-accent/10 flex items-center justify-center shrink-0 mt-0.5">
                  <Monitor className="h-4 w-4 text-accent" />
                </div>
                <div>
                  <div className="text-sm font-medium">System audio capture</div>
                  <div className="text-xs text-muted-foreground">
                    Clone a voice from a YouTube video, podcast, or any app playing audio.
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right: Animated UI mock */}
          <div className="rounded-xl border border-app-line bg-app-darkBox overflow-hidden pointer-events-none select-none">
            <div className="p-5">
              {/* Tab bar */}
              <div className="flex rounded-lg border border-border bg-card/50 p-1 mb-4">
                {TABS.map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => {
                        setActiveTab(tab.id);
                        setCycleKey((k) => k + 1);
                      }}
                      className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                        isActive
                          ? 'bg-background text-foreground shadow-sm'
                          : 'text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      <Icon className="h-3.5 w-3.5" />
                      <span className="hidden sm:inline">{tab.label}</span>
                    </button>
                  );
                })}
              </div>

              {/* Panel */}
              <AnimatePresence mode="wait">
                <motion.div
                  key={`${activeTab}-${cycleKey}`}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.2 }}
                >
                  {activeTab === 'upload' && <UploadPanel />}
                  {activeTab === 'record' && <RecordPanel />}
                  {activeTab === 'system' && <SystemPanel />}
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
