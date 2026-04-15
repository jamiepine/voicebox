import { ChevronRight, Mic, Pause, Play, Square } from 'lucide-react';
import { memo, useEffect, useRef, useState } from 'react';
import { Visualizer } from 'react-sound-visualizer';
import { Button } from '@/components/ui/button';
import { FormControl, FormItem, FormMessage } from '@/components/ui/form';
import { formatAudioDuration } from '@/lib/utils/audio';

const MemoizedWaveform = memo(function MemoizedWaveform({
  audioStream,
}: {
  audioStream: MediaStream;
}) {
  return (
    <div className="absolute inset-0 pointer-events-none flex items-center justify-center opacity-30">
      <Visualizer audio={audioStream} autoStart strokeColor="#b39a3d">
        {({ canvasRef }) => (
          <canvas
            ref={canvasRef}
            width={500}
            height={150}
            className="w-full h-full"
          />
        )}
      </Visualizer>
    </div>
  );
});

interface AudioSampleRecordingProps {
  file: File | null | undefined;
  isRecording: boolean;
  duration: number;
  onStart: () => void;
  onStop: () => void;
  onCancel: () => void;
  onPlayPause: () => void;
  isPlaying: boolean;
  showWaveform?: boolean;
}

export const SCRIPT_LINES = [
  { cue: 'Neutral, natural', text: 'Hola, este es un ejemplo de voz para captura.' },
  { cue: 'Curious, rising intonation', text: '¿Puedes ver cómo el murciélago vuela mientras como kiwi y cardillo?' },
  { cue: 'Slight surprise, higher pitch', text: '¡Qué extraño y fascinante suena todo esto!' },
  { cue: 'Lower, slow, controlled', text: 'Ahora hablo más despacio, con un tono más bajo y relajado.' },
  { cue: 'Rising energy and speed', text: 'Y ahora cambio el ritmo, hablo más rápido y con mayor claridad.' },
  { cue: 'Soft, almost whisper', text: 'Esta es una prueba en voz baja, tranquila y controlada.' },
  { cue: 'Firm, projected', text: 'Y esta es mi voz con más fuerza y proyección.' },
  { cue: 'Relaxed, natural close', text: 'Finalmente, cierro esta grabación de forma clara y natural.' },
] as const;

const SECS_PER_LINE = 40 / SCRIPT_LINES.length; // 5s per line

export function AudioSampleRecording({
  file,
  isRecording,
  duration,
  onStart,
  onStop,
  onCancel,
  onPlayPause,
  isPlaying,
  showWaveform = true,
}: AudioSampleRecordingProps) {
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
  const [currentLineIndex, setCurrentLineIndex] = useState(0);
  const currentLineRef = useRef<HTMLDivElement>(null);

  // Request microphone access when component mounts
  useEffect(() => {
    if (!showWaveform) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;

    let stream: MediaStream | null = null;

    navigator.mediaDevices
      .getUserMedia({ audio: true, video: false })
      .then((s) => {
        stream = s;
        setAudioStream(s);
      })
      .catch((err) => {
        console.warn('Could not access microphone for visualization:', err);
      });

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, [showWaveform]);

  // Reset line index when recording starts
  useEffect(() => {
    if (isRecording) {
      setCurrentLineIndex(0);
    }
  }, [isRecording]);

  // Auto-advance line based on elapsed time
  useEffect(() => {
    if (!isRecording) return;
    const autoIndex = Math.min(
      Math.floor(duration / SECS_PER_LINE),
      SCRIPT_LINES.length - 1,
    );
    setCurrentLineIndex(autoIndex);
  }, [isRecording, duration]);

  // Scroll current line into view when it changes
  useEffect(() => {
    if (isRecording && currentLineRef.current) {
      currentLineRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [currentLineIndex, isRecording]);

  const handleAdvanceLine = () => {
    setCurrentLineIndex((prev) => Math.min(prev + 1, SCRIPT_LINES.length - 1));
  };

  return (
    <FormItem>
      <FormControl>
        <div className="space-y-4">
          {/* PRE-RECORDING: full guide preview */}
          {!isRecording && !file && (
            <div className="relative flex flex-col items-center gap-4 p-4 border-2 border-dashed rounded-lg overflow-hidden">
              {showWaveform && audioStream && (
                <MemoizedWaveform audioStream={audioStream} />
              )}

              <div className="relative z-10 w-full max-w-md space-y-3">
                <p className="text-sm font-medium text-center">Recording Guide</p>
                <div className="text-xs space-y-2 bg-muted/50 rounded-lg p-3 max-h-[200px] overflow-y-auto">
                  <p className="text-muted-foreground italic">Read naturally with emotional intention:</p>

                  <div className="space-y-1.5">
                    {SCRIPT_LINES.map((line, i) => (
                      <p key={i}>
                        <span className="text-muted-foreground">[{line.cue}]</span>
                        <br />
                        {line.text}
                      </p>
                    ))}
                  </div>

                  <div className="pt-1 border-t border-border/50 space-y-0.5 text-muted-foreground">
                    <p>⚠ Don't read like a tongue twister — say it with real intention</p>
                    <p>⚠ Vary volume, rhythm and emotion between lines</p>
                    <p>⚠ Brief pauses between sections</p>
                  </div>
                </div>
              </div>

              <Button
                type="button"
                onClick={onStart}
                size="lg"
                className="relative z-10 flex items-center gap-2"
              >
                <Mic className="h-5 w-5" />
                Start Recording
              </Button>
              <p className="relative z-10 text-sm text-muted-foreground text-center">
                Click to start recording. Maximum duration: 40 seconds.
              </p>
            </div>
          )}

          {/* RECORDING: interactive guide */}
          {isRecording && (
            <div className="relative flex flex-col items-center gap-4 p-4 border-2 border-accent rounded-lg bg-accent/5 overflow-hidden">
              {showWaveform && audioStream && (
                <MemoizedWaveform audioStream={audioStream} />
              )}

              {/* Timer row */}
              <div className="relative z-10 flex items-center justify-between w-full max-w-md">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-accent animate-pulse" />
                  <span className="text-lg font-mono font-semibold">
                    {formatAudioDuration(duration)}
                  </span>
                </div>
                <span className="text-sm text-muted-foreground font-mono">
                  {formatAudioDuration(40 - duration)} left
                </span>
              </div>

              {/* Progress dots */}
              <div className="relative z-10 flex gap-1.5 items-center">
                {SCRIPT_LINES.map((_, i) => (
                  <div
                    key={i}
                    className={
                      i < currentLineIndex
                        ? 'h-1.5 w-1.5 rounded-full bg-accent/40'
                        : i === currentLineIndex
                          ? 'h-2.5 w-2.5 rounded-full bg-accent ring-2 ring-accent/30'
                          : 'h-1.5 w-1.5 rounded-full bg-muted-foreground/20'
                    }
                  />
                ))}
                <span className="ml-1 text-xs text-muted-foreground">
                  {currentLineIndex + 1}/{SCRIPT_LINES.length}
                </span>
              </div>

              {/* Scrollable script */}
              <div className="relative z-10 w-full max-w-md max-h-[260px] overflow-y-auto space-y-2 py-1 px-1">
                {SCRIPT_LINES.map((line, i) => {
                  const isCurrent = i === currentLineIndex;
                  const isPast = i < currentLineIndex;

                  return (
                    <div
                      key={i}
                      ref={isCurrent ? currentLineRef : null}
                      className={[
                        'rounded-lg px-3 py-2 transition-all duration-300',
                        isCurrent
                          ? 'bg-accent/15 border border-accent/40 shadow-sm'
                          : isPast
                            ? 'opacity-30'
                            : 'opacity-50',
                      ].join(' ')}
                    >
                      <p
                        className={[
                          'text-xs mb-0.5',
                          isCurrent ? 'text-accent font-medium' : 'text-muted-foreground',
                        ].join(' ')}
                      >
                        [{line.cue}]
                      </p>
                      <p
                        className={[
                          'transition-all duration-300',
                          isCurrent
                            ? 'text-base font-medium leading-snug'
                            : isPast
                              ? 'text-sm line-through text-muted-foreground'
                              : 'text-sm text-muted-foreground',
                        ].join(' ')}
                      >
                        {line.text}
                      </p>
                    </div>
                  );
                })}
              </div>

              {/* Tips */}
              <div className="relative z-10 w-full max-w-md text-xs text-muted-foreground space-y-0.5 border-t border-border/30 pt-2">
                <p>⚠ Say it with real intention, not like a tongue twister</p>
                <p>⚠ Vary volume, rhythm and emotion between lines</p>
              </div>

              {/* Controls */}
              <div className="relative z-10 flex items-center gap-3">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleAdvanceLine}
                  disabled={currentLineIndex >= SCRIPT_LINES.length - 1}
                  className="flex items-center gap-1"
                >
                  Next
                  <ChevronRight className="h-4 w-4" />
                </Button>
                <Button
                  type="button"
                  onClick={onStop}
                  className="flex items-center gap-2 bg-accent text-accent-foreground hover:bg-accent/90"
                >
                  <Square className="h-4 w-4" />
                  Stop Recording
                </Button>
              </div>
            </div>
          )}

          {/* POST-RECORDING: completion state — unchanged */}
          {file && !isRecording && (
            <div className="flex flex-col items-center justify-center gap-4 p-4 border-2 border-primary rounded-lg bg-primary/5 min-h-[180px]">
              <div className="flex items-center gap-2">
                <Mic className="h-5 w-5 text-primary" />
                <span className="font-medium">Recording complete</span>
              </div>
              <p className="text-sm text-muted-foreground text-center">File: {file.name}</p>
              <p className="text-xs text-muted-foreground">Transcript auto-filled from guide</p>
              <div className="flex gap-2">
                <Button
                  type="button"
                  size="icon"
                  variant="outline"
                  onClick={onPlayPause}
                  aria-label={isPlaying ? 'Pause' : 'Play'}
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={onCancel}
                  className="flex items-center gap-2"
                >
                  Record Again
                </Button>
              </div>
            </div>
          )}
        </div>
      </FormControl>
      <FormMessage />
    </FormItem>
  );
}
