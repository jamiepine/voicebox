import { Mic, Pause, Play, Square } from 'lucide-react';
import { memo, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Visualizer } from 'react-sound-visualizer';
import { MicrophonePermissionNotice } from '@/components/MicrophoneGate/MicrophoneGate';
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
          <canvas ref={canvasRef} width={500} height={150} className="w-full h-full" />
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
  onTranscribe: () => void;
  onPlayPause: () => void;
  isPlaying: boolean;
  isTranscribing?: boolean;
  showWaveform?: boolean;
  /** The parent's last recording attempt failed with a denied microphone. */
  permissionDenied?: boolean;
}

/**
 * The record-a-sample surface shared by the Create Voice and Add Sample
 * dialogs: a mic button that becomes a live waveform, then playback and
 * transcribe controls once a clip exists.
 *
 * When the microphone is denied it swaps the record button for
 * {@link MicrophonePermissionNotice}, which is the only affordance the user
 * gets: macOS does not re-prompt once a denial is on record.
 */
export function AudioSampleRecording({
  file,
  isRecording,
  duration,
  onStart,
  onStop,
  onCancel,
  onTranscribe,
  onPlayPause,
  isPlaying,
  isTranscribing = false,
  showWaveform = true,
  permissionDenied = false,
}: AudioSampleRecordingProps) {
  const { t } = useTranslation();
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
  // The visualizer's own mic request is the first thing to hit the denial —
  // catching it here means the notice is up before the user clicks Record,
  // rather than after a failed attempt.
  const [visualizerDenied, setVisualizerDenied] = useState(false);
  // Retrying clears the denial flags, so the notice unmounts and remounts on
  // every attempt. Remember here, where the state survives, that an attempt
  // has already been made and come back denied.
  const [retriedWhileDenied, setRetriedWhileDenied] = useState(false);

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
      .catch((err: unknown) => {
        console.warn('Could not access microphone for visualization:', err);
        const name = err instanceof Error ? err.name : '';
        if (name === 'NotAllowedError' || name === 'SecurityError') {
          setVisualizerDenied(true);
        }
      });

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, [showWaveform]);

  const micDenied = permissionDenied || visualizerDenied;

  return (
    <FormItem>
      <FormControl>
        <div className="space-y-4">
          {!isRecording && !file && (
            <div className="relative flex flex-col items-center justify-center gap-4 p-4 border-2 border-dashed rounded-lg min-h-[180px] overflow-hidden">
              {micDenied ? (
                <MicrophonePermissionNotice
                  showRestartHint={retriedWhileDenied}
                  onRetry={() => {
                    setRetriedWhileDenied(true);
                    setVisualizerDenied(false);
                    onStart();
                  }}
                />
              ) : (
                <>
                  {showWaveform && audioStream && <MemoizedWaveform audioStream={audioStream} />}
                  <Button
                    type="button"
                    onClick={onStart}
                    size="lg"
                    className="relative z-10 flex items-center gap-2"
                  >
                    <Mic className="h-5 w-5" />
                    {t('audioSample.startRecording')}
                  </Button>
                  <p className="relative z-10 text-sm text-muted-foreground text-center">
                    {t('audioSample.recordHint')}
                  </p>
                </>
              )}
            </div>
          )}

          {isRecording && (
            <div className="relative flex flex-col items-center justify-center gap-4 p-4 border-2 border-accent rounded-lg bg-accent/5 min-h-[180px] overflow-hidden">
              {showWaveform && audioStream && <MemoizedWaveform audioStream={audioStream} />}
              <div className="relative z-10 flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-accent animate-pulse" />
                  <span className="text-lg font-mono font-semibold">
                    {formatAudioDuration(duration)}
                  </span>
                </div>
              </div>
              <Button
                type="button"
                onClick={onStop}
                className="relative z-10 flex items-center gap-2 bg-accent text-accent-foreground hover:bg-accent/90"
              >
                <Square className="h-4 w-4" />
                {t('audioSample.stopRecording')}
              </Button>
              <p className="relative z-10 text-sm text-muted-foreground text-center">
                {t('audioSample.remaining', { time: formatAudioDuration(30 - duration) })}
              </p>
            </div>
          )}

          {file && !isRecording && (
            <div className="flex flex-col items-center justify-center gap-4 p-4 border-2 border-primary rounded-lg bg-primary/5 min-h-[180px]">
              <div className="flex items-center gap-2">
                <Mic className="h-5 w-5 text-primary" />
                <span className="font-medium">{t('audioSample.recordingComplete')}</span>
              </div>
              <p className="text-sm text-muted-foreground text-center">
                {t('audioSample.fileLabel', { name: file.name })}
              </p>
              <div className="flex gap-2">
                <Button
                  type="button"
                  size="icon"
                  variant="outline"
                  onClick={onPlayPause}
                  aria-label={isPlaying ? t('audioSample.pause') : t('audioSample.play')}
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={onTranscribe}
                  disabled={isTranscribing}
                  className="flex items-center gap-2"
                >
                  <Mic className="h-4 w-4" />
                  {isTranscribing ? t('audioSample.transcribing') : t('audioSample.transcribe')}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={onCancel}
                  className="flex items-center gap-2"
                >
                  {t('audioSample.recordAgain')}
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
