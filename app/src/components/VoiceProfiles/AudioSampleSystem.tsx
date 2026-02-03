import { HugeiconsIcon } from '@hugeicons/react';
import { Mic01Icon, DeskIcon, PauseIcon, PlayIcon, SquareIcon } from '@hugeicons/core-free-icons';
import { Button } from '@/components/ui/button';
import { FormControl, FormItem, FormMessage } from '@/components/ui/form';
import { formatAudioDuration } from '@/lib/utils/audio';

interface AudioSampleSystemProps {
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
}

export function AudioSampleSystem({
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
}: AudioSampleSystemProps) {
  return (
    <FormItem>
      <FormControl>
        <div className="space-y-4">
          {!isRecording && !file && (
            <div className="flex flex-col items-center justify-center gap-4 p-4 border-2 border-dashed rounded-lg min-h-[180px]">
              <Button type="button" onClick={onStart} size="lg" className="flex items-center gap-2">
                <HugeiconsIcon icon={DeskIcon} size={20} className="h-5 w-5" />
                Start Capture
              </Button>
              <p className="text-sm text-muted-foreground text-center">
                Capture audio from your system. Maximum duration: 30 seconds.
              </p>
            </div>
          )}

          {isRecording && (
            <div className="flex flex-col items-center justify-center gap-4 p-4 border-2 border-destructive rounded-lg bg-destructive/5 min-h-[180px]">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-destructive animate-pulse" />
                  <span className="text-lg font-mono font-semibold">
                    {formatAudioDuration(duration)}
                  </span>
                </div>
              </div>
              <Button
                type="button"
                onClick={onStop}
                variant="destructive"
                className="flex items-center gap-2"
              >
                <HugeiconsIcon icon={SquareIcon} size={16} className="h-4 w-4" />
                Stop Capture
              </Button>
              <p className="text-sm text-muted-foreground text-center">
                {formatAudioDuration(30 - duration)} remaining
              </p>
            </div>
          )}

          {file && !isRecording && (
            <div className="flex flex-col items-center justify-center gap-4 p-4 border-2 border-primary rounded-lg bg-primary/5 min-h-[180px]">
              <div className="flex items-center gap-2">
                <HugeiconsIcon icon={DeskIcon} size={20} className="h-5 w-5 text-primary" />
                <span className="font-medium">Capture complete</span>
              </div>
              <p className="text-sm text-muted-foreground text-center">File: {file.name}</p>
              <div className="flex gap-2">
                <Button type="button" size="icon" variant="outline" onClick={onPlayPause}>
                  {isPlaying ? <HugeiconsIcon icon={PauseIcon} size={16} className="h-4 w-4" /> : <HugeiconsIcon icon={PlayIcon} size={16} className="h-4 w-4" />}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={onTranscribe}
                  disabled={isTranscribing}
                  className="flex items-center gap-2"
                >
                  <HugeiconsIcon icon={Mic01Icon} size={16} className="h-4 w-4" />
                  {isTranscribing ? 'Transcribing...' : 'Transcribe'}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={onCancel}
                  className="flex items-center gap-2"
                >
                  Capture Again
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
