import { memo, useEffect, useRef, useState } from 'react';
import { Download, Mic, Pause, Play, RotateCcw, Square, Star, Trash2, Upload } from 'lucide-react';
import { Visualizer } from 'react-sound-visualizer';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { FinetuneSampleResponse } from '@/lib/api/types';
import type { LanguageCode } from '@/lib/constants/languages';
import {
  useAddFinetuneSample,
  useDeleteFinetuneSample,
  useImportProfileSamples,
  useSetRefAudio,
} from '@/lib/hooks/useFinetune';
import { useProfileSamples } from '@/lib/hooks/useProfiles';
import { useAudioRecording } from '@/lib/hooks/useAudioRecording';
import { useTranscription } from '@/lib/hooks/useTranscription';
import { formatAudioDuration } from '@/lib/utils/audio';
import { cn } from '@/lib/utils/cn';

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

interface FinetuneSampleManagerProps {
  profileId: string;
  profileLanguage?: string;
  samples: FinetuneSampleResponse[];
}

export function FinetuneSampleManager({
  profileId,
  profileLanguage,
  samples,
}: FinetuneSampleManagerProps) {
  const { toast } = useToast();
  const addSample = useAddFinetuneSample();
  const deleteSample = useDeleteFinetuneSample();
  const importSamples = useImportProfileSamples();
  const setRefAudio = useSetRefAudio();
  const transcription = useTranscription();
  const { data: profileSamples } = useProfileSamples(profileId);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [showRecorder, setShowRecorder] = useState(false);
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);

  // Review dialog state
  const [reviewFile, setReviewFile] = useState<File | null>(null);
  const [reviewTranscript, setReviewTranscript] = useState('');
  const [isReviewOpen, setIsReviewOpen] = useState(false);
  const [isPreviewPlaying, setIsPreviewPlaying] = useState(false);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const previewUrlRef = useRef<string | null>(null);

  const {
    isRecording,
    duration,
    error: recordingError,
    startRecording,
    stopRecording,
    cancelRecording,
  } = useAudioRecording({
    maxDurationSeconds: 59,
    onRecordingComplete: (blob) => {
      const file = new File([blob], `finetune-${Date.now()}.wav`, {
        type: 'audio/wav',
      });
      // Don't upload immediately — open review dialog
      setReviewFile(file);
      setReviewTranscript('');
      setIsReviewOpen(true);
      setShowRecorder(false);

      // Stop waveform stream
      if (audioStream) {
        audioStream.getTracks().forEach((t) => t.stop());
        setAudioStream(null);
      }

      // Auto-transcribe
      transcription.mutate(
        { file, language: (profileLanguage as LanguageCode) || undefined },
        {
          onSuccess: (result) => {
            setReviewTranscript(result.text);
          },
          onError: () => {
            // User can still type manually
          },
        },
      );
    },
  });

  // Cleanup preview URL on unmount
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  const totalDuration = samples.reduce((sum, s) => sum + s.duration_seconds, 0);

  const getDatasetHealth = () => {
    if (samples.length >= 120 && totalDuration >= 1200) return { label: 'Excellent', color: 'bg-green-500' };
    if (samples.length >= 30 && totalDuration >= 300) return { label: 'Adequate', color: 'bg-yellow-500' };
    return { label: 'Need more samples', color: 'bg-red-500' };
  };

  const health = getDatasetHealth();

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    for (const file of Array.from(files)) {
      try {
        await addSample.mutateAsync({ profileId, file });
        toast({ title: 'Sample added', description: `${file.name} uploaded and auto-transcribed` });
      } catch (error) {
        toast({
          title: 'Upload failed',
          description: error instanceof Error ? error.message : 'Failed to upload sample',
          variant: 'destructive',
        });
      }
    }
    e.target.value = '';
  };

  const handleStartRecording = async () => {
    setShowRecorder(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setAudioStream(stream);
    } catch {
      // Waveform won't show but recording can still work
    }
    startRecording();
  };

  const handleStopRecording = () => {
    stopRecording();
  };

  const handleCancelRecording = () => {
    cancelRecording();
    setShowRecorder(false);
    if (audioStream) {
      audioStream.getTracks().forEach((t) => t.stop());
      setAudioStream(null);
    }
  };

  // Review dialog actions
  const handleReviewSave = async () => {
    if (!reviewFile) return;
    const transcript = reviewTranscript.trim();
    if (!transcript) {
      toast({ title: 'Transcript required', description: 'Please enter or edit the transcript before saving.', variant: 'destructive' });
      return;
    }
    try {
      await addSample.mutateAsync({ profileId, file: reviewFile, transcript });
      toast({ title: 'Sample saved', description: 'Recording added to training set' });
      handleReviewClose();
    } catch (error) {
      toast({
        title: 'Save failed',
        description: error instanceof Error ? error.message : 'Failed to save sample',
        variant: 'destructive',
      });
    }
  };

  const handleReviewRetranscribe = () => {
    if (!reviewFile) return;
    transcription.mutate(
      { file: reviewFile, language: (profileLanguage as LanguageCode) || undefined },
      {
        onSuccess: (result) => {
          setReviewTranscript(result.text);
        },
        onError: () => {
          toast({ title: 'Transcription failed', description: 'Please type the transcript manually.', variant: 'destructive' });
        },
      },
    );
  };

  const handleReviewPlayPause = () => {
    if (!reviewFile) return;

    if (isPreviewPlaying) {
      previewAudioRef.current?.pause();
      setIsPreviewPlaying(false);
      return;
    }

    // Create object URL for playback
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
    }
    const url = URL.createObjectURL(reviewFile);
    previewUrlRef.current = url;

    const audio = new Audio(url);
    audio.onended = () => setIsPreviewPlaying(false);
    previewAudioRef.current = audio;
    audio.play();
    setIsPreviewPlaying(true);
  };

  const handleReviewClose = () => {
    setIsReviewOpen(false);
    setReviewFile(null);
    setReviewTranscript('');
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current = null;
    }
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setIsPreviewPlaying(false);
  };

  const handleImport = async () => {
    try {
      const result = await importSamples.mutateAsync({ profileId });
      toast({
        title: 'Samples imported',
        description: `${result.length} samples imported from profile`,
      });
    } catch (error) {
      toast({
        title: 'Import failed',
        description: error instanceof Error ? error.message : 'Failed to import samples',
        variant: 'destructive',
      });
    }
  };

  const handlePlay = (sampleId: string) => {
    if (playingId === sampleId) {
      audioRef.current?.pause();
      setPlayingId(null);
      return;
    }

    if (audioRef.current) {
      audioRef.current.pause();
    }

    const audio = new Audio(apiClient.getFinetuneSampleAudioUrl(sampleId));
    audio.onended = () => setPlayingId(null);
    audioRef.current = audio;
    audio.play();
    setPlayingId(sampleId);
  };

  const handleDelete = async (sampleId: string) => {
    try {
      await deleteSample.mutateAsync({ profileId, sampleId });
    } catch (error) {
      toast({
        title: 'Delete failed',
        description: error instanceof Error ? error.message : 'Failed to delete sample',
        variant: 'destructive',
      });
    }
  };

  const handleSetRef = async (sampleId: string) => {
    try {
      await setRefAudio.mutateAsync({ profileId, sampleId });
      toast({ title: 'Reference audio set' });
    } catch (error) {
      toast({
        title: 'Failed to set reference',
        description: error instanceof Error ? error.message : 'Error',
        variant: 'destructive',
      });
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Training Samples</h3>
          <div className="flex items-center gap-3 mt-1 text-sm text-muted-foreground">
            <span>{samples.length} samples</span>
            <span>{Math.floor(totalDuration / 60)}m {Math.floor(totalDuration % 60)}s total</span>
            <div className="flex items-center gap-1.5">
              <div className={cn('w-2 h-2 rounded-full', health.color)} />
              <span>{health.label}</span>
            </div>
          </div>
        </div>

        {!showRecorder && (
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleStartRecording}
            >
              <Mic className="h-4 w-4 mr-1" />
              Record
            </Button>
            <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
              <Upload className="h-4 w-4 mr-1" />
              Upload
            </Button>
            {profileSamples && profileSamples.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleImport}
                disabled={importSamples.isPending}
              >
                <Download className="h-4 w-4 mr-1" />
                Import ({profileSamples.length})
              </Button>
            )}
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        multiple
        className="hidden"
        onChange={handleFileUpload}
      />

      {/* Recording UI */}
      {showRecorder && (
        <div className="relative flex flex-col items-center justify-center gap-4 p-4 border-2 border-accent rounded-lg bg-accent/5 min-h-[180px] overflow-hidden">
          {audioStream && <MemoizedWaveform audioStream={audioStream} />}

          {isRecording ? (
            <>
              <div className="relative z-10 flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-accent animate-pulse" />
                  <span className="text-lg font-mono font-semibold">
                    {formatAudioDuration(duration)}
                  </span>
                </div>
              </div>
              <div className="relative z-10 flex gap-2">
                <Button
                  onClick={handleStopRecording}
                  className="flex items-center gap-2 bg-accent text-accent-foreground hover:bg-accent/90"
                >
                  <Square className="h-4 w-4" />
                  Stop Recording
                </Button>
                <Button variant="outline" onClick={handleCancelRecording}>
                  Cancel
                </Button>
              </div>
              <p className="relative z-10 text-sm text-muted-foreground text-center">
                {formatAudioDuration(60 - duration)} remaining
              </p>
            </>
          ) : (
            <>
              <div className="relative z-10 flex items-center gap-2">
                <Mic className="h-5 w-5 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Preparing...</span>
              </div>
              <Button variant="outline" onClick={handleCancelRecording} className="relative z-10">
                Cancel
              </Button>
            </>
          )}

          {recordingError && (
            <p className="relative z-10 text-sm text-destructive">{recordingError}</p>
          )}
        </div>
      )}

      {/* Review Dialog — shown after recording stops */}
      <Dialog open={isReviewOpen} onOpenChange={(open) => { if (!open) handleReviewClose(); }}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Review Recording</DialogTitle>
            <DialogDescription>
              Listen to your recording and verify the transcript before saving.
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-col gap-4 pt-2">
            {/* Playback */}
            <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
              <Button
                variant="outline"
                size="icon"
                className="h-10 w-10 shrink-0"
                onClick={handleReviewPlayPause}
              >
                {isPreviewPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <div className="flex flex-col">
                <span className="text-sm font-medium">
                  {isPreviewPlaying ? 'Playing...' : 'Click to preview'}
                </span>
                <span className="text-xs text-muted-foreground">
                  {reviewFile?.name}
                </span>
              </div>
            </div>

            {/* Transcript */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Transcript</label>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={handleReviewRetranscribe}
                  disabled={transcription.isPending}
                >
                  <RotateCcw className="h-3 w-3 mr-1" />
                  {transcription.isPending ? 'Transcribing...' : 'Re-transcribe'}
                </Button>
              </div>

              {transcription.isPending && !reviewTranscript ? (
                <div className="flex items-center justify-center p-4 rounded-lg border border-dashed">
                  <span className="text-sm text-muted-foreground animate-pulse">
                    Transcribing audio...
                  </span>
                </div>
              ) : (
                <Textarea
                  value={reviewTranscript}
                  onChange={(e) => setReviewTranscript(e.target.value)}
                  placeholder="Type or edit the transcript here..."
                  rows={4}
                  className="resize-none"
                />
              )}
            </div>

            {/* Actions */}
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="outline" onClick={handleReviewClose}>
                Discard
              </Button>
              <Button
                onClick={handleReviewSave}
                disabled={addSample.isPending || !reviewTranscript.trim()}
              >
                {addSample.isPending ? 'Saving...' : 'Save Sample'}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Samples Table */}
      {samples.length === 0 && !showRecorder ? (
        <div className="text-center py-12 text-muted-foreground border border-dashed rounded-lg">
          <Mic className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p>No training samples yet</p>
          <p className="text-sm mt-1">Record, upload, or import samples to get started</p>
        </div>
      ) : samples.length > 0 ? (
        <div className="border rounded-lg overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-10" />
                <TableHead>Transcript</TableHead>
                <TableHead className="w-24">Duration</TableHead>
                <TableHead className="w-20">Ref</TableHead>
                <TableHead className="w-16" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {samples.map((sample) => (
                <TableRow key={sample.id}>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0"
                      onClick={() => handlePlay(sample.id)}
                    >
                      {playingId === sample.id ? (
                        <Square className="h-3 w-3" />
                      ) : (
                        <Play className="h-3 w-3" />
                      )}
                    </Button>
                  </TableCell>
                  <TableCell className="max-w-[300px] truncate text-sm">
                    {sample.transcript}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {sample.duration_seconds.toFixed(1)}s
                  </TableCell>
                  <TableCell>
                    {sample.is_ref_audio ? (
                      <Badge variant="default" className="text-xs">Ref</Badge>
                    ) : (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 px-1 text-xs text-muted-foreground"
                        onClick={() => handleSetRef(sample.id)}
                        title="Set as reference audio"
                      >
                        <Star className="h-3 w-3" />
                      </Button>
                    )}
                  </TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                      onClick={() => handleDelete(sample.id)}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      ) : null}
    </div>
  );
}
