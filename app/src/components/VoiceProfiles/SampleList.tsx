import { HugeiconsIcon } from '@hugeicons/react';
import { CheckmarkCircle01Icon, Edit01Icon, PauseIcon, PlayIcon, Add01Icon, Delete01Icon, VolumeHighIcon, Cancel01Icon } from '@hugeicons/core-free-icons';
import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { CircleButton } from '@/components/ui/circle-button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { useDeleteSample, useProfileSamples, useUpdateSample } from '@/lib/hooks/useProfiles';
import { formatAudioDuration } from '@/lib/utils/audio';
import { cn } from '@/lib/utils/cn';
import { SampleUpload } from './SampleUpload';

interface MiniSamplePlayerProps {
  audioUrl: string;
}

function MiniSamplePlayer({ audioUrl }: MiniSamplePlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const audio = new Audio(audioUrl);
    audioRef.current = audio;

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
      setIsLoading(false);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);

    return () => {
      audio.pause();
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.src = '';
    };
  }, [audioUrl]);

  const handlePlayPause = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
  };

  const handleSeek = (value: number[]) => {
    if (!audioRef.current || duration === 0) return;
    const progress = value[0] / 100;
    audioRef.current.currentTime = progress * duration;
  };

  const handleStop = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setIsPlaying(false);
    setCurrentTime(0);
  };

  return (
    <div className="border-t bg-muted/30 px-3 py-2 mt-2">
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0"
          onClick={handlePlayPause}
          disabled={isLoading}
        >
          {isPlaying ? <HugeiconsIcon icon={PauseIcon} size={14} className="h-3.5 w-3.5" /> : <HugeiconsIcon icon={PlayIcon} size={14} className="h-3.5 w-3.5 ml-0.5" />}
        </Button>

        <div className="flex-1 min-w-0 flex items-center gap-2">
          <Slider
            value={duration > 0 ? [(currentTime / duration) * 100] : [0]}
            onValueChange={handleSeek}
            max={100}
            step={0.1}
            className="flex-1"
          />
          <div className="flex items-center gap-1 text-xs text-muted-foreground shrink-0 min-w-[70px]">
            <span className="font-mono">{formatAudioDuration(currentTime)}</span>
            <span>/</span>
            <span className="font-mono">{formatAudioDuration(duration)}</span>
          </div>
        </div>

        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0"
          onClick={handleStop}
          title="Stop"
        >
          <HugeiconsIcon icon={Cancel01Icon} size={14} className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  );
}

interface SampleListProps {
  profileId: string;
}

export function SampleList({ profileId }: SampleListProps) {
  const { data: samples, isLoading } = useProfileSamples(profileId);
  const deleteSample = useDeleteSample();
  const updateSample = useUpdateSample();
  const { toast } = useToast();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [editingSampleId, setEditingSampleId] = useState<string | null>(null);
  const [editedText, setEditedText] = useState<string>('');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [sampleToDelete, setSampleToDelete] = useState<string | null>(null);

  const handleDeleteClick = (sampleId: string) => {
    setSampleToDelete(sampleId);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (sampleToDelete) {
      deleteSample.mutate(sampleToDelete);
      setDeleteDialogOpen(false);
      setSampleToDelete(null);
    }
  };

  const handleStartEdit = (sampleId: string, currentText: string) => {
    setEditingSampleId(sampleId);
    setEditedText(currentText);
  };

  const handleCancelEdit = () => {
    setEditingSampleId(null);
    setEditedText('');
  };

  const handleSaveEdit = async (sampleId: string) => {
    if (!editedText.trim()) {
      toast({
        title: 'Invalid text',
        description: 'Reference text cannot be empty.',
        variant: 'destructive',
      });
      return;
    }

    try {
      await updateSample.mutateAsync({ sampleId, referenceText: editedText.trim() });
      toast({
        title: 'Sample updated',
        description: 'Reference text has been updated successfully.',
      });
      setEditingSampleId(null);
      setEditedText('');
    } catch (error) {
      toast({
        title: 'Update failed',
        description: error instanceof Error ? error.message : 'Failed to update sample',
        variant: 'destructive',
      });
    }
  };

  if (isLoading) {
    return <div className="text-sm text-muted-foreground">Loading samples...</div>;
  }

  return (
    <div className="space-y-4 pt-4">
      {samples && samples.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-8 text-center border border-dashed rounded-lg">
          <HugeiconsIcon icon={VolumeHighIcon} size={32} className="h-8 w-8 text-muted-foreground/50 mb-2" />
          <p className="text-sm text-muted-foreground">No samples yet</p>
          <p className="text-xs text-muted-foreground/70 mt-1">
            Add your first audio sample to get started
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {samples?.map((sample, index) => {
            const isEditing = editingSampleId === sample.id;

            return (
              <div
                key={sample.id}
                className={cn(
                  'group relative rounded-lg border bg-card transition-all duration-200',
                  isEditing ? 'ring-2 ring-primary/20' : 'hover:border-primary/30',
                )}
              >
                {isEditing ? (
                  /* Edit Mode */
                  <div className="p-4 space-y-3">
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
                      <HugeiconsIcon icon={Edit01Icon} size={12} className="h-3 w-3" />
                      <span>Editing transcription</span>
                    </div>
                    <Textarea
                      value={editedText}
                      onChange={(e) => setEditedText(e.target.value)}
                      className="min-h-[100px] text-sm resize-none"
                      placeholder="Enter reference text..."
                      autoFocus
                    />
                    <div className="flex items-center justify-end gap-2 pt-1">
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        onClick={handleCancelEdit}
                        disabled={updateSample.isPending}
                      >
                        <HugeiconsIcon icon={Cancel01Icon} size={16} className="h-4 w-4 mr-1" />
                        Cancel
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        onClick={() => handleSaveEdit(sample.id)}
                        disabled={updateSample.isPending}
                      >
                        <HugeiconsIcon icon={CheckmarkCircle01Icon} size={16} className="h-4 w-4 mr-1" />
                        {updateSample.isPending ? 'Saving...' : 'Save'}
                      </Button>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* View Mode */}
                    <div className="flex items-center gap-3 p-3 h-[72px]">
                      {/* Text Content */}
                      <div className="flex-1 min-w-0 py-0.5">
                        <p className="text-sm font-medium line-clamp-2 leading-snug">
                          {sample.reference_text}
                        </p>
                      </div>

                      {/* Action Buttons */}
                      <div className="shrink-0 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                        <CircleButton
                          icon={(props) => <HugeiconsIcon icon={Edit01Icon} size={14} {...props} />}
                          title="Edit transcription"
                          onClick={() => handleStartEdit(sample.id, sample.reference_text)}
                        />
                        <CircleButton
                          icon={(props) => <HugeiconsIcon icon={Delete01Icon} size={14} {...props} />}
                          title="Delete sample"
                          onClick={() => handleDeleteClick(sample.id)}
                          disabled={deleteSample.isPending}
                        />
                      </div>

                      {/* Sample Number Badge */}
                      <div className="absolute top-1 right-2 text-[10px] text-muted-foreground/50 font-medium">
                        #{index + 1}
                      </div>
                    </div>

                    {/* Mini Player - Always visible */}
                    <MiniSamplePlayer audioUrl={apiClient.getSampleUrl(sample.id)} />
                  </>
                )}
              </div>
            );
          })}
        </div>
      )}

      <Button
        type="button"
        variant="outline"
        className="w-full"
        onClick={() => setUploadOpen(true)}
      >
        <HugeiconsIcon icon={Add01Icon} size={16} className="mr-2 h-4 w-4" />
        Add Sample
      </Button>

      <p className="text-xs text-muted-foreground text-center px-2">
        Note: A single 30-second sample is the sweet spot. Quality may decrease with multiple
        samples. In a future update samples might be interchangeable and tagged for varying styles
        of the same voice.
      </p>

      <SampleUpload profileId={profileId} open={uploadOpen} onOpenChange={setUploadOpen} />

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Sample</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this audio sample? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setDeleteDialogOpen(false);
                setSampleToDelete(null);
              }}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteConfirm}
              disabled={deleteSample.isPending}
            >
              {deleteSample.isPending ? 'Deleting...' : 'Delete'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
