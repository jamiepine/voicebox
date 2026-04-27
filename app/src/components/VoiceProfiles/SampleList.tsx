import { Check, Edit, Pause, Play, Plus, Trash2, Volume2, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
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
  const { t } = useTranslation();
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
          aria-label={isPlaying ? t('sampleList.player.pause') : t('sampleList.player.play')}
        >
          {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5 ms-0.5" />}
        </Button>

        <div className="flex-1 min-w-0 flex items-center gap-2">
          <Slider
            value={duration > 0 ? [(currentTime / duration) * 100] : [0]}
            onValueChange={handleSeek}
            max={100}
            step={0.1}
            className="flex-1"
            aria-label={t('sampleList.player.position')}
            aria-valuetext={t('sampleList.player.positionValue', {
              current: formatAudioDuration(currentTime),
              total: formatAudioDuration(duration),
            })}
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
          title={t('sampleList.player.stop')}
          aria-label={t('sampleList.player.stopAria')}
        >
          <X className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  );
}

interface SampleListProps {
  profileId: string;
}

export function SampleList({ profileId }: SampleListProps) {
  const { t } = useTranslation();
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
        title: t('sampleList.toast.invalidText'),
        description: t('sampleList.toast.invalidTextDescription'),
        variant: 'destructive',
      });
      return;
    }

    try {
      await updateSample.mutateAsync({ sampleId, referenceText: editedText.trim() });
      toast({
        title: t('sampleList.toast.updated'),
        description: t('sampleList.toast.updatedDescription'),
      });
      setEditingSampleId(null);
      setEditedText('');
    } catch (error) {
      toast({
        title: t('sampleList.toast.updateFailed'),
        description:
          error instanceof Error ? error.message : t('sampleList.toast.updateFailedFallback'),
        variant: 'destructive',
      });
    }
  };

  if (isLoading) {
    return <div className="text-sm text-muted-foreground">{t('sampleList.loading')}</div>;
  }

  return (
    <div className="space-y-4 pt-4">
      {samples && samples.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-8 text-center border border-dashed rounded-lg">
          <Volume2 className="h-8 w-8 text-muted-foreground/50 mb-2" />
          <p className="text-sm text-muted-foreground">{t('sampleList.empty.title')}</p>
          <p className="text-xs text-muted-foreground/70 mt-1">{t('sampleList.empty.hint')}</p>
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
                      <Edit className="h-3 w-3" />
                      <span>{t('sampleList.editing')}</span>
                    </div>
                    <Textarea
                      value={editedText}
                      onChange={(e) => setEditedText(e.target.value)}
                      className="min-h-[100px] text-sm resize-none"
                      placeholder={t('sampleList.placeholder')}
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
                        <X className="h-4 w-4 me-1" />
                        {t('common.cancel')}
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        onClick={() => handleSaveEdit(sample.id)}
                        disabled={updateSample.isPending}
                      >
                        <Check className="h-4 w-4 me-1" />
                        {updateSample.isPending ? t('sampleList.saving') : t('common.save')}
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
                          icon={Edit}
                          title={t('sampleList.editTranscription')}
                          onClick={() => handleStartEdit(sample.id, sample.reference_text)}
                        />
                        <CircleButton
                          icon={Trash2}
                          title={t('sampleList.deleteSample')}
                          onClick={() => handleDeleteClick(sample.id)}
                          disabled={deleteSample.isPending}
                        />
                      </div>

                      {/* Sample Number Badge */}
                      <div className="absolute top-1 end-2 text-[10px] text-muted-foreground/50 font-medium">
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
        <Plus className="me-2 h-4 w-4" />
        {t('sampleList.addSample')}
      </Button>

      <p className="text-xs text-muted-foreground text-center px-2">{t('sampleList.note')}</p>

      <SampleUpload profileId={profileId} open={uploadOpen} onOpenChange={setUploadOpen} />

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('sampleList.deleteDialog.title')}</DialogTitle>
            <DialogDescription>{t('sampleList.deleteDialog.description')}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setDeleteDialogOpen(false);
                setSampleToDelete(null);
              }}
            >
              {t('common.cancel')}
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteConfirm}
              disabled={deleteSample.isPending}
            >
              {deleteSample.isPending ? t('sampleList.deleteDialog.deleting') : t('common.delete')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
