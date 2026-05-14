import { useRef, useState } from 'react';
import { useToast } from '@/components/ui/use-toast';

export function useAudioPlayer() {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // Track the File object that the current audio element was created from so
  // we can detect when the caller switches to a different file.
  const fileRef = useRef<File | null>(null);
  const { toast } = useToast();

  const _destroyCurrent = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      if (audioRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioRef.current.src);
      }
      audioRef.current = null;
    }
    fileRef.current = null;
  };

  const playPause = (file: File | null | undefined) => {
    if (!file) return;

    // If a different file is requested, tear down the old element first so
    // the object URL is released and we start fresh rather than replaying
    // the old audio.
    if (audioRef.current && fileRef.current !== file) {
      _destroyCurrent();
      setIsPlaying(false);
    }

    if (audioRef.current) {
      // Same file — toggle play/pause on the existing element.
      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        audioRef.current.play();
        setIsPlaying(true);
      }
    } else {
      const objectUrl = URL.createObjectURL(file);
      const audio = new Audio(objectUrl);
      audioRef.current = audio;
      fileRef.current = file;

      audio.addEventListener('ended', () => {
        setIsPlaying(false);
        URL.revokeObjectURL(objectUrl);
        audioRef.current = null;
        fileRef.current = null;
      });

      audio.addEventListener('error', () => {
        setIsPlaying(false);
        toast({
          title: 'Playback error',
          description: 'Failed to play audio file',
          variant: 'destructive',
        });
        URL.revokeObjectURL(objectUrl);
        audioRef.current = null;
        fileRef.current = null;
      });

      audio.play();
      setIsPlaying(true);
    }
  };

  const cleanup = () => {
    _destroyCurrent();
    setIsPlaying(false);
  };

  return {
    isPlaying,
    playPause,
    cleanup,
  };
}
