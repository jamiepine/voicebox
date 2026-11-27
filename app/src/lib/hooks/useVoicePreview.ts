import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useToast } from '@/components/ui/use-toast';

export type VoicePreviewState = 'idle' | 'loading' | 'playing';

/**
 * Plays short voice previews, one at a time.
 *
 * Clips are kept as object URLs for the life of the hook, so replaying a voice
 * (or flicking back to one) is instant. The server renders a clip on first use,
 * which can take a few seconds.
 */
export function useVoicePreview() {
  const { t } = useTranslation();
  const { toast } = useToast();
  const [playingKey, setPlayingKey] = useState<string | null>(null);
  const [loadingKey, setLoadingKey] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlsRef = useRef(new Map<string, string>());
  // The key the user most recently asked for. A slow response for an earlier
  // click compares against this and stays silent instead of talking over them.
  const activeKeyRef = useRef<string | null>(null);

  const halt = useCallback(() => {
    activeKeyRef.current = null;
    audioRef.current?.pause();
    audioRef.current = null;
    setPlayingKey(null);
    setLoadingKey(null);
  }, []);

  const toggle = useCallback(
    async (key: string, load: () => Promise<Blob>) => {
      // Clicking the voice that is loading or playing stops it.
      const wasActive = activeKeyRef.current === key;
      halt();
      if (wasActive) return;

      activeKeyRef.current = key;

      let url = urlsRef.current.get(key);
      if (!url) {
        setLoadingKey(key);
        try {
          url = URL.createObjectURL(await load());
          urlsRef.current.set(key, url);
        } catch (err) {
          if (activeKeyRef.current === key) halt();
          toast({
            title: t('generation.voicePicker.previewFailed'),
            description: err instanceof Error ? err.message : undefined,
            variant: 'destructive',
          });
          return;
        }
        if (activeKeyRef.current !== key) return;
        setLoadingKey(null);
      }

      const audio = new Audio(url);
      audioRef.current = audio;
      const finish = () => {
        if (audioRef.current === audio) halt();
      };
      audio.addEventListener('ended', finish);
      audio.addEventListener('error', finish);
      setPlayingKey(key);
      try {
        await audio.play();
      } catch {
        finish();
      }
    },
    [halt, t, toast],
  );

  const stateOf = useCallback(
    (key: string): VoicePreviewState =>
      loadingKey === key ? 'loading' : playingKey === key ? 'playing' : 'idle',
    [loadingKey, playingKey],
  );

  useEffect(() => {
    const urls = urlsRef.current;
    return () => {
      halt();
      for (const url of urls.values()) URL.revokeObjectURL(url);
      urls.clear();
    };
  }, [halt]);

  return { toggle, stateOf, stop: halt };
}
