import { Mic, Sparkles, Wand2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { apiClient } from '@/lib/api/client';
import type { VoiceProfileResponse } from '@/lib/api/types';
import { useProfileSamples } from '@/lib/hooks/useProfiles';
import { MiniSamplePlayer } from './SampleList';

interface VoicePreviewDialogProps {
  profile: VoiceProfileResponse | null;
  onOpenChange: (open: boolean) => void;
}

/**
 * Hear a voice without leaving the Generate tab.
 *
 * Uses the same MiniSamplePlayer as the Voices tab rather than a second
 * implementation, so playback behaves identically in both places. Read-only by
 * design: this is for deciding which voice to use, not for editing it — the
 * Voices tab still owns adding, retitling and deleting samples.
 */
export function VoicePreviewDialog({ profile, onOpenChange }: VoicePreviewDialogProps) {
  const { t } = useTranslation();
  const { data: samples, isLoading, isError } = useProfileSamples(profile?.id ?? '');

  return (
    <Dialog open={profile !== null} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {profile?.name}
            {profile?.language && (
              <Badge variant="outline" className="h-5 px-1.5 text-xs text-muted-foreground">
                {profile.language}
              </Badge>
            )}
            {profile?.effects_chain && profile.effects_chain.length > 0 && (
              <Sparkles className="h-3.5 w-3.5 fill-accent text-accent" />
            )}
            {profile?.personality?.trim() && <Wand2 className="h-3.5 w-3.5 text-accent" />}
          </DialogTitle>
          {profile?.description && <DialogDescription>{profile.description}</DialogDescription>}
        </DialogHeader>

        <div className="max-h-[50vh] space-y-3 overflow-y-auto">
          {isLoading ? (
            <p className="py-6 text-center text-sm text-muted-foreground">{t('common.loading')}</p>
          ) : isError ? (
            // A failed request must not read as "this voice has no audio" —
            // one is a problem to retry, the other is normal for presets.
            <p className="py-6 text-center text-sm text-destructive">
              {t('profiles.preview.loadFailed')}
            </p>
          ) : !samples || samples.length === 0 ? (
            // Preset and designed voices have no reference audio to play —
            // say so rather than showing an empty box.
            <div className="flex flex-col items-center gap-2 py-8 text-center">
              <Mic className="h-8 w-8 text-muted-foreground/50" />
              <p className="text-sm text-muted-foreground">{t('profiles.preview.noSamples')}</p>
            </div>
          ) : (
            samples.map((sample) => (
              <div key={sample.id} className="rounded-md border p-3">
                <p className="mb-2 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
                  {sample.reference_text}
                </p>
                <MiniSamplePlayer audioUrl={apiClient.getSampleUrl(sample.id)} />
              </div>
            ))
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
