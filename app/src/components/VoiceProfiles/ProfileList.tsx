import { Mic, Music, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { useUIStore } from '@/stores/uiStore';
import { ProfileCard } from './ProfileCard';
import { ProfileForm } from './ProfileForm';

/** Engines that use preset (built-in) voices instead of cloned profiles. */
const PRESET_ENGINES = new Set(['kokoro', 'qwen_custom_voice']);

/** Human-readable engine names for empty state messages. */
const ENGINE_NAMES: Record<string, string> = {
  kokoro: 'Kokoro',
  qwen_custom_voice: 'Qwen CustomVoice',
};

export function ProfileList() {
  const { data: profiles, isLoading, error } = useProfiles();
  const setDialogOpen = useUIStore((state) => state.setProfileDialogOpen);
  const selectedEngine = useUIStore((state) => state.selectedEngine);

  if (isLoading) {
    return null;
  }

  if (error) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-destructive">Error loading profiles: {error.message}</div>
      </div>
    );
  }

  const allProfiles = profiles || [];
  const isPresetEngine = PRESET_ENGINES.has(selectedEngine);

  // Filter profiles based on selected engine
  const filteredProfiles = isPresetEngine
    ? allProfiles.filter((p) => p.voice_type === 'preset' && p.preset_engine === selectedEngine)
    : allProfiles.filter((p) => p.voice_type !== 'preset');

  return (
    <div className="flex flex-col">
      <div className="shrink-0">
        {allProfiles.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Mic className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground mb-4">
                No voice profiles yet. Create your first profile to get started.
              </p>
              <Button onClick={() => setDialogOpen(true)}>
                <Sparkles className="mr-2 h-4 w-4" />
                Create Voice
              </Button>
            </CardContent>
          </Card>
        ) : filteredProfiles.length === 0 && isPresetEngine ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Music className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground mb-2">
                No {ENGINE_NAMES[selectedEngine] ?? selectedEngine} voices created yet.
              </p>
              <p className="text-sm text-muted-foreground mb-4">
                Create a profile to choose a specific voice before generating.
              </p>
              <Button onClick={() => setDialogOpen(true)}>
                <Sparkles className="mr-2 h-4 w-4" />
                Create {ENGINE_NAMES[selectedEngine] ?? selectedEngine} Voice
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="flex gap-4 overflow-x-auto p-1 pb-1 lg:grid lg:grid-cols-3 lg:auto-rows-auto lg:overflow-x-visible lg:pb-[150px]">
            {filteredProfiles.map((profile) => (
              <div key={profile.id} className="shrink-0 w-[200px] lg:w-auto lg:shrink">
                <ProfileCard profile={profile} />
              </div>
            ))}
          </div>
        )}
      </div>

      <ProfileForm />
    </div>
  );
}
