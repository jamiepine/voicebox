import { Info, Mic, Sparkles } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { useUIStore } from '@/stores/uiStore';
import { ProfileCard } from './ProfileCard';
import { ProfileForm } from './ProfileForm';

/** Engines that use preset (built-in) voices instead of cloned profiles. */
const PRESET_ENGINES = new Set(['kokoro', 'qwen_custom_voice']);

export function ProfileList() {
  const { data: profiles, isLoading, error } = useProfiles();
  const setDialogOpen = useUIStore((state) => state.setProfileDialogOpen);
  const selectedEngine = useUIStore((state) => state.selectedEngine);
  const selectedProfileId = useUIStore((state) => state.selectedProfileId);
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  // Scroll to the selected profile after engine/sort changes
  useEffect(() => {
    if (!selectedProfileId) return;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    const rafId = requestAnimationFrame(() => {
      const el = cardRefs.current.get(selectedProfileId);
      if (!el) return;

      // Temporarily apply scroll-margin so it doesn't land flush at the top
      el.style.scrollMarginTop = '180px';
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
      timeoutId = setTimeout(() => {
        el.style.scrollMarginTop = '';
      }, 500);
    });
    return () => {
      cancelAnimationFrame(rafId);
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [selectedProfileId, selectedEngine]);

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

  /** Whether a profile is supported by the currently selected engine. */
  const isSupported = (p: (typeof allProfiles)[number]) =>
    isPresetEngine
      ? p.voice_type === 'preset' && p.preset_engine === selectedEngine
      : p.voice_type !== 'preset';

  // Sort so supported profiles come first
  const sortedProfiles = [...allProfiles].sort(
    (a, b) => (isSupported(a) ? 0 : 1) - (isSupported(b) ? 0 : 1),
  );

  const hasUnsupported = sortedProfiles.some((p) => !isSupported(p));

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
        ) : (
          <div className="flex gap-4 overflow-x-auto p-1 pb-1 lg:grid lg:grid-cols-3 lg:auto-rows-auto lg:overflow-x-visible lg:pb-[150px]">
            {sortedProfiles.map((profile) => (
              <div
                key={profile.id}
                className="shrink-0 w-[200px] lg:w-auto lg:shrink"
                ref={(el) => {
                  if (el) cardRefs.current.set(profile.id, el);
                  else cardRefs.current.delete(profile.id);
                }}
              >
                <ProfileCard profile={profile} disabled={!isSupported(profile)} />
              </div>
            ))}
            {hasUnsupported && (
              <div className="col-span-full flex items-center gap-2 text-xs text-muted-foreground py-2">
                <Info className="h-3.5 w-3.5 shrink-0" />
                <span>Only supported voice profiles can be selected for the current model.</span>
              </div>
            )}
          </div>
        )}
      </div>

      <ProfileForm />
    </div>
  );
}
