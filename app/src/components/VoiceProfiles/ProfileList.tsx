import { Info, Mic, Search, Sparkles } from 'lucide-react';
import { useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { useUIStore } from '@/stores/uiStore';
import { ProfileCard } from './ProfileCard';
import { ProfileForm } from './ProfileForm';

/** Engines that use preset (built-in) voices instead of cloned profiles. */
const PRESET_ENGINES = new Set(['kokoro', 'qwen_custom_voice']);

interface ProfileListProps {
  search?: string;
  onClearSearch?: () => void;
}

export function ProfileList({ search = '', onClearSearch }: ProfileListProps) {
  const { t } = useTranslation();
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

  const allProfiles = useMemo(() => profiles || [], [profiles]);
  const isPresetEngine = PRESET_ENGINES.has(selectedEngine);

  /** Whether a profile is supported by the currently selected engine. */
  const isSupported = (p: (typeof allProfiles)[number]) =>
    isPresetEngine
      ? p.voice_type === 'preset' && p.preset_engine === selectedEngine
      : p.voice_type !== 'preset';

  // Sort so supported profiles come first, selected profile at top, then alphabetical by name
  const sortedProfiles = useMemo(() => {
    return [...allProfiles].sort((a, b) => {
      const suppA = isSupported(a) ? 0 : 1;
      const suppB = isSupported(b) ? 0 : 1;
      if (suppA !== suppB) return suppA - suppB;

      const selA = selectedProfileId === a.id ? 0 : 1;
      const selB = selectedProfileId === b.id ? 0 : 1;
      if (selA !== selB) return selA - selB;

      return a.name.localeCompare(b.name);
    });
  }, [allProfiles, selectedEngine, selectedProfileId]);

  const filteredProfiles = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return sortedProfiles;
    return sortedProfiles.filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        p.description?.toLowerCase().includes(q) ||
        p.language.toLowerCase().includes(q) ||
        p.preset_engine?.toLowerCase().includes(q) ||
        p.default_engine?.toLowerCase().includes(q),
    );
  }, [sortedProfiles, search]);

  if (isLoading) {
    return null;
  }

  if (error) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-destructive">
          {t('profiles.list.errorLoading', { message: error.message })}
        </div>
      </div>
    );
  }

  const hasUnsupported = filteredProfiles.some((p) => !isSupported(p));

  return (
    <div className="flex flex-col">
      <div className="shrink-0">
        {allProfiles.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Mic className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground mb-4">{t('profiles.list.empty')}</p>
              <Button onClick={() => setDialogOpen(true)}>
                <Sparkles className="mr-2 h-4 w-4" />
                {t('profiles.list.createVoice')}
              </Button>
            </CardContent>
          </Card>
        ) : filteredProfiles.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12 text-center">
              <Search className="h-10 w-10 text-muted-foreground mb-3 opacity-50" />
              <p className="font-medium text-sm mb-1">{t('main.noVoicesFound')}</p>
              <p className="text-xs text-muted-foreground mb-4">
                {t('profiles.list.noVoicesMatch', { query: search })}
              </p>
              {onClearSearch && (
                <Button variant="outline" size="sm" onClick={onClearSearch}>
                  {t('main.clearSearch')}
                </Button>
              )}
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 p-1 pb-12 lg:pb-24">
            {filteredProfiles.map((profile) => (
              <div
                key={profile.id}
                className="w-full min-w-0"
                ref={(el) => {
                  if (el) cardRefs.current.set(profile.id, el);
                  else cardRefs.current.delete(profile.id);
                }}
              >
                <ProfileCard profile={profile} disabled={!isSupported(profile)} />
              </div>
            ))}
            {hasUnsupported && (
              <div className="col-span-full flex items-center gap-2 text-xs text-muted-foreground pt-3 pb-8 border-t border-border/20 mt-1">
                <Info className="h-4 w-4 text-accent shrink-0" />
                <span className="leading-normal">{t('profiles.list.unsupportedNote')}</span>
              </div>
            )}
          </div>
        )}
      </div>

      <ProfileForm />
    </div>
  );
}

