import { FolderPlus, Info, LayoutGrid, List, Mic, Sparkles } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import type { VoiceProfileResponse } from '@/lib/api/types';
import {
  useCreateFolder,
  useDeleteFolder,
  useFolders,
  useUpdateFolder,
} from '@/lib/hooks/useFolders';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';
import { useUIStore } from '@/stores/uiStore';
import { FolderSection } from './FolderSection';
import { ProfileCard } from './ProfileCard';
import { ProfileForm } from './ProfileForm';
import { ProfileRow } from './ProfileRow';

/** Engines that use preset (built-in) voices instead of cloned profiles. */
const PRESET_ENGINES = new Set(['kokoro', 'qwen_custom_voice']);

/** Sentinel key for the Uncategorised bucket, which has no folder id. */
const UNCATEGORISED = '__uncategorised__';

export function ProfileList() {
  const { t } = useTranslation();
  const { data: profiles, isLoading, error } = useProfiles();
  const { data: folders } = useFolders('voice');

  const setDialogOpen = useUIStore((state) => state.setProfileDialogOpen);
  const selectedEngine = useUIStore((state) => state.selectedEngine);
  const selectedProfileId = useUIStore((state) => state.selectedProfileId);
  const viewMode = useUIStore((state) => state.voiceViewMode);
  const setViewMode = useUIStore((state) => state.setVoiceViewMode);
  const collapsedIds = useUIStore((state) => state.collapsedFolderIds.voice);
  const toggleCollapsed = useUIStore((state) => state.toggleFolderCollapsed);

  const createFolder = useCreateFolder('voice');
  const updateFolder = useUpdateFolder('voice');
  const deleteFolder = useDeleteFolder('voice');

  const [newFolderOpen, setNewFolderOpen] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');

  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  // Scroll to the selected profile after engine/sort changes
  // biome-ignore lint/correctness/useExhaustiveDependencies: selectedEngine reorders the list, so it must re-trigger the scroll even though the effect never reads it
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
  const voiceFolders = useMemo(() => folders || [], [folders]);
  const isPresetEngine = PRESET_ENGINES.has(selectedEngine);

  /** Whether a profile is supported by the currently selected engine. */
  const isSupported = useMemo(
    () => (p: VoiceProfileResponse) =>
      isPresetEngine
        ? p.voice_type === 'preset' && p.preset_engine === selectedEngine
        : p.voice_type !== 'preset',
    [isPresetEngine, selectedEngine],
  );

  // Sort so supported profiles come first, then bucket by folder. Sorting
  // before grouping keeps the supported-first ordering inside each folder.
  const grouped = useMemo(() => {
    const sorted = [...allProfiles].sort(
      (a, b) => (isSupported(a) ? 0 : 1) - (isSupported(b) ? 0 : 1),
    );

    const buckets = new Map<string, VoiceProfileResponse[]>();
    buckets.set(UNCATEGORISED, []);
    for (const folder of voiceFolders) buckets.set(folder.id, []);

    for (const profile of sorted) {
      // A folder_id can outlive its folder if another window deleted it
      // between renders — fall back rather than dropping the voice.
      const key =
        profile.folder_id && buckets.has(profile.folder_id) ? profile.folder_id : UNCATEGORISED;
      buckets.get(key)?.push(profile);
    }
    return buckets;
  }, [allProfiles, voiceFolders, isSupported]);

  const hasUnsupported = allProfiles.some((p) => !isSupported(p));

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

  const renderProfiles = (items: VoiceProfileResponse[]) => {
    if (items.length === 0) {
      return (
        <p className="px-1 py-2 text-xs text-muted-foreground/70">{t('folders.emptyFolder')}</p>
      );
    }

    return (
      <div
        className={cn(
          viewMode === 'card'
            ? 'flex gap-4 overflow-x-auto p-1 pb-1 lg:grid lg:grid-cols-3 lg:auto-rows-auto lg:overflow-x-visible'
            : 'flex flex-col gap-1',
        )}
      >
        {items.map((profile) => (
          <div
            key={profile.id}
            className={viewMode === 'card' ? 'shrink-0 w-[200px] lg:w-auto lg:shrink' : undefined}
            ref={(el) => {
              if (el) cardRefs.current.set(profile.id, el);
              else cardRefs.current.delete(profile.id);
            }}
          >
            {viewMode === 'card' ? (
              <ProfileCard profile={profile} disabled={!isSupported(profile)} />
            ) : (
              <ProfileRow
                profile={profile}
                disabled={!isSupported(profile)}
                folders={voiceFolders}
              />
            )}
          </div>
        ))}
      </div>
    );
  };

  const submitNewFolder = () => {
    const trimmed = newFolderName.trim();
    if (!trimmed) return;
    createFolder.mutate({ name: trimmed });
    setNewFolderName('');
    setNewFolderOpen(false);
  };

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
        ) : (
          <div className="flex flex-col gap-2 pb-[150px]">
            <div className="flex items-center justify-end gap-1">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 gap-1.5 px-2 text-xs"
                onClick={() => setNewFolderOpen(true)}
              >
                <FolderPlus className="h-3.5 w-3.5" />
                {t('folders.new')}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                aria-label={t(
                  viewMode === 'list' ? 'profiles.list.showCards' : 'profiles.list.showList',
                )}
                aria-pressed={viewMode === 'card'}
                onClick={() => setViewMode(viewMode === 'list' ? 'card' : 'list')}
              >
                {viewMode === 'list' ? (
                  <LayoutGrid className="h-4 w-4" />
                ) : (
                  <List className="h-4 w-4" />
                )}
              </Button>
            </div>

            {voiceFolders.map((folder) => (
              <FolderSection
                key={folder.id}
                folderId={folder.id}
                name={folder.name}
                count={grouped.get(folder.id)?.length ?? 0}
                collapsed={collapsedIds.includes(folder.id)}
                onToggle={() => toggleCollapsed('voice', folder.id)}
                onRename={(name) => updateFolder.mutate({ folderId: folder.id, data: { name } })}
                onDelete={() => deleteFolder.mutate(folder.id)}
              >
                {renderProfiles(grouped.get(folder.id) ?? [])}
              </FolderSection>
            ))}

            {/* Only worth a header once folders exist to contrast it with. */}
            {voiceFolders.length > 0 ? (
              <FolderSection
                folderId={null}
                name={t('folders.uncategorised')}
                count={grouped.get(UNCATEGORISED)?.length ?? 0}
                collapsed={collapsedIds.includes(UNCATEGORISED)}
                onToggle={() => toggleCollapsed('voice', UNCATEGORISED)}
              >
                {renderProfiles(grouped.get(UNCATEGORISED) ?? [])}
              </FolderSection>
            ) : (
              renderProfiles(grouped.get(UNCATEGORISED) ?? [])
            )}

            {hasUnsupported && (
              <div className="flex items-center gap-2 py-2 text-xs text-muted-foreground">
                <Info className="h-3.5 w-3.5 shrink-0" />
                <span>{t('profiles.list.unsupportedNote')}</span>
              </div>
            )}
          </div>
        )}
      </div>

      <Dialog open={newFolderOpen} onOpenChange={setNewFolderOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('folders.newDialog.title')}</DialogTitle>
          </DialogHeader>
          <Input
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            placeholder={t('folders.newDialog.placeholder')}
            onKeyDown={(e) => {
              if (e.key === 'Enter') submitNewFolder();
            }}
            aria-label={t('folders.newDialog.title')}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setNewFolderOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button onClick={submitNewFolder} disabled={!newFolderName.trim()}>
              {t('common.create')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ProfileForm />
    </div>
  );
}
