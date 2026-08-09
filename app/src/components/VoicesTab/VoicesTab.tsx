import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Mic, Plus, Search, Sparkles, Upload, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { MultiSelect } from '@/components/ui/multi-select';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useToast } from '@/components/ui/use-toast';
import { ProfileForm } from '@/components/VoiceProfiles/ProfileForm';
import { apiClient } from '@/lib/api/client';
import type { VoiceProfileResponse } from '@/lib/api/types';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { useImportProfile, useProfiles } from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';
import { usePlayerStore } from '@/stores/playerStore';
import { useServerStore } from '@/stores/serverStore';
import { useUIStore } from '@/stores/uiStore';
import { VoiceInspector } from './VoiceInspector';

export function VoicesTab() {
  const { t } = useTranslation();
  const { data: profiles, isLoading } = useProfiles();
  const queryClient = useQueryClient();
  const setDialogOpen = useUIStore((state) => state.setProfileDialogOpen);
  const selectedVoiceId = useUIStore((state) => state.selectedVoiceId);
  const setSelectedVoiceId = useUIStore((state) => state.setSelectedVoiceId);
  const scrollRef = useRef<HTMLDivElement>(null);
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;
  const importProfile = useImportProfile();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [languageFilter, setLanguageFilter] = useState<string>('all');
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Extract unique languages for filter dropdown
  const availableLanguages = useMemo(() => {
    if (!profiles) return [];
    const langs = new Set<string>();
    profiles.forEach((p) => {
      if (p.language) langs.add(p.language);
    });
    return Array.from(langs).sort();
  }, [profiles]);

  const filteredProfiles = useMemo(() => {
    if (!profiles) return [];
    return profiles.filter((p) => {
      // Type filter
      if (typeFilter !== 'all' && p.voice_type !== typeFilter) {
        return false;
      }
      // Language filter
      if (languageFilter !== 'all' && p.language !== languageFilter) {
        return false;
      }
      // Search filter
      if (search.trim()) {
        const q = search.trim().toLowerCase();
        const matchesName = p.name.toLowerCase().includes(q);
        const matchesDesc = p.description?.toLowerCase().includes(q) ?? false;
        const matchesLang = p.language.toLowerCase().includes(q);
        const matchesEngine =
          (p.preset_engine?.toLowerCase().includes(q) ?? false) ||
          (p.default_engine?.toLowerCase().includes(q) ?? false);

        if (!matchesName && !matchesDesc && !matchesLang && !matchesEngine) {
          return false;
        }
      }
      return true;
    });
  }, [profiles, search, typeFilter, languageFilter]);

  const handleClearFilters = () => {
    setSearch('');
    setTypeFilter('all');
    setLanguageFilter('all');
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.voicebox.zip')) {
        toast({
          title: t('main.import.invalidTitle'),
          description: t('main.import.invalidDescription'),
          variant: 'destructive',
        });
        return;
      }
      setSelectedFile(file);
      setImportDialogOpen(true);
    }
  };

  const handleImportConfirm = () => {
    if (selectedFile) {
      importProfile.mutate(selectedFile, {
        onSuccess: () => {
          setImportDialogOpen(false);
          setSelectedFile(null);
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
          toast({
            title: t('main.import.successTitle'),
            description: t('main.import.successDescription'),
          });
        },
        onError: (error) => {
          toast({
            title: t('main.import.failedTitle'),
            description: error.message,
            variant: 'destructive',
          });
        },
      });
    }
  };

  // Auto-select first profile if none selected
  useEffect(() => {
    if (!selectedVoiceId && profiles && profiles.length > 0) {
      setSelectedVoiceId(profiles[0].id);
    }
    // Clear selection if selected profile was deleted
    if (selectedVoiceId && profiles && !profiles.find((p) => p.id === selectedVoiceId)) {
      setSelectedVoiceId(profiles.length > 0 ? profiles[0].id : null);
    }
  }, [profiles, selectedVoiceId, setSelectedVoiceId]);

  // Get channel assignments for each profile
  const { data: channelAssignments } = useQuery({
    queryKey: ['profile-channels'],
    queryFn: async () => {
      if (!profiles) return {};
      const assignments: Record<string, string[]> = {};
      for (const profile of profiles) {
        try {
          const result = await apiClient.getProfileChannels(profile.id);
          assignments[profile.id] = result.channel_ids;
        } catch {
          assignments[profile.id] = [];
        }
      }
      return assignments;
    },
    enabled: !!profiles,
  });

  // Get all channels
  const { data: channels } = useQuery({
    queryKey: ['channels'],
    queryFn: () => apiClient.listChannels(),
  });

  const handleChannelChange = async (profileId: string, channelIds: string[]) => {
    try {
      await apiClient.setProfileChannels(profileId, channelIds);
      queryClient.invalidateQueries({ queryKey: ['profile-channels'] });
    } catch (error) {
      console.error('Failed to update channels:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-muted-foreground">{t('voicesTab.loading')}</div>
      </div>
    );
  }

  const isFiltered = search || typeFilter !== 'all' || languageFilter !== 'all';

  return (
    <div className="h-full flex gap-0 overflow-hidden -mx-8">
      {/* Left: Table */}
      <div className="flex-1 min-w-0 flex flex-col relative overflow-hidden">
        {/* Scroll Mask */}
        <div className="absolute top-0 left-0 right-0 h-28 bg-gradient-to-b from-background to-transparent z-10 pointer-events-none" />

        {/* Fixed Header */}
        <div className="absolute top-0 left-0 right-0 z-20 pl-8 pr-8 pb-3 bg-background/80 backdrop-blur-sm">
          {/* Row 1: Title & Actions */}
          <div className="flex items-center justify-between mb-3">
            <h1 className="text-2xl font-bold">{t('voicesTab.title')}</h1>
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleImportClick}>
                <Upload className="h-4 w-4 mr-2" />
                {t('main.importVoice')}
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".voicebox.zip"
                onChange={handleFileChange}
                className="hidden"
              />
              <Button onClick={() => setDialogOpen(true)}>
                <Plus className="h-4 w-4 mr-2" />
                {t('voicesTab.newVoice')}
              </Button>
            </div>
          </div>

          {/* Row 2: Search & Filter Toolbar */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative flex-1 min-w-[200px]">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground z-10 pointer-events-none" />
              <Input
                placeholder={t('voicesTab.searchPlaceholder')}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="h-9 pl-9 pr-8 text-sm rounded-full focus-visible:ring-0 focus-visible:ring-offset-0"
              />
              {search && (
                <button
                  type="button"
                  onClick={() => setSearch('')}
                  aria-label={t('main.clearSearch')}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-0.5 rounded-full text-muted-foreground hover:text-foreground hover:bg-muted transition-colors z-10"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>

            {/* Type Filter */}
            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger className="h-9 w-[130px] rounded-full text-xs">
                <SelectValue placeholder={t('voicesTab.filterType')} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">{t('voicesTab.allTypes')}</SelectItem>
                <SelectItem value="preset">{t('voicesTab.typePreset')}</SelectItem>
                <SelectItem value="cloned">{t('voicesTab.typeCloned')}</SelectItem>
                <SelectItem value="designed">{t('voicesTab.typeDesigned')}</SelectItem>
              </SelectContent>
            </Select>

            {/* Language Filter */}
            {availableLanguages.length > 0 && (
              <Select value={languageFilter} onValueChange={setLanguageFilter}>
                <SelectTrigger className="h-9 w-[140px] rounded-full text-xs">
                  <SelectValue placeholder={t('voicesTab.filterLanguage')} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">{t('voicesTab.allLanguages')}</SelectItem>
                  {availableLanguages.map((lang) => (
                    <SelectItem key={lang} value={lang}>
                      {lang.toUpperCase()}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}

            {isFiltered && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleClearFilters}
                className="h-9 px-3 text-xs text-muted-foreground hover:text-foreground"
              >
                <X className="h-3.5 w-3.5 mr-1" />
                {t('voicesTab.clearFilters')}
              </Button>
            )}
          </div>
        </div>

        {/* Scrollable Content */}
        <div
          ref={scrollRef}
          className={cn(
            'flex-1 overflow-y-auto hover-scrollbar overflow-x-hidden pt-28 relative z-0',
            isPlayerVisible && BOTTOM_SAFE_AREA_PADDING,
          )}
        >
          {filteredProfiles.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
              <Search className="h-10 w-10 text-muted-foreground mb-3 opacity-50" />
              <h3 className="font-semibold text-base mb-1">{t('voicesTab.noVoicesFound')}</h3>
              <p className="text-xs text-muted-foreground mb-4 max-w-sm">
                {t('voicesTab.noVoicesFoundDesc')}
              </p>
              {isFiltered && (
                <Button variant="outline" size="sm" onClick={handleClearFilters}>
                  {t('voicesTab.clearFilters')}
                </Button>
              )}
            </div>
          ) : (
            <Table className="table-fixed [&_td:first-child]:pl-8 [&_th:first-child]:pl-8">
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[30%]">{t('voicesTab.columns.name')}</TableHead>
                  <TableHead className="w-[10%]">{t('voicesTab.columns.language')}</TableHead>
                  <TableHead className="w-[10%]">{t('voicesTab.columns.generations')}</TableHead>
                  <TableHead className="w-[8%]">{t('voicesTab.columns.samples')}</TableHead>
                  <TableHead className="w-[8%]">{t('voicesTab.columns.effects')}</TableHead>
                  <TableHead className="w-[24%]">{t('voicesTab.columns.channels')}</TableHead>
                  <TableHead className="w-6"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredProfiles.map((profile) => (
                  <VoiceRow
                    key={profile.id}
                    profile={profile}
                    isSelected={selectedVoiceId === profile.id}
                    onSelect={() => setSelectedVoiceId(profile.id)}
                    channelIds={channelAssignments?.[profile.id] || []}
                    channels={channels || []}
                    onChannelChange={(channelIds) => handleChannelChange(profile.id, channelIds)}
                  />
                ))}
              </TableBody>
            </Table>
          )}
        </div>
      </div>

      {/* Right: Inspector */}
      {selectedVoiceId && (
        <div className="w-[340px] shrink-0 border-l border-t rounded-tl-xl bg-muted/30">
          <VoiceInspector key={selectedVoiceId} profileId={selectedVoiceId} />
        </div>
      )}

      <ProfileForm />

      <Dialog
        open={importDialogOpen}
        onOpenChange={(open) => {
          setImportDialogOpen(open);
          if (!open) {
            setSelectedFile(null);
            if (fileInputRef.current) {
              fileInputRef.current.value = '';
            }
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('main.import.dialogTitle')}</DialogTitle>
            <DialogDescription>
              {t('main.import.dialogDescription', { name: selectedFile?.name })}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setImportDialogOpen(false);
                setSelectedFile(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
            >
              {t('common.cancel')}
            </Button>
            <Button
              onClick={handleImportConfirm}
              disabled={importProfile.isPending || !selectedFile}
            >
              {importProfile.isPending ? t('main.import.importing') : t('main.import.action')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

interface VoiceRowProps {
  profile: VoiceProfileResponse;
  isSelected: boolean;
  onSelect: () => void;
  channelIds: string[];
  channels: Array<{ id: string; name: string; is_default: boolean }>;
  onChannelChange: (channelIds: string[]) => void;
}

function VoiceRow({
  profile,
  isSelected,
  onSelect,
  channelIds,
  channels,
  onChannelChange,
}: VoiceRowProps) {
  const { t } = useTranslation();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const [avatarError, setAvatarError] = useState(false);
  const avatarUrl = profile.avatar_path ? `${serverUrl}/profiles/${profile.id}/avatar` : null;

  const enabledEffects = profile.effects_chain?.filter((e) => e.enabled) ?? [];
  const effectsSummary = enabledEffects.map((e) => e.type).join(' → ');

  return (
    <TableRow
      className={cn('cursor-pointer', isSelected ? 'bg-muted/50' : 'hover:bg-muted/50')}
      onClick={onSelect}
    >
      <TableCell>
        <div className="flex w-full min-w-0 items-center gap-2">
          <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center shrink-0 overflow-hidden">
            {avatarUrl && !avatarError ? (
              <img
                src={avatarUrl}
                alt={t('voicesTab.avatarAlt', { name: profile.name })}
                className="h-full w-full object-cover"
                onError={() => setAvatarError(true)}
              />
            ) : (
              <Mic className="h-4 w-4 text-muted-foreground" />
            )}
          </div>
          <div className="min-w-0">
            <div className="font-medium truncate">{profile.name}</div>
            {profile.description && (
              <div className="text-sm text-muted-foreground truncate">{profile.description}</div>
            )}
          </div>
        </div>
      </TableCell>
      <TableCell>{profile.language}</TableCell>
      <TableCell>{profile.generation_count}</TableCell>
      <TableCell>{profile.sample_count}</TableCell>
      <TableCell>
        {enabledEffects.length > 0 ? (
          <span
            className="inline-flex items-center gap-1 text-xs text-accent"
            title={effectsSummary}
          >
            <Sparkles className="h-3 w-3 fill-accent" />
            {enabledEffects.length}
          </span>
        ) : (
          <span className="text-xs text-muted-foreground">—</span>
        )}
      </TableCell>
      <TableCell onClick={(e) => e.stopPropagation()}>
        <MultiSelect
          options={channels.map((ch) => ({
            value: ch.id,
            label: ch.is_default ? t('voicesTab.channelDefaultLabel', { name: ch.name }) : ch.name,
          }))}
          value={channelIds}
          onChange={onChannelChange}
          placeholder={t('voicesTab.selectChannels')}
          className="w-full"
        />
      </TableCell>
      <TableCell />
    </TableRow>
  );
}
