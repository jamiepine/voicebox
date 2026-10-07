import {
  Copy,
  Download,
  Edit,
  FolderInput,
  MoreHorizontal,
  Sparkles,
  Trash2,
  Wand2,
} from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useToast } from '@/components/ui/use-toast';
import type { FolderResponse, VoiceProfileResponse } from '@/lib/api/types';
import { useSetProfileFolder } from '@/lib/hooks/useFolders';
import { useDeleteProfile, useDuplicateProfile, useExportProfile } from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';
import { useUIStore } from '@/stores/uiStore';

/** Human-readable display names for preset engine badges. */
const ENGINE_DISPLAY_NAMES: Record<string, string> = {
  kokoro: 'Kokoro',
  qwen_custom_voice: 'CustomVoice',
};

interface ProfileRowProps {
  profile: VoiceProfileResponse;
  /** Not usable by the selected engine — dimmed but still selectable. */
  disabled?: boolean;
  /** Voice folders, for the "Move to" submenu. */
  folders: FolderResponse[];
}

/**
 * One voice as a two-line row: name, language and trait icons on the first
 * line, description on the second.
 *
 * Row actions live behind a menu rather than always-visible buttons —
 * at list density a row is ~48px tall, too tight for four icon buttons
 * without crowding the text.
 */
export function ProfileRow({ profile, disabled, folders }: ProfileRowProps) {
  const { t } = useTranslation();
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const { toast } = useToast();

  const deleteProfile = useDeleteProfile();
  const exportProfile = useExportProfile();
  const duplicateProfile = useDuplicateProfile();
  const setProfileFolder = useSetProfileFolder();

  const setEditingProfileId = useUIStore((state) => state.setEditingProfileId);
  const setProfileDialogOpen = useUIStore((state) => state.setProfileDialogOpen);
  const selectedProfileId = useUIStore((state) => state.selectedProfileId);
  const setSelectedProfileId = useUIStore((state) => state.setSelectedProfileId);

  const isSelected = selectedProfileId === profile.id;

  const handleSelect = () => {
    // Re-selecting a disabled voice re-fires the selection so the generate
    // form can surface its unsupported-engine hint again.
    if (disabled && isSelected) {
      setSelectedProfileId(null);
      setTimeout(() => setSelectedProfileId(profile.id), 0);
      return;
    }
    setSelectedProfileId(isSelected ? null : profile.id);
  };

  const handleDuplicate = () => {
    duplicateProfile.mutate(
      { profileId: profile.id },
      {
        onSuccess: (copy) => {
          toast({
            title: t('profiles.duplicate.successTitle'),
            description: t('profiles.duplicate.successDescription', { name: copy.name }),
          });
        },
        onError: (error) => {
          toast({
            title: t('profiles.duplicate.failedTitle'),
            description: error.message,
            variant: 'destructive',
          });
        },
      },
    );
  };

  const selectLabel = t(
    isSelected ? 'profiles.card.selectLabelSelected' : 'profiles.card.selectLabel',
    { name: profile.name, language: profile.language },
  );

  return (
    <>
      {/* The row is a plain container rather than role="button": the actions
          menu is itself a button, and nesting interactive elements is invalid.
          The selectable area is a real <button> instead. */}
      <div
        className={cn(
          'group flex items-start gap-3 rounded-md border px-3 py-2 transition-colors',
          disabled ? 'opacity-40 hover:opacity-60' : 'hover:bg-accent/40',
          isSelected && !disabled && 'ring-2 ring-accent border-transparent bg-accent/30',
        )}
      >
        <button
          type="button"
          className="min-w-0 flex-1 cursor-pointer text-left"
          onClick={handleSelect}
          aria-label={selectLabel}
          aria-pressed={isSelected}
        >
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium">{profile.name}</span>
            <Badge variant="outline" className="h-5 shrink-0 px-1.5 text-xs text-muted-foreground">
              {profile.language}
            </Badge>
            {profile.voice_type === 'preset' && (
              <Badge variant="secondary" className="h-5 shrink-0 px-1.5 text-xs">
                {ENGINE_DISPLAY_NAMES[profile.preset_engine ?? ''] ?? profile.preset_engine}
              </Badge>
            )}
            {profile.voice_type === 'designed' && (
              <Badge variant="secondary" className="h-5 shrink-0 px-1.5 text-xs">
                {t('profiles.card.designed')}
              </Badge>
            )}
            {profile.effects_chain && profile.effects_chain.length > 0 && (
              <Sparkles
                className="h-3.5 w-3.5 shrink-0 fill-accent text-accent"
                aria-label={t('profiles.row.hasEffects')}
              />
            )}
            {profile.personality?.trim() && (
              <Wand2
                className="h-3.5 w-3.5 shrink-0 text-accent"
                aria-label={t('profiles.row.hasPersonality')}
              />
            )}
          </div>
          <p className="truncate text-xs leading-relaxed text-muted-foreground">
            {profile.description || t('profiles.card.noDescription')}
          </p>
        </button>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 shrink-0 opacity-0 transition-opacity focus-visible:opacity-100 group-hover:opacity-100 data-[state=open]:opacity-100"
              onClick={(e) => e.stopPropagation()}
              aria-label={t('profiles.row.actions', { name: profile.name })}
            >
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
            <DropdownMenuItem
              onClick={() => {
                setEditingProfileId(profile.id);
                setProfileDialogOpen(true);
              }}
            >
              <Edit className="mr-2 h-4 w-4" />
              {t('profiles.card.edit')}
            </DropdownMenuItem>
            <DropdownMenuItem onClick={handleDuplicate} disabled={duplicateProfile.isPending}>
              <Copy className="mr-2 h-4 w-4" />
              {t('profiles.card.duplicate')}
            </DropdownMenuItem>

            <DropdownMenuSub>
              <DropdownMenuSubTrigger>
                <FolderInput className="mr-2 h-4 w-4" />
                {t('profiles.row.moveTo')}
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                <DropdownMenuLabel>{t('folders.voice.label')}</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  disabled={!profile.folder_id}
                  onClick={() => setProfileFolder.mutate({ profileId: profile.id, folderId: null })}
                >
                  {t('folders.uncategorised')}
                </DropdownMenuItem>
                {folders.map((folder) => (
                  <DropdownMenuItem
                    key={folder.id}
                    disabled={folder.id === profile.folder_id}
                    onClick={() =>
                      setProfileFolder.mutate({ profileId: profile.id, folderId: folder.id })
                    }
                  >
                    {folder.name}
                  </DropdownMenuItem>
                ))}
                {folders.length === 0 && (
                  <DropdownMenuItem disabled>{t('folders.none')}</DropdownMenuItem>
                )}
              </DropdownMenuSubContent>
            </DropdownMenuSub>

            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => exportProfile.mutate(profile.id)}
              disabled={exportProfile.isPending}
            >
              <Download className="mr-2 h-4 w-4" />
              {t('profiles.card.export')}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-destructive focus:text-destructive"
              onClick={() => setDeleteDialogOpen(true)}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              {t('profiles.card.delete')}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('profiles.deleteDialog.title')}</DialogTitle>
            <DialogDescription>
              {t('profiles.deleteDialog.body', { name: profile.name })}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                deleteProfile.mutate(profile.id);
                setDeleteDialogOpen(false);
              }}
              disabled={deleteProfile.isPending}
            >
              {deleteProfile.isPending ? t('profiles.deleteDialog.deleting') : t('common.delete')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
