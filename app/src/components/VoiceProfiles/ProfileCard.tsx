import { Download, Edit, Sparkles, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { CircleButton } from '@/components/ui/circle-button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import type { VoiceProfileResponse } from '@/lib/api/types';
import { useDeleteProfile, useExportProfile } from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';
import { useUIStore } from '@/stores/uiStore';

/** Human-readable display names for preset engine badges. */
const ENGINE_DISPLAY_NAMES: Record<string, string> = {
  kokoro: 'Kokoro',
  qwen_custom_voice: 'CustomVoice',
};

interface ProfileCardProps {
  profile: VoiceProfileResponse;
  disabled?: boolean;
}

export function ProfileCard({ profile, disabled }: ProfileCardProps) {
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

  const deleteProfile = useDeleteProfile();
  const exportProfile = useExportProfile();
  const setEditingProfileId = useUIStore((state) => state.setEditingProfileId);
  const setProfileDialogOpen = useUIStore((state) => state.setProfileDialogOpen);
  const selectedProfileId = useUIStore((state) => state.selectedProfileId);
  const setSelectedProfileId = useUIStore((state) => state.setSelectedProfileId);

  const isSelected = selectedProfileId === profile.id;

  const handleSelect = () => {
    // If disabled but already selected, bounce the selection to re-trigger engine auto-switch
    if (disabled && isSelected) {
      setSelectedProfileId(null);
      setTimeout(() => setSelectedProfileId(profile.id), 0);
      return;
    }
    setSelectedProfileId(isSelected ? null : profile.id);
  };

  const handleEdit = () => {
    setEditingProfileId(profile.id);
    setProfileDialogOpen(true);
  };

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    deleteProfile.mutate(profile.id);
    setDeleteDialogOpen(false);
  };

  const handleExport = (e: React.MouseEvent) => {
    e.stopPropagation();
    exportProfile.mutate(profile.id);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    const target = e.target as HTMLElement;
    if (target.closest('button')) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleSelect();
    }
  };

  const selectLabel = isSelected
    ? `${profile.name}, ${profile.language}. Selected as voice for generation.`
    : `${profile.name}, ${profile.language}. Select as voice for generation.`;

  return (
    <>
      <Card
        className={cn(
          'cursor-pointer transition-all flex flex-col h-[162px]',
          disabled ? 'opacity-40 hover:opacity-60' : 'hover:shadow-md',
          isSelected && !disabled && 'ring-2 ring-accent shadow-md',
        )}
        onClick={handleSelect}
        tabIndex={0}
        role="button"
        aria-label={selectLabel}
        aria-pressed={isSelected}
        onKeyDown={handleKeyDown}
      >
        <CardHeader className="p-3 pb-2">
          <CardTitle className="text-base font-medium">
            <span className="break-words">{profile.name}</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-3 pt-0 flex flex-col flex-1">
          <p className="text-xs text-muted-foreground mb-1.5 line-clamp-2 leading-relaxed">
            {profile.description || 'No description'}
          </p>
          <div className="mb-2 flex items-center gap-1.5">
            <Badge variant="outline" className="text-xs h-5 px-1.5 text-muted-foreground">
              {profile.language}
            </Badge>
            {profile.voice_type === 'preset' && (
              <Badge variant="secondary" className="text-xs h-5 px-1.5">
                {ENGINE_DISPLAY_NAMES[profile.preset_engine ?? ''] ?? profile.preset_engine}
              </Badge>
            )}
            {profile.voice_type === 'designed' && (
              <Badge variant="secondary" className="text-xs h-5 px-1.5">
                designed
              </Badge>
            )}
            {profile.effects_chain && profile.effects_chain.length > 0 && (
              <Sparkles className="h-3.5 w-3.5 text-accent fill-accent" />
            )}
          </div>
          <div className="flex gap-0.5 justify-end items-end mt-auto">
            <CircleButton
              icon={Download}
              onClick={handleExport}
              disabled={exportProfile.isPending}
              aria-label="Export profile"
            />
            <CircleButton
              icon={Edit}
              onClick={(e) => {
                e.stopPropagation();
                handleEdit();
              }}
              aria-label="Edit profile"
            />
            <CircleButton
              icon={Trash2}
              onClick={handleDeleteClick}
              disabled={deleteProfile.isPending}
              aria-label="Delete profile"
            />
          </div>
        </CardContent>
      </Card>

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Profile</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete "{profile.name}"? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteConfirm}
              disabled={deleteProfile.isPending}
            >
              {deleteProfile.isPending ? 'Deleting...' : 'Delete'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
