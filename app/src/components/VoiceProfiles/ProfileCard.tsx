import { Edit, Eye, Mic, Trash2 } from 'lucide-react';
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
import { useDeleteProfile } from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';
import { useUIStore } from '@/stores/uiStore';
import { ProfileDetail } from './ProfileDetail';

interface ProfileCardProps {
  profile: VoiceProfileResponse;
}

export function ProfileCard({ profile }: ProfileCardProps) {
  const [detailOpen, setDetailOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const deleteProfile = useDeleteProfile();
  const setEditingProfileId = useUIStore((state) => state.setEditingProfileId);
  const setProfileDialogOpen = useUIStore((state) => state.setProfileDialogOpen);
  const selectedProfileId = useUIStore((state) => state.selectedProfileId);
  const setSelectedProfileId = useUIStore((state) => state.setSelectedProfileId);

  const isSelected = selectedProfileId === profile.id;

  const handleSelect = () => {
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

  return (
    <>
      <Card
        className={cn(
          'cursor-pointer hover:shadow-md transition-all flex flex-col',
          isSelected && 'ring-2 ring-primary shadow-md',
        )}
        onClick={handleSelect}
      >
        <CardHeader className="p-3 pb-2">
          <CardTitle className="flex items-center gap-1.5 text-base font-medium">
            <div className="h-6 w-6 rounded-full bg-muted flex items-center justify-center shrink-0">
              <Mic className="h-3.5 w-3.5 text-muted-foreground" />
            </div>
            <span className="break-words">{profile.name}</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-3 pt-0 flex flex-col flex-1">
          <p className="text-xs text-muted-foreground mb-1.5 line-clamp-2 leading-relaxed">
            {profile.description || 'No description'}
          </p>
          <div className="mb-2">
            <Badge variant="outline" className="text-xs h-5 px-1.5 text-muted-foreground">
              {profile.language}
            </Badge>
          </div>
          <div className="flex gap-0.5 justify-end items-end mt-auto">
            <CircleButton
              icon={Eye}
              onClick={(e) => {
                e.stopPropagation();
                setDetailOpen(true);
              }}
              aria-label="View details"
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

      <ProfileDetail profileId={profile.id} open={detailOpen} onOpenChange={setDetailOpen} />

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
