import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Edit, MoreHorizontal, Plus, Trash2, Mic } from 'lucide-react';
import { useMemo, useRef } from 'react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { MultiSelect } from '@/components/ui/multi-select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { ProfileForm } from '@/components/VoiceProfiles/ProfileForm';
import { apiClient } from '@/lib/api/client';
import type { VoiceProfileResponse } from '@/lib/api/types';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { useHistory } from '@/lib/hooks/useHistory';
import { useDeleteProfile, useProfileSamples, useProfiles } from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';
import { usePlayerStore } from '@/stores/playerStore';
import { useUIStore } from '@/stores/uiStore';

export function VoicesTab() {
  const { data: profiles, isLoading } = useProfiles();
  const { data: historyData } = useHistory({ limit: 100 });
  const queryClient = useQueryClient();
  const setDialogOpen = useUIStore((state) => state.setProfileDialogOpen);
  const setEditingProfileId = useUIStore((state) => state.setEditingProfileId);
  const deleteProfile = useDeleteProfile();
  const scrollRef = useRef<HTMLDivElement>(null);
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;

  // Get generation counts per profile
  const generationCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    if (historyData?.items) {
      historyData.items.forEach((item) => {
        counts[item.profile_id] = (counts[item.profile_id] || 0) + 1;
      });
    }
    return counts;
  }, [historyData]);

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

  const handleEdit = (profileId: string) => {
    setEditingProfileId(profileId);
    setDialogOpen(true);
  };

  const handleDelete = (profileId: string) => {
    if (confirm('Are you sure you want to delete this profile?')) {
      deleteProfile.mutate(profileId);
    }
  };

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
        <div className="text-muted-foreground">Loading voices...</div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col relative overflow-hidden">
      {/* Scroll Mask - Always visible, behind content */}
      <div className="absolute top-0 left-0 right-0 h-16 bg-gradient-to-b from-background to-transparent z-10 pointer-events-none" />

      {/* Fixed Header */}
      <div className="absolute top-0 left-0 right-0 z-20">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">Voices</h1>
          <Button onClick={() => setDialogOpen(true)}>
            <Plus className="h-4 w-4 mr-2" />
            New Voice
          </Button>
        </div>
      </div>

      {/* Scrollable Content */}
      <div
        ref={scrollRef}
        className={cn(
          'flex-1 overflow-y-auto pt-16 relative z-0',
          isPlayerVisible && BOTTOM_SAFE_AREA_PADDING,
        )}
      >
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Language</TableHead>
              <TableHead>Generations</TableHead>
              <TableHead>Samples</TableHead>
              <TableHead>Channels</TableHead>
              <TableHead className="w-[50px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {profiles?.map((profile) => (
              <VoiceRow
                key={profile.id}
                profile={profile}
                generationCount={generationCounts[profile.id] || 0}
                channelIds={channelAssignments?.[profile.id] || []}
                channels={channels || []}
                onChannelChange={(channelIds) => handleChannelChange(profile.id, channelIds)}
                onEdit={() => handleEdit(profile.id)}
                onDelete={() => handleDelete(profile.id)}
              />
            ))}
          </TableBody>
        </Table>
      </div>

      <ProfileForm />
    </div>
  );
}

interface VoiceRowProps {
  profile: VoiceProfileResponse;
  generationCount: number;
  channelIds: string[];
  channels: Array<{ id: string; name: string; is_default: boolean }>;
  onChannelChange: (channelIds: string[]) => void;
  onEdit: () => void;
  onDelete: () => void;
}

function VoiceRow({
  profile,
  generationCount,
  channelIds,
  channels,
  onChannelChange,
  onEdit,
  onDelete,
}: VoiceRowProps) {
  const { data: samples } = useProfileSamples(profile.id);

  return (
    <TableRow className="cursor-pointer" onClick={onEdit}>
      <TableCell>
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-muted flex items-center justify-center shrink-0">
            <Mic className="h-4 w-4 text-muted-foreground" />
          </div>
          <div>
            <div className="font-medium">{profile.name}</div>
            {profile.description && (
              <div className="text-sm text-muted-foreground">{profile.description}</div>
            )}
          </div>
        </div>
      </TableCell>
      <TableCell onClick={(e) => e.stopPropagation()}>{profile.language}</TableCell>
      <TableCell onClick={(e) => e.stopPropagation()}>{generationCount}</TableCell>
      <TableCell onClick={(e) => e.stopPropagation()}>{samples?.length || 0}</TableCell>
      <TableCell onClick={(e) => e.stopPropagation()}>
        <MultiSelect
          options={channels.map((ch) => ({
            value: ch.id,
            label: `${ch.name}${ch.is_default ? ' (Default)' : ''}`,
          }))}
          value={channelIds}
          onChange={onChannelChange}
          placeholder="Select channels..."
          className="min-w-[200px]"
        />
      </TableCell>
      <TableCell onClick={(e) => e.stopPropagation()}>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent>
            <DropdownMenuItem onClick={onEdit}>
              <Edit className="h-4 w-4 mr-2" />
              Edit
            </DropdownMenuItem>
            <DropdownMenuItem onClick={onDelete} className="text-destructive">
              <Trash2 className="h-4 w-4 mr-2" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </TableCell>
    </TableRow>
  );
}
