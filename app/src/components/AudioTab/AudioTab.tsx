import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Check, CheckCircle2, Edit, Plus, Speaker, Trash2 } from 'lucide-react';
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
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { apiClient } from '@/lib/api/client';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { cn } from '@/lib/utils/cn';
import { usePlatform } from '@/platform/PlatformContext';
import { usePlayerStore } from '@/stores/playerStore';

interface AudioDevice {
  id: string;
  name: string;
  is_default: boolean;
}

export function AudioTab() {
  const { t } = useTranslation();
  const platform = usePlatform();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editingChannel, setEditingChannel] = useState<string | null>(null);
  const [selectedChannelId, setSelectedChannelId] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;

  const { data: channels, isLoading: channelsLoading } = useQuery({
    queryKey: ['channels'],
    queryFn: () => apiClient.listChannels(),
  });

  const { data: devices, isLoading: devicesLoading } = useQuery({
    queryKey: ['audio-devices'],
    queryFn: async () => {
      if (!platform.metadata.isTauri) {
        return [];
      }
      try {
        return await platform.audio.listOutputDevices();
      } catch (error) {
        console.error('Failed to list audio devices:', error);
        return [];
      }
    },
    enabled: platform.metadata.isTauri,
  });

  const { data: profiles } = useQuery({
    queryKey: ['profiles'],
    queryFn: () => apiClient.listProfiles(),
  });

  const createChannel = useMutation({
    mutationFn: (data: { name: string; device_ids: string[] }) => apiClient.createChannel(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['channels'] });
      setCreateDialogOpen(false);
    },
  });

  const updateChannel = useMutation({
    mutationFn: ({
      channelId,
      data,
    }: {
      channelId: string;
      data: { name?: string; device_ids?: string[] };
    }) => apiClient.updateChannel(channelId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['channels'] });
      queryClient.invalidateQueries({ queryKey: ['profile-channels'] });
      setEditingChannel(null);
    },
  });

  const deleteChannel = useMutation({
    mutationFn: (channelId: string) => apiClient.deleteChannel(channelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['channels'] });
      queryClient.invalidateQueries({ queryKey: ['profile-channels'] });
    },
  });

  const { data: channelVoices } = useQuery({
    queryKey: ['channel-voices', editingChannel],
    queryFn: async () => {
      if (!editingChannel) return { profile_ids: [] };
      return apiClient.getChannelVoices(editingChannel);
    },
    enabled: !!editingChannel,
  });

  const setChannelVoices = useMutation({
    mutationFn: ({ channelId, profileIds }: { channelId: string; profileIds: string[] }) =>
      apiClient.setChannelVoices(channelId, profileIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['channel-voices'] });
      queryClient.invalidateQueries({ queryKey: ['profile-channels'] });
    },
  });

  if (channelsLoading || devicesLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-muted-foreground">{t('audioChannels.loading')}</div>
      </div>
    );
  }

  const handleChannelDelete = async (e: React.MouseEvent, channelId: string) => {
    e.stopPropagation();
    if (await confirm(t('audioChannels.confirmDelete'))) {
      deleteChannel.mutate(channelId);
    }
  };

  const allChannels = channels || [];
  const allDevices = devices || [];
  const selectedChannel = selectedChannelId
    ? allChannels.find((c) => c.id === selectedChannelId)
    : null;

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-6 shrink-0">
        <h2 className="text-2xl font-bold">{t('audioChannels.title')}</h2>
        <Button onClick={() => setCreateDialogOpen(true)}>
          <Plus className="h-4 w-4 me-2" />
          {t('audioChannels.newChannel')}
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full min-h-0">
        {/* Left Column - Channels */}
        <div
          className={cn(
            'flex flex-col min-h-0 overflow-y-auto',
            isPlayerVisible && BOTTOM_SAFE_AREA_PADDING,
          )}
        >
          {allChannels.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-muted rounded-md">
              <Speaker className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground mb-4">{t('audioChannels.empty.message')}</p>
              <Button onClick={() => setCreateDialogOpen(true)}>
                <Plus className="h-4 w-4 me-2" />
                {t('audioChannels.empty.action')}
              </Button>
            </div>
          ) : (
            <div className="space-y-3">
              {allChannels.map((channel) => {
                const isSelected = selectedChannelId === channel.id;
                return (
                  <button
                    key={channel.id}
                    type="button"
                    className={cn(
                      'group border rounded-lg p-4 transition-colors cursor-pointer text-start w-full',
                      isSelected && 'ring-2 ring-primary bg-primary/5 border-primary',
                    )}
                    onClick={() => setSelectedChannelId(isSelected ? null : channel.id)}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-3">
                          <div className="h-8 w-8 rounded-lg bg-muted flex items-center justify-center shrink-0">
                            <Speaker className="h-4 w-4 text-muted-foreground" />
                          </div>
                          <div className="flex items-center gap-2 min-w-0">
                            <h3 className="font-semibold text-base truncate">{channel.name}</h3>
                          </div>
                        </div>

                        <div className="space-y-2.5 ms-10">
                          <div>
                            <div className="text-xs font-medium text-muted-foreground mb-1">
                              {t('audioChannels.labels.outputDevices')}
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                              {channel.device_ids.length > 0
                                ? channel.device_ids.map((deviceId) => {
                                    const device = allDevices.find((d) => d.id === deviceId);
                                    return (
                                      <Badge
                                        key={deviceId}
                                        variant="outline"
                                        className="text-xs font-normal"
                                      >
                                        {device?.name || deviceId}
                                      </Badge>
                                    );
                                  })
                                : (() => {
                                    const defaultDevice = allDevices.find((d) => d.is_default);
                                    return defaultDevice ? (
                                      <Badge variant="outline" className="text-xs font-normal">
                                        {defaultDevice.name}
                                      </Badge>
                                    ) : null;
                                  })()}
                            </div>
                          </div>

                          <div>
                            <div className="text-xs font-medium text-muted-foreground mb-1">
                              {t('audioChannels.labels.assignedVoices')}
                            </div>
                            <ChannelVoicesList channelId={channel.id} />
                          </div>
                        </div>
                      </div>

                      {!channel.is_default && (
                        <div className="flex gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 p-0"
                            onClick={(e) => {
                              e.stopPropagation();
                              setEditingChannel(channel.id);
                            }}
                          >
                            <Edit className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 p-0"
                            onClick={(e) => handleChannelDelete(e, channel.id)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Right Column - Available Devices */}
        <div
          className={cn(
            'flex flex-col min-h-0 overflow-y-auto',
            isPlayerVisible && BOTTOM_SAFE_AREA_PADDING,
          )}
        >
          <div className="shrink-0 mb-4">
            <h3 className="text-lg font-semibold">{t('audioChannels.devices.title')}</h3>
            <p className="text-sm text-muted-foreground mt-1">
              {selectedChannelId
                ? selectedChannel?.is_default
                  ? t('audioChannels.devices.defaultNote')
                  : t('audioChannels.devices.toggleHint')
                : t('audioChannels.devices.selectHint')}
            </p>
          </div>
          {allDevices.length > 0 ? (
            <div className="space-y-2">
              {allDevices.map((device) => {
                const isConnected =
                  selectedChannelId &&
                  selectedChannel &&
                  (selectedChannel.device_ids.length === 0
                    ? device.is_default
                    : selectedChannel.device_ids.includes(device.id));
                const canToggle =
                  selectedChannelId && selectedChannel && !selectedChannel.is_default;

                const handleDeviceClick = () => {
                  if (!canToggle || !selectedChannel) return;

                  const currentDeviceIds = selectedChannel.device_ids;
                  const newDeviceIds = isConnected
                    ? currentDeviceIds.filter((id) => id !== device.id)
                    : [...currentDeviceIds, device.id];

                  updateChannel.mutate({
                    channelId: selectedChannelId,
                    data: { device_ids: newDeviceIds },
                  });
                };

                return (
                  <button
                    key={device.id}
                    type="button"
                    onClick={handleDeviceClick}
                    disabled={!canToggle}
                    className={cn(
                      'flex items-center gap-2 text-sm p-3 rounded-lg border transition-colors text-start w-full',
                      isConnected
                        ? 'bg-primary/10 border-primary ring-1 ring-primary/20'
                        : 'hover:bg-muted/50',
                      !canToggle && 'cursor-default opacity-60',
                      canToggle && 'cursor-pointer',
                    )}
                  >
                    {canToggle ? (
                      <div
                        className={cn(
                          'h-4 w-4 rounded border-2 flex items-center justify-center shrink-0',
                          isConnected ? 'bg-accent border-accent' : 'border-muted-foreground/30',
                        )}
                      >
                        {isConnected && <Check className="h-3 w-3 text-accent-foreground" />}
                      </div>
                    ) : device.is_default ? (
                      <CheckCircle2 className="h-4 w-4 text-primary shrink-0" />
                    ) : null}
                    <span className={cn('truncate flex-1', device.is_default && 'font-medium')}>
                      {device.name}
                    </span>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-muted rounded-md">
              <CheckCircle2 className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground text-center">
                {platform.metadata.isTauri
                  ? t('audioChannels.devices.empty')
                  : t('audioChannels.devices.requiresTauri')}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Create Channel Dialog */}
      <CreateChannelDialog
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
        devices={devices || []}
        onCreate={(name, deviceIds) => {
          createChannel.mutate({ name, device_ids: deviceIds });
        }}
      />

      {/* Edit Channel Dialog */}
      {editingChannel &&
        (() => {
          const channel = channels?.find((c) => c.id === editingChannel);
          return channel ? (
            <EditChannelDialog
              open={!!editingChannel}
              onOpenChange={(open) => !open && setEditingChannel(null)}
              channel={channel}
              devices={devices || []}
              profiles={profiles || []}
              channelVoices={channelVoices?.profile_ids || []}
              onUpdate={(name, deviceIds) => {
                updateChannel.mutate({
                  channelId: editingChannel,
                  data: { name, device_ids: deviceIds },
                });
              }}
              onSetVoices={(profileIds) => {
                setChannelVoices.mutate({
                  channelId: editingChannel,
                  profileIds,
                });
              }}
            />
          ) : null;
        })()}
    </div>
  );
}

function ChannelVoicesList({ channelId }: { channelId: string }) {
  const { t } = useTranslation();
  const { data: voices } = useQuery({
    queryKey: ['channel-voices', channelId],
    queryFn: () => apiClient.getChannelVoices(channelId),
  });

  const { data: profiles } = useQuery({
    queryKey: ['profiles'],
    queryFn: () => apiClient.listProfiles(),
  });

  const voiceNames =
    voices?.profile_ids.map((id) => profiles?.find((p) => p.id === id)?.name).filter(Boolean) || [];

  return (
    <div className="flex flex-wrap gap-1.5">
      {voiceNames.length > 0 ? (
        voiceNames.map((name) => (
          <Badge key={name} variant="outline" className="text-xs font-normal">
            {name}
          </Badge>
        ))
      ) : (
        <span className="text-sm text-muted-foreground">{t('audioChannels.noVoicesAssigned')}</span>
      )}
    </div>
  );
}

interface CreateChannelDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  devices: AudioDevice[];
  onCreate: (name: string, deviceIds: string[]) => void;
}

function CreateChannelDialog({ open, onOpenChange, devices, onCreate }: CreateChannelDialogProps) {
  const { t } = useTranslation();
  const [name, setName] = useState('');
  const [selectedDevices, setSelectedDevices] = useState<string[]>([]);

  const handleSubmit = () => {
    if (name.trim()) {
      onCreate(name.trim(), selectedDevices);
      setName('');
      setSelectedDevices([]);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t('audioChannels.createDialog.title')}</DialogTitle>
          <DialogDescription>{t('audioChannels.createDialog.description')}</DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label htmlFor="channel-name">{t('audioChannels.fields.name')}</Label>
            <Input
              id="channel-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={t('audioChannels.fields.namePlaceholder')}
            />
          </div>
          <div>
            <Label>{t('audioChannels.labels.outputDevices')}</Label>
            <Select
              value={selectedDevices[0] || ''}
              onValueChange={(value) => {
                if (value && !selectedDevices.includes(value)) {
                  setSelectedDevices([...selectedDevices, value]);
                }
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder={t('audioChannels.selectDevice')} />
              </SelectTrigger>
              <SelectContent>
                {devices.map((device) => (
                  <SelectItem key={device.id} value={device.id}>
                    {device.name} {device.is_default && `(${t('audioChannels.defaultSuffix')})`}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedDevices.length > 0 && (
              <div className="mt-2 space-y-1">
                {selectedDevices.map((deviceId) => {
                  const device = devices.find((d) => d.id === deviceId);
                  return (
                    <div
                      key={deviceId}
                      className="flex items-center justify-between text-sm bg-muted p-2 rounded"
                    >
                      <span>{device?.name || deviceId}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          setSelectedDevices(selectedDevices.filter((id) => id !== deviceId))
                        }
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </Button>
          <Button onClick={handleSubmit} disabled={!name.trim()}>
            {t('audioChannels.createDialog.action')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

interface EditChannelDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  channel: {
    id: string;
    name: string;
    device_ids: string[];
  };
  devices: AudioDevice[];
  profiles: Array<{ id: string; name: string }>;
  channelVoices: string[];
  onUpdate: (name: string, deviceIds: string[]) => void;
  onSetVoices: (profileIds: string[]) => void;
}

function EditChannelDialog({
  open,
  onOpenChange,
  channel,
  devices,
  profiles,
  channelVoices,
  onUpdate,
  onSetVoices,
}: EditChannelDialogProps) {
  const { t } = useTranslation();
  const [name, setName] = useState(channel.name);
  const [selectedDevices, setSelectedDevices] = useState<string[]>(channel.device_ids);
  const [selectedVoices, setSelectedVoices] = useState<string[]>(channelVoices);

  const handleSubmit = () => {
    if (name.trim()) {
      onUpdate(name.trim(), selectedDevices);
      onSetVoices(selectedVoices);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t('audioChannels.editDialog.title')}</DialogTitle>
          <DialogDescription>{t('audioChannels.editDialog.description')}</DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label htmlFor="edit-channel-name">{t('audioChannels.fields.name')}</Label>
            <Input id="edit-channel-name" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div>
            <Label>{t('audioChannels.labels.outputDevices')}</Label>
            <Select
              value=""
              onValueChange={(value) => {
                if (value && !selectedDevices.includes(value)) {
                  setSelectedDevices([...selectedDevices, value]);
                }
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder={t('audioChannels.addDevice')} />
              </SelectTrigger>
              <SelectContent>
                {devices.map((device) => (
                  <SelectItem key={device.id} value={device.id}>
                    {device.name} {device.is_default && `(${t('audioChannels.defaultSuffix')})`}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedDevices.length > 0 && (
              <div className="mt-2 space-y-1">
                {selectedDevices.map((deviceId) => {
                  const device = devices.find((d) => d.id === deviceId);
                  return (
                    <div
                      key={deviceId}
                      className="flex items-center justify-between text-sm bg-muted p-2 rounded"
                    >
                      <span>{device?.name || deviceId}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          setSelectedDevices(selectedDevices.filter((id) => id !== deviceId))
                        }
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
          <div>
            <Label>{t('audioChannels.labels.assignedVoices')}</Label>
            <Select
              value=""
              onValueChange={(value) => {
                if (value && !selectedVoices.includes(value)) {
                  setSelectedVoices([...selectedVoices, value]);
                }
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder={t('audioChannels.addVoice')} />
              </SelectTrigger>
              <SelectContent>
                {profiles.map((profile) => (
                  <SelectItem key={profile.id} value={profile.id}>
                    {profile.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedVoices.length > 0 && (
              <div className="mt-2 space-y-1">
                {selectedVoices.map((profileId) => {
                  const profile = profiles.find((p) => p.id === profileId);
                  return (
                    <div
                      key={profileId}
                      className="flex items-center justify-between text-sm bg-muted p-2 rounded"
                    >
                      <span>{profile?.name || profileId}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          setSelectedVoices(selectedVoices.filter((id) => id !== profileId))
                        }
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t('common.cancel')}
          </Button>
          <Button onClick={handleSubmit} disabled={!name.trim()}>
            {t('common.save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
