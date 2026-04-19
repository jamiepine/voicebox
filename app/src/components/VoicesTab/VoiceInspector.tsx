import { zodResolver } from '@hookform/resolvers/zod';
import { Edit2, Mic, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { EffectsChainEditor } from '@/components/Effects/EffectsChainEditor';
import { Button } from '@/components/ui/button';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import { SampleList } from '@/components/VoiceProfiles/SampleList';
import { apiClient } from '@/lib/api/client';
import type { EffectConfig } from '@/lib/api/types';
import { LANGUAGE_CODES, LANGUAGE_OPTIONS, type LanguageCode } from '@/lib/constants/languages';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import {
  useDeleteAvatar,
  useProfile,
  useUpdateProfile,
  useUploadAvatar,
} from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';
import { usePlayerStore } from '@/stores/playerStore';
import { useServerStore } from '@/stores/serverStore';

const profileSchema = z.object({
  name: z.string().min(1, 'Name is required').max(100),
  description: z.string().max(500).optional(),
  language: z.enum(LANGUAGE_CODES as [LanguageCode, ...LanguageCode[]]),
});

type ProfileFormValues = z.infer<typeof profileSchema>;

interface VoiceInspectorProps {
  profileId: string;
}

export function VoiceInspector({ profileId }: VoiceInspectorProps) {
  const { data: profile } = useProfile(profileId);
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;
  const updateProfile = useUpdateProfile();
  const uploadAvatar = useUploadAvatar();
  const deleteAvatar = useDeleteAvatar();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const { toast } = useToast();

  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
  const [avatarError, setAvatarError] = useState(false);
  const avatarInputRef = useRef<HTMLInputElement>(null);

  const [effectsChain, setEffectsChain] = useState<EffectConfig[]>([]);
  const [effectsDirty, setEffectsDirty] = useState(false);

  const form = useForm<ProfileFormValues>({
    resolver: zodResolver(profileSchema),
    defaultValues: {
      name: '',
      description: '',
      language: 'en',
    },
  });

  // Populate form when profile loads
  useEffect(() => {
    if (profile) {
      form.reset({
        name: profile.name,
        description: profile.description || '',
        language: profile.language as LanguageCode,
      });
      setEffectsChain(profile.effects_chain ?? []);
      setEffectsDirty(false);
    }
  }, [profile, form]);

  // Avatar preview
  useEffect(() => {
    if (profile?.avatar_path) {
      setAvatarPreview(`${serverUrl}/profiles/${profile.id}/avatar`);
    } else {
      setAvatarPreview(null);
    }
    setAvatarError(false);
  }, [profile, serverUrl]);

  function handleAvatarFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      toast({
        title: 'Invalid file type',
        description: 'Please select PNG, JPG, or WebP',
        variant: 'destructive',
      });
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      toast({
        title: 'File too large',
        description: 'Image must be less than 5MB',
        variant: 'destructive',
      });
      return;
    }
    // Upload immediately
    uploadAvatar.mutate(
      { profileId, file },
      {
        onSuccess: () => {
          setAvatarPreview(URL.createObjectURL(file));
          toast({ title: 'Avatar updated' });
        },
        onError: (err) => {
          toast({
            title: 'Avatar upload failed',
            description: err instanceof Error ? err.message : 'Unknown error',
            variant: 'destructive',
          });
        },
      },
    );
  }

  async function handleRemoveAvatar() {
    if (profile?.avatar_path) {
      try {
        await deleteAvatar.mutateAsync(profileId);
        toast({ title: 'Avatar removed' });
      } catch (err) {
        toast({
          title: 'Failed to remove avatar',
          description: err instanceof Error ? err.message : 'Unknown error',
          variant: 'destructive',
        });
      }
    }
    setAvatarPreview(null);
    if (avatarInputRef.current) avatarInputRef.current.value = '';
  }

  async function onSubmit(data: ProfileFormValues) {
    try {
      await updateProfile.mutateAsync({
        profileId,
        data: {
          name: data.name,
          description: data.description,
          language: data.language,
        },
      });

      if (effectsDirty) {
        try {
          await apiClient.updateProfileEffects(
            profileId,
            effectsChain.length > 0 ? effectsChain : null,
          );
          setEffectsDirty(false);
        } catch (fxError) {
          toast({
            title: 'Effects update failed',
            description: fxError instanceof Error ? fxError.message : 'Failed to save effects',
            variant: 'destructive',
          });
          return;
        }
      }

      toast({ title: 'Voice updated', description: `"${data.name}" saved.` });
    } catch (error) {
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to save profile',
        variant: 'destructive',
      });
    }
  }

  if (!profile) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        Loading...
      </div>
    );
  }

  const isDirty = form.formState.isDirty || effectsDirty;

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className={cn('flex-1 overflow-y-auto', isPlayerVisible && BOTTOM_SAFE_AREA_PADDING)}>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-0">
            {/* Avatar */}
            <div className="flex justify-center pt-5 pb-3">
              <div className="relative group">
                <div className="h-20 w-20 rounded-full bg-muted flex items-center justify-center shrink-0 overflow-hidden border-2 border-border">
                  {avatarPreview && !avatarError ? (
                    <img
                      src={avatarPreview}
                      alt={profile.name}
                      className="h-full w-full object-cover"
                      onError={() => setAvatarError(true)}
                    />
                  ) : (
                    <Mic className="h-8 w-8 text-muted-foreground" />
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => avatarInputRef.current?.click()}
                  className="absolute inset-0 rounded-full bg-accent/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center cursor-pointer"
                >
                  <Edit2 className="h-5 w-5 text-accent-foreground" />
                </button>
                {avatarPreview && (
                  <button
                    type="button"
                    onClick={handleRemoveAvatar}
                    disabled={deleteAvatar.isPending}
                    className="absolute bottom-0 right-0 h-5 w-5 rounded-full bg-background/60 backdrop-blur-sm text-muted-foreground flex items-center justify-center hover:bg-background/80 hover:text-foreground transition-colors shadow-sm border border-border/50"
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>
              <input
                ref={avatarInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={handleAvatarFileChange}
                className="hidden"
              />
            </div>

            {/* Fields */}
            <div className="space-y-3 px-5">
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Name</FormLabel>
                    <FormControl>
                      <Input placeholder="My Voice" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="description"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Description</FormLabel>
                    <FormControl>
                      <Textarea placeholder="Describe this voice..." rows={2} {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="language"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Language</FormLabel>
                    <Select onValueChange={field.onChange} value={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {LANGUAGE_OPTIONS.map((lang) => (
                          <SelectItem key={lang.value} value={lang.value}>
                            {lang.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {/* Effects */}
              <div className="space-y-2">
                <FormLabel>Default Effects</FormLabel>
                <p className="text-xs text-muted-foreground">
                  Applied automatically to new generations with this voice.
                </p>
                <EffectsChainEditor
                  value={effectsChain}
                  onChange={(chain) => {
                    setEffectsChain(chain);
                    setEffectsDirty(true);
                  }}
                  compact
                />
              </div>

              {/* Save */}
              {isDirty && (
                <Button type="submit" className="w-full" disabled={updateProfile.isPending}>
                  {updateProfile.isPending ? 'Saving...' : 'Save Changes'}
                </Button>
              )}
            </div>

            {/* Samples */}
            <div className="px-5 pb-5">
              <SampleList profileId={profileId} />
            </div>
          </form>
        </Form>
      </div>
    </div>
  );
}
