import { zodResolver } from '@hookform/resolvers/zod';
import { useQuery } from '@tanstack/react-query';
import { Edit2, Mic, Monitor, Music, Upload, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import * as z from 'zod';
import { EffectsChainEditor } from '@/components/Effects/EffectsChainEditor';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Form,
  FormControl,
  FormDescription,
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { EffectConfig, PresetVoice, VoiceType } from '@/lib/api/types';
import { LANGUAGE_CODES, LANGUAGE_OPTIONS, type LanguageCode } from '@/lib/constants/languages';
import { useAudioPlayer } from '@/lib/hooks/useAudioPlayer';
import { useAudioRecording } from '@/lib/hooks/useAudioRecording';
import {
  useAddSample,
  useCreateProfile,
  useDeleteAvatar,
  useDeleteProfile,
  useProfile,
  useUpdateProfile,
  useUploadAvatar,
} from '@/lib/hooks/useProfiles';
import { useSystemAudioCapture } from '@/lib/hooks/useSystemAudioCapture';
import { useTranscription } from '@/lib/hooks/useTranscription';
import { convertToWav, formatAudioDuration, getAudioDuration } from '@/lib/utils/audio';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';
import { type ProfileFormDraft, useUIStore } from '@/stores/uiStore';
import { AudioSampleRecording } from './AudioSampleRecording';
import { AudioSampleSystem } from './AudioSampleSystem';
import { AudioSampleUpload } from './AudioSampleUpload';
import { SampleList } from './SampleList';

const MAX_AUDIO_DURATION_SECONDS = 30;
const PRESET_ONLY_ENGINES = new Set(['kokoro', 'qwen_custom_voice']);
const DEFAULT_ENGINE_OPTIONS = [
  { value: 'qwen', label: 'Qwen3-TTS' },
  { value: 'qwen_custom_voice', label: 'Qwen CustomVoice' },
  { value: 'luxtts', label: 'LuxTTS' },
  { value: 'chatterbox', label: 'Chatterbox' },
  { value: 'chatterbox_turbo', label: 'Chatterbox Turbo' },
  { value: 'tada', label: 'TADA' },
  { value: 'kokoro', label: 'Kokoro 82M' },
] as const;

function makeProfileSchema(t: (key: string) => string) {
  const baseProfileSchema = z.object({
    name: z.string().min(1, t('profileForm.validation.nameRequired')).max(100),
    description: z.string().max(500).optional(),
    language: z.enum(LANGUAGE_CODES as [LanguageCode, ...LanguageCode[]]),
    personality: z.string().max(2000).optional(),
    sampleFile: z.instanceof(File).optional(),
    referenceText: z.string().max(1000).optional(),
    avatarFile: z.instanceof(File).optional(),
  });

  return baseProfileSchema.refine(
    (data) => {
      if (data.sampleFile && (!data.referenceText || data.referenceText.trim().length === 0)) {
        return false;
      }
      return true;
    },
    {
      message: t('profileForm.validation.referenceRequired'),
      path: ['referenceText'],
    },
  );
}

type ProfileFormValues = {
  name: string;
  description?: string;
  language: LanguageCode;
  personality?: string;
  sampleFile?: File;
  referenceText?: string;
  avatarFile?: File;
};

// Helper to convert File to base64
async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Helper to convert base64 to File
function base64ToFile(base64: string, fileName: string, fileType: string): File {
  const arr = base64.split(',');
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], fileName, { type: fileType });
}

export function ProfileForm() {
  const { t } = useTranslation();
  const platform = usePlatform();
  const open = useUIStore((state) => state.profileDialogOpen);
  const setOpen = useUIStore((state) => state.setProfileDialogOpen);
  const editingProfileId = useUIStore((state) => state.editingProfileId);
  const setEditingProfileId = useUIStore((state) => state.setEditingProfileId);
  const profileFormDraft = useUIStore((state) => state.profileFormDraft);
  const setProfileFormDraft = useUIStore((state) => state.setProfileFormDraft);
  const { data: editingProfile } = useProfile(editingProfileId || '');
  const createProfile = useCreateProfile();
  const updateProfile = useUpdateProfile();
  const addSample = useAddSample();
  const deleteProfile = useDeleteProfile();
  const uploadAvatar = useUploadAvatar();
  const deleteAvatar = useDeleteAvatar();
  const transcribe = useTranscription();
  const { toast } = useToast();
  const [voiceSource, setVoiceSource] = useState<'clone' | 'builtin'>('clone');
  const [sampleMode, setSampleMode] = useState<'upload' | 'record' | 'system'>('record');
  const [audioDuration, setAudioDuration] = useState<number | null>(null);
  const [isValidatingAudio, setIsValidatingAudio] = useState(false);
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
  const [selectedPresetEngine, setSelectedPresetEngine] = useState<string>('kokoro');
  const [selectedPresetVoiceId, setSelectedPresetVoiceId] = useState<string>('');
  const avatarInputRef = useRef<HTMLInputElement>(null);
  const { isPlaying, playPause, cleanup: cleanupAudio } = useAudioPlayer();
  const isCreating = !editingProfileId;
  const serverUrl = useServerStore((state) => state.serverUrl);
  const [profileEffectsChain, setProfileEffectsChain] = useState<EffectConfig[]>([]);
  const [effectsDirty, setEffectsDirty] = useState(false);
  const [defaultEngine, setDefaultEngine] = useState<string>('');

  const form = useForm<ProfileFormValues>({
    resolver: zodResolver(makeProfileSchema(t)),
    defaultValues: {
      name: '',
      description: '',
      language: 'en',
      personality: '',
      sampleFile: undefined,
      referenceText: '',
      avatarFile: undefined,
    },
  });

  const selectedFile = form.watch('sampleFile');
  const selectedAvatarFile = form.watch('avatarFile');

  // Validate audio duration when file is selected
  useEffect(() => {
    if (selectedFile && selectedFile instanceof File) {
      setIsValidatingAudio(true);
      getAudioDuration(selectedFile as File & { recordedDuration?: number })
        .then((duration) => {
          setAudioDuration(duration);
          if (duration > MAX_AUDIO_DURATION_SECONDS) {
            form.setError('sampleFile', {
              type: 'manual',
              message: t('profileForm.validation.audioTooLong', {
                duration: formatAudioDuration(duration),
                max: formatAudioDuration(MAX_AUDIO_DURATION_SECONDS),
              }),
            });
          } else {
            form.clearErrors('sampleFile');
          }
        })
        .catch((error) => {
          console.error('Failed to get audio duration:', error);
          setAudioDuration(null);
          const isRecordedFile =
            selectedFile.name.startsWith('recording-') ||
            selectedFile.name.startsWith('system-audio-');
          if (!isRecordedFile) {
            form.setError('sampleFile', {
              type: 'manual',
              message: t('profileForm.validation.audioFailed'),
            });
          } else {
            // Clear any existing errors for recorded files
            form.clearErrors('sampleFile');
          }
        })
        .finally(() => {
          setIsValidatingAudio(false);
        });
    } else {
      setAudioDuration(null);
      form.clearErrors('sampleFile');
    }
  }, [selectedFile, form, t]);

  const {
    isRecording,
    duration,
    error: recordingError,
    startRecording,
    stopRecording,
    cancelRecording,
  } = useAudioRecording({
    maxDurationSeconds: 29,
    onRecordingComplete: (blob, recordedDuration) => {
      const file = new File([blob], `recording-${Date.now()}.webm`, {
        type: blob.type || 'audio/webm',
      }) as File & { recordedDuration?: number };
      // Store the actual recorded duration to bypass metadata reading issues on Windows
      if (recordedDuration !== undefined) {
        file.recordedDuration = recordedDuration;
      }
      form.setValue('sampleFile', file, { shouldValidate: true });
      toast({
        title: t('profileForm.toast.recordingComplete'),
        description: t('profileForm.toast.recordingCompleteDescription'),
      });
    },
  });

  const {
    isRecording: isSystemRecording,
    duration: systemDuration,
    error: systemRecordingError,
    isSupported: isSystemAudioSupported,
    startRecording: startSystemRecording,
    stopRecording: stopSystemRecording,
    cancelRecording: cancelSystemRecording,
  } = useSystemAudioCapture({
    maxDurationSeconds: 29,
    onRecordingComplete: (blob, recordedDuration) => {
      const file = new File([blob], `system-audio-${Date.now()}.wav`, {
        type: blob.type || 'audio/wav',
      }) as File & { recordedDuration?: number };
      // Store the actual recorded duration to bypass metadata reading issues on Windows
      if (recordedDuration !== undefined) {
        file.recordedDuration = recordedDuration;
      }
      form.setValue('sampleFile', file, { shouldValidate: true });
      toast({
        title: t('profileForm.toast.systemAudioCaptured'),
        description: t('profileForm.toast.systemAudioCapturedDescription'),
      });
    },
  });

  // Fetch available preset voices for the selected engine
  const presetEngineToQuery = isCreating
    ? selectedPresetEngine
    : (editingProfile?.preset_engine ?? '');
  const { data: presetVoicesData } = useQuery({
    queryKey: ['presetVoices', presetEngineToQuery],
    queryFn: () => apiClient.listPresetVoices(presetEngineToQuery),
    enabled:
      !!presetEngineToQuery &&
      ((voiceSource === 'builtin' && isCreating) ||
        (!isCreating && editingProfile?.voice_type === 'preset')),
  });
  const presetVoices = presetVoicesData?.voices ?? [];
  const isSampleBasedProfile = isCreating
    ? voiceSource === 'clone'
    : editingProfile?.voice_type !== 'preset';
  const availableDefaultEngines = DEFAULT_ENGINE_OPTIONS.filter(
    (option) => !isSampleBasedProfile || !PRESET_ONLY_ENGINES.has(option.value),
  );

  // Show recording errors
  useEffect(() => {
    if (recordingError) {
      toast({
        title: t('profileForm.toast.recordingError'),
        description: recordingError,
        variant: 'destructive',
      });
    }
  }, [recordingError, toast, t]);

  useEffect(() => {
    if (systemRecordingError) {
      toast({
        title: t('profileForm.toast.systemAudioError'),
        description: systemRecordingError,
        variant: 'destructive',
      });
    }
  }, [systemRecordingError, toast, t]);

  // Handle avatar preview
  useEffect(() => {
    if (selectedAvatarFile instanceof File) {
      const url = URL.createObjectURL(selectedAvatarFile);
      setAvatarPreview(url);
      return () => URL.revokeObjectURL(url);
    } else if (editingProfile?.avatar_path) {
      setAvatarPreview(`${serverUrl}/profiles/${editingProfile.id}/avatar`);
    } else {
      setAvatarPreview(null);
    }
  }, [selectedAvatarFile, editingProfile, serverUrl]);

  // Restore form state from draft or editing profile
  useEffect(() => {
    if (editingProfile) {
      form.reset({
        name: editingProfile.name,
        description: editingProfile.description || '',
        language: editingProfile.language as LanguageCode,
        personality: editingProfile.personality || '',
        sampleFile: undefined,
        referenceText: undefined,
        avatarFile: undefined,
      });
      setProfileEffectsChain(editingProfile.effects_chain ?? []);
      setEffectsDirty(false);
      setDefaultEngine(editingProfile.default_engine ?? '');
    } else if (profileFormDraft && open) {
      // Restore from draft when opening in create mode
      form.reset({
        name: profileFormDraft.name,
        description: profileFormDraft.description,
        language: profileFormDraft.language as LanguageCode,
        personality: profileFormDraft.personality || '',
        referenceText: profileFormDraft.referenceText,
        sampleFile: undefined,
        avatarFile: undefined,
      });
      setSampleMode(profileFormDraft.sampleMode);
      // Restore the file if we have it saved
      if (
        profileFormDraft.sampleFileData &&
        profileFormDraft.sampleFileName &&
        profileFormDraft.sampleFileType
      ) {
        const file = base64ToFile(
          profileFormDraft.sampleFileData,
          profileFormDraft.sampleFileName,
          profileFormDraft.sampleFileType,
        );
        form.setValue('sampleFile', file);
      }
    } else if (!open) {
      // Only reset to defaults when modal is closed and no draft
      form.reset({
        name: '',
        description: '',
        language: 'en',
        personality: '',
        sampleFile: undefined,
        referenceText: undefined,
        avatarFile: undefined,
      });
      setSampleMode('record');
      setAvatarPreview(null);
    }
  }, [editingProfile, profileFormDraft, open, form]);

  useEffect(() => {
    if (
      defaultEngine &&
      !availableDefaultEngines.some((option) => option.value === defaultEngine)
    ) {
      setDefaultEngine('');
    }
  }, [availableDefaultEngines, defaultEngine]);

  useEffect(() => {
    if (!selectedPresetVoiceId) {
      return;
    }

    if (!presetVoices.some((voice: PresetVoice) => voice.voice_id === selectedPresetVoiceId)) {
      setSelectedPresetVoiceId('');
    }
  }, [presetVoices, selectedPresetVoiceId]);
  async function handleTranscribe() {
    const file = form.getValues('sampleFile');
    if (!file) {
      toast({
        title: t('profileForm.toast.noFile'),
        description: t('profileForm.toast.noFileDescription'),
        variant: 'destructive',
      });
      return;
    }

    try {
      const language = form.getValues('language');
      const result = await transcribe.mutateAsync({ file, language });

      form.setValue('referenceText', result.text, { shouldValidate: true });
    } catch (error) {
      toast({
        title: t('profileForm.toast.transcribeFailed'),
        description:
          error instanceof Error ? error.message : t('profileForm.toast.transcribeFailedFallback'),
        variant: 'destructive',
      });
    }
  }

  function handleCancelRecording() {
    if (sampleMode === 'record') {
      cancelRecording();
    } else if (sampleMode === 'system') {
      cancelSystemRecording();
    }
    form.resetField('sampleFile');
    cleanupAudio();
  }

  function handlePlayPause() {
    const file = form.getValues('sampleFile');
    playPause(file);
  }

  function handleAvatarFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        toast({
          title: t('profileForm.toast.invalidFile'),
          description: t('profileForm.toast.invalidImageFormat'),
          variant: 'destructive',
        });
        return;
      }
      if (file.size > 5 * 1024 * 1024) {
        toast({
          title: t('profileForm.toast.fileTooLarge'),
          description: t('profileForm.toast.imageTooLargeDescription'),
          variant: 'destructive',
        });
        return;
      }
      form.setValue('avatarFile', file);
    }
  }

  async function handleRemoveAvatar() {
    if (editingProfileId && editingProfile?.avatar_path) {
      try {
        await deleteAvatar.mutateAsync(editingProfileId);
        toast({
          title: t('profileForm.toast.avatarRemoved'),
          description: t('profileForm.toast.avatarRemovedDescription'),
        });
      } catch (error) {
        toast({
          title: t('profileForm.toast.avatarRemoveFailed'),
          description: error instanceof Error ? error.message : t('common.unknownError'),
          variant: 'destructive',
        });
      }
    }
    form.setValue('avatarFile', undefined);
    setAvatarPreview(null);
    if (avatarInputRef.current) {
      avatarInputRef.current.value = '';
    }
  }

  async function onSubmit(data: ProfileFormValues) {
    try {
      if (editingProfileId) {
        // Editing: update profile
        await updateProfile.mutateAsync({
          profileId: editingProfileId,
          data: {
            name: data.name,
            description: data.description,
            language: data.language,
            default_engine: defaultEngine || undefined,
            personality: data.personality?.trim() ? data.personality.trim() : undefined,
          },
        });

        // Handle avatar upload/update if file changed
        if (data.avatarFile) {
          try {
            await uploadAvatar.mutateAsync({
              profileId: editingProfileId,
              file: data.avatarFile,
            });
          } catch (avatarError) {
            toast({
              title: t('profileForm.toast.avatarUploadFailed'),
              description:
                avatarError instanceof Error
                  ? avatarError.message
                  : t('profileForm.toast.avatarUploadFailedFallback'),
              variant: 'destructive',
            });
          }
        }

        // Save effects chain if changed
        if (effectsDirty) {
          try {
            await apiClient.updateProfileEffects(
              editingProfileId,
              profileEffectsChain.length > 0 ? profileEffectsChain : null,
            );
          } catch (fxError) {
            toast({
              title: t('profileForm.toast.effectsUpdateFailed'),
              description:
                fxError instanceof Error
                  ? fxError.message
                  : t('profileForm.toast.effectsUpdateFailedFallback'),
              variant: 'destructive',
            });
            return;
          }
        }

        toast({
          title: t('profileForm.toast.voiceUpdated'),
          description: t('profileForm.toast.voiceUpdatedDescription', { name: data.name }),
        });
      } else if (voiceSource === 'builtin') {
        // Creating preset profile from built-in voice
        if (!selectedPresetVoiceId) {
          toast({
            title: t('profileForm.toast.noVoiceSelected'),
            description: t('profileForm.toast.noVoiceSelectedDescription'),
            variant: 'destructive',
          });
          return;
        }

        const profile = await createProfile.mutateAsync({
          name: data.name,
          description: data.description,
          language: data.language,
          voice_type: 'preset' as VoiceType,
          preset_engine: selectedPresetEngine,
          preset_voice_id: selectedPresetVoiceId,
          default_engine: selectedPresetEngine,
          personality: data.personality?.trim() ? data.personality.trim() : undefined,
        });

        // Handle avatar upload if provided
        if (data.avatarFile) {
          try {
            await uploadAvatar.mutateAsync({
              profileId: profile.id,
              file: data.avatarFile,
            });
          } catch (avatarError) {
            toast({
              title: t('profileForm.toast.avatarUploadFailed'),
              description:
                avatarError instanceof Error
                  ? avatarError.message
                  : t('profileForm.toast.avatarUploadFailedFallback'),
              variant: 'destructive',
            });
          }
        }

        toast({
          title: t('profileForm.toast.profileCreated'),
          description: t('profileForm.toast.profileCreatedBuiltin', { name: data.name }),
        });
      } else {
        // Creating cloned profile: require sample file and reference text
        const sampleFile = form.getValues('sampleFile');
        const referenceText = form.getValues('referenceText');

        if (!sampleFile) {
          form.setError('sampleFile', {
            type: 'manual',
            message: t('profileForm.validation.sampleRequired'),
          });
          toast({
            title: t('profileForm.toast.sampleRequired'),
            description: t('profileForm.toast.sampleRequiredDescription'),
            variant: 'destructive',
          });
          return;
        }

        if (!referenceText || referenceText.trim().length === 0) {
          form.setError('referenceText', {
            type: 'manual',
            message: t('profileForm.validation.referenceTextRequired'),
          });
          toast({
            title: t('profileForm.toast.referenceTextRequired'),
            description: t('profileForm.toast.referenceTextRequiredDescription'),
            variant: 'destructive',
          });
          return;
        }

        try {
          const duration = await getAudioDuration(sampleFile);
          if (duration > MAX_AUDIO_DURATION_SECONDS) {
            form.setError('sampleFile', {
              type: 'manual',
              message: t('profileForm.validation.audioTooLong', {
                duration: formatAudioDuration(duration),
                max: formatAudioDuration(MAX_AUDIO_DURATION_SECONDS),
              }),
            });
            toast({
              title: t('profileForm.toast.invalidAudio'),
              description: t('profileForm.toast.invalidAudioDescription', {
                duration: formatAudioDuration(duration),
                max: formatAudioDuration(MAX_AUDIO_DURATION_SECONDS),
              }),
              variant: 'destructive',
            });
            return;
          }
        } catch (error) {
          form.setError('sampleFile', {
            type: 'manual',
            message: t('profileForm.validation.audioFailed'),
          });
          toast({
            title: t('profileForm.toast.validationError'),
            description:
              error instanceof Error ? error.message : t('profileForm.validation.audioFailed'),
            variant: 'destructive',
          });
          return;
        }

        // Creating: create profile, then add sample
        const profile = await createProfile.mutateAsync({
          name: data.name,
          description: data.description,
          language: data.language,
          default_engine: defaultEngine || undefined,
          personality: data.personality?.trim() ? data.personality.trim() : undefined,
        });

        // Convert non-WAV uploads to WAV so the backend can always use soundfile.
        // Recorded audio is already WAV (from useAudioRecording's convertToWav call).
        let fileToUpload: File = sampleFile;
        if (!sampleFile.type.includes('wav') && !sampleFile.name.toLowerCase().endsWith('.wav')) {
          try {
            const wavBlob = await convertToWav(sampleFile);
            const wavName = sampleFile.name.replace(/\.[^.]+$/, '.wav');
            fileToUpload = new File([wavBlob], wavName, { type: 'audio/wav' });
          } catch {
            // If browser can't decode the format, send the original and let the backend try.
          }
        }

        try {
          await addSample.mutateAsync({
            profileId: profile.id,
            file: fileToUpload,
            referenceText: referenceText,
          });

          // Handle avatar upload if provided
          if (data.avatarFile) {
            try {
              await uploadAvatar.mutateAsync({
                profileId: profile.id,
                file: data.avatarFile,
              });
            } catch (avatarError) {
              toast({
                title: 'Avatar upload failed',
                description:
                  avatarError instanceof Error ? avatarError.message : 'Failed to upload avatar',
                variant: 'destructive',
              });
            }
          }

          toast({
            title: t('profileForm.toast.profileCreated'),
            description: t('profileForm.toast.profileCreatedSample', { name: data.name }),
          });
        } catch (sampleError) {
          let rollbackSucceeded = false;
          try {
            await deleteProfile.mutateAsync(profile.id);
            rollbackSucceeded = true;
          } catch (rollbackError) {
            toast({
              title: t('profileForm.toast.rollbackFailed'),
              description:
                rollbackError instanceof Error
                  ? rollbackError.message
                  : t('profileForm.toast.rollbackFailedDescription'),
              variant: 'destructive',
            });
          }

          const rollbackSuffix = rollbackSucceeded
            ? ` ${t('profileForm.toast.profileRolledBack')}`
            : '';
          toast({
            title: t('profileForm.toast.sampleFailed'),
            description:
              sampleError instanceof Error
                ? `${sampleError.message}${rollbackSuffix}`
                : rollbackSucceeded
                  ? t('profileForm.toast.sampleFailedRolledBack')
                  : t('profileForm.toast.sampleFailedDescription'),
            variant: 'destructive',
          });
          return;
        }
      }

      // Clear draft and reset form on success
      setProfileFormDraft(null);
      form.reset();
      setEditingProfileId(null);
      setOpen(false);
    } catch (error) {
      toast({
        title: t('common.error'),
        description: error instanceof Error ? error.message : t('profileForm.toast.saveFailed'),
        variant: 'destructive',
      });
    }
  }

  async function handleOpenChange(newOpen: boolean) {
    if (!newOpen && isCreating) {
      // Save draft when closing the create modal
      const values = form.getValues();
      const hasContent =
        values.name || values.description || values.referenceText || values.sampleFile;

      if (hasContent) {
        const draft: ProfileFormDraft = {
          name: values.name || '',
          description: values.description || '',
          language: values.language || 'en',
          personality: values.personality || '',
          referenceText: values.referenceText || '',
          sampleMode,
        };

        // Save file as base64 if present
        if (values.sampleFile) {
          try {
            draft.sampleFileName = values.sampleFile.name;
            draft.sampleFileType = values.sampleFile.type;
            draft.sampleFileData = await fileToBase64(values.sampleFile);
          } catch {
            // If file conversion fails, just don't save the file
          }
        }

        setProfileFormDraft(draft);
      }
    }

    setOpen(newOpen);
    if (!newOpen) {
      setEditingProfileId(null);
      // Don't reset form here - let the effect handle it based on draft state
      if (isRecording) {
        cancelRecording();
      }
      if (isSystemRecording) {
        cancelSystemRecording();
      }
      cleanupAudio();
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-none w-screen h-screen left-0 top-0 translate-x-0 translate-y-0 rounded-none p-6 overflow-hidden">
        <div className="max-w-5xl h-[85vh] mx-auto my-auto w-full flex flex-col overflow-hidden">
          <DialogHeader>
            <DialogTitle className="text-2xl">
              {editingProfileId ? t('profileForm.editTitle') : t('profileForm.createTitle')}
            </DialogTitle>
            <DialogDescription>
              {editingProfileId
                ? t('profileForm.editDescription')
                : t('profileForm.createDescription')}
            </DialogDescription>
            {isCreating && profileFormDraft && (
              <div className="flex items-center gap-2 pt-2">
                <span className="text-xs text-muted-foreground">
                  {t('profileForm.draftRestored')}
                </span>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-6 px-2 text-xs text-muted-foreground"
                  onClick={() => {
                    setProfileFormDraft(null);
                    form.reset({
                      name: '',
                      description: '',
                      language: 'en',
                      personality: '',
                      sampleFile: undefined,
                      referenceText: '',
                      avatarFile: undefined,
                    });
                    setSampleMode('record');
                  }}
                >
                  <X className="h-3 w-3 mr-1" />
                  {t('profileForm.discard')}
                </Button>
              </div>
            )}
          </DialogHeader>

          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="flex-1 min-h-0 flex flex-col">
              <div className="grid gap-6 grid-cols-2 flex-1 min-h-0 overflow-hidden">
                {/* Left column: Sample management */}
                <div className="space-y-4 border-r pr-6 overflow-y-auto min-h-0">
                  {isCreating ? (
                    <>
                      {/* Voice source selector */}
                      <div className="flex pt-4 pb-2">
                        <div className="inline-flex rounded-lg border border-border p-0.5 bg-muted/50">
                          <button
                            type="button"
                            onClick={() => setVoiceSource('clone')}
                            className={`inline-flex items-center gap-2 px-3 py-1.5 text-sm rounded-md transition-colors ${
                              voiceSource === 'clone'
                                ? 'bg-accent text-accent-foreground shadow-sm'
                                : 'text-muted-foreground hover:text-foreground'
                            }`}
                          >
                            <Mic className="h-3.5 w-3.5" />
                            {t('profileForm.source.clone')}
                          </button>
                          <button
                            type="button"
                            onClick={() => setVoiceSource('builtin')}
                            className={`inline-flex items-center gap-2 px-3 py-1.5 text-sm rounded-md transition-colors ${
                              voiceSource === 'builtin'
                                ? 'bg-accent text-accent-foreground shadow-sm'
                                : 'text-muted-foreground hover:text-foreground'
                            }`}
                          >
                            <Music className="h-3.5 w-3.5" />
                            {t('profileForm.source.builtin')}
                          </button>
                        </div>
                      </div>

                      {voiceSource === 'builtin' ? (
                        <div className="space-y-4">
                          <FormDescription>{t('profileForm.builtin.hint')}</FormDescription>

                          <FormItem>
                            <FormLabel>{t('profileForm.fields.engine')}</FormLabel>
                            <Select
                              value={selectedPresetEngine}
                              onValueChange={setSelectedPresetEngine}
                            >
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="kokoro">Kokoro 82M</SelectItem>
                                <SelectItem value="qwen_custom_voice">Qwen CustomVoice</SelectItem>
                              </SelectContent>
                            </Select>
                          </FormItem>

                          {/* Voice picker */}
                          <FormItem>
                            <FormLabel>{t('profileForm.fields.voice')}</FormLabel>
                            <div className="grid grid-cols-2 gap-1.5 max-h-[340px] overflow-y-auto pr-1">
                              {presetVoices.map((voice: PresetVoice) => (
                                <button
                                  key={voice.voice_id}
                                  type="button"
                                  onClick={() => {
                                    setSelectedPresetVoiceId(voice.voice_id);
                                    // Auto-set language from voice
                                    if (voice.language) {
                                      form.setValue('language', voice.language as LanguageCode);
                                    }
                                  }}
                                  className={`text-left px-3 py-2 rounded-md border text-sm transition-colors ${
                                    selectedPresetVoiceId === voice.voice_id
                                      ? 'border-accent bg-accent/10 text-accent-foreground'
                                      : 'border-border hover:bg-muted'
                                  }`}
                                >
                                  <div className="font-medium">{voice.name}</div>
                                  <div className="flex gap-1.5 mt-0.5">
                                    <Badge variant="outline" className="text-[10px] h-4 px-1">
                                      {voice.gender}
                                    </Badge>
                                    <Badge variant="outline" className="text-[10px] h-4 px-1">
                                      {voice.language}
                                    </Badge>
                                  </div>
                                </button>
                              ))}
                            </div>
                          </FormItem>
                        </div>
                      ) : (
                        <>
                          <Tabs
                            className="pt-0"
                            value={sampleMode}
                            onValueChange={(v) => {
                              const newMode = v as 'upload' | 'record' | 'system';
                              // Cancel any active recordings when switching modes
                              if (isRecording && newMode !== 'record') {
                                cancelRecording();
                              }
                              if (isSystemRecording && newMode !== 'system') {
                                cancelSystemRecording();
                              }
                              setSampleMode(newMode);
                            }}
                          >
                            <TabsList
                              className={`grid w-full ${platform.metadata.isTauri && isSystemAudioSupported ? 'grid-cols-3' : 'grid-cols-2'}`}
                            >
                              <TabsTrigger value="upload" className="flex items-center gap-2">
                                <Upload className="h-4 w-4 shrink-0" />
                                {t('profileForm.sampleTabs.upload')}
                              </TabsTrigger>
                              <TabsTrigger value="record" className="flex items-center gap-2">
                                <Mic className="h-4 w-4 shrink-0" />
                                {t('profileForm.sampleTabs.record')}
                              </TabsTrigger>
                              {platform.metadata.isTauri && isSystemAudioSupported && (
                                <TabsTrigger value="system" className="flex items-center gap-2">
                                  <Monitor className="h-4 w-4 shrink-0" />
                                  {t('profileForm.sampleTabs.system')}
                                </TabsTrigger>
                              )}
                            </TabsList>

                            <TabsContent value="upload" className="space-y-4">
                              <FormField
                                control={form.control}
                                name="sampleFile"
                                render={({ field: { onChange, name } }) => (
                                  <AudioSampleUpload
                                    file={selectedFile}
                                    onFileChange={onChange}
                                    onTranscribe={handleTranscribe}
                                    onPlayPause={handlePlayPause}
                                    isPlaying={isPlaying}
                                    isValidating={isValidatingAudio}
                                    isTranscribing={transcribe.isPending}
                                    isDisabled={
                                      audioDuration !== null &&
                                      audioDuration > MAX_AUDIO_DURATION_SECONDS
                                    }
                                    fieldName={name}
                                  />
                                )}
                              />
                            </TabsContent>

                            <TabsContent value="record" className="space-y-4">
                              <FormField
                                control={form.control}
                                name="sampleFile"
                                render={() => (
                                  <AudioSampleRecording
                                    file={selectedFile}
                                    isRecording={isRecording}
                                    duration={duration}
                                    onStart={startRecording}
                                    onStop={stopRecording}
                                    onCancel={handleCancelRecording}
                                    onTranscribe={handleTranscribe}
                                    onPlayPause={handlePlayPause}
                                    isPlaying={isPlaying}
                                    isTranscribing={transcribe.isPending}
                                  />
                                )}
                              />
                            </TabsContent>

                            {platform.metadata.isTauri && isSystemAudioSupported && (
                              <TabsContent value="system" className="space-y-4">
                                <FormField
                                  control={form.control}
                                  name="sampleFile"
                                  render={() => (
                                    <AudioSampleSystem
                                      file={selectedFile}
                                      isRecording={isSystemRecording}
                                      duration={systemDuration}
                                      onStart={startSystemRecording}
                                      onStop={stopSystemRecording}
                                      onCancel={handleCancelRecording}
                                      onTranscribe={handleTranscribe}
                                      onPlayPause={handlePlayPause}
                                      isPlaying={isPlaying}
                                      isTranscribing={transcribe.isPending}
                                    />
                                  )}
                                />
                              </TabsContent>
                            )}
                          </Tabs>

                          <FormField
                            control={form.control}
                            name="referenceText"
                            render={({ field }) => (
                              <FormItem>
                                <FormLabel>{t('profileForm.fields.referenceText')}</FormLabel>
                                <FormControl>
                                  <Textarea
                                    placeholder={t('profileForm.fields.referenceTextPlaceholder')}
                                    className="min-h-[100px]"
                                    {...field}
                                  />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />
                        </>
                      )}
                    </>
                  ) : (
                    // Editing mode
                    editingProfileId &&
                    editingProfile &&
                    (editingProfile.voice_type === 'preset' ? (
                      <div className="space-y-4 pt-4">
                        <div className="rounded-lg border border-border p-4 space-y-3">
                          <div className="text-sm font-medium text-muted-foreground">
                            {t('profileForm.builtin.badge')}
                          </div>
                          <div className="flex items-center gap-3">
                            <div className="text-lg font-semibold">
                              {presetVoices.find(
                                (v: PresetVoice) => v.voice_id === editingProfile.preset_voice_id,
                              )?.name ?? editingProfile.preset_voice_id}
                            </div>
                            <Badge variant="secondary" className="text-xs">
                              {editingProfile.preset_engine}
                            </Badge>
                          </div>
                          {(() => {
                            const voice = presetVoices.find(
                              (v: PresetVoice) => v.voice_id === editingProfile.preset_voice_id,
                            );
                            return voice ? (
                              <div className="flex gap-1.5">
                                <Badge variant="outline" className="text-xs">
                                  {voice.gender}
                                </Badge>
                                <Badge variant="outline" className="text-xs">
                                  {voice.language}
                                </Badge>
                              </div>
                            ) : null;
                          })()}
                        </div>
                        <p className="text-xs text-muted-foreground">
                          {t('profileForm.builtin.note')}
                        </p>
                      </div>
                    ) : (
                      <div>
                        <SampleList profileId={editingProfileId} />
                      </div>
                    ))
                  )}
                </div>

                {/* Right column: Profile info */}
                <div className="space-y-4 overflow-y-auto min-h-0">
                  {/* Avatar Upload */}
                  <FormField
                    control={form.control}
                    name="avatarFile"
                    render={() => (
                      <FormItem>
                        <FormControl>
                          <div className="flex justify-center pt-4 pb-2">
                            <div className="relative group">
                              <div className="h-24 w-24 rounded-full bg-muted flex items-center justify-center shrink-0 overflow-hidden border-2 border-border">
                                {avatarPreview ? (
                                  <img
                                    src={avatarPreview}
                                    alt={t('profileForm.avatar.alt')}
                                    className="h-full w-full object-cover"
                                  />
                                ) : (
                                  <Mic className="h-10 w-10 text-muted-foreground" />
                                )}
                              </div>
                              <button
                                type="button"
                                onClick={() => avatarInputRef.current?.click()}
                                className="absolute inset-0 rounded-full bg-accent/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center cursor-pointer"
                              >
                                <Edit2 className="h-6 w-6 text-accent-foreground" />
                              </button>
                              {(avatarPreview || editingProfile?.avatar_path) && (
                                <button
                                  type="button"
                                  onClick={handleRemoveAvatar}
                                  disabled={deleteAvatar.isPending}
                                  className="absolute bottom-0 right-0 h-6 w-6 rounded-full bg-background/60 backdrop-blur-sm text-muted-foreground flex items-center justify-center hover:bg-background/80 hover:text-foreground transition-colors shadow-sm border border-border/50"
                                >
                                  <X className="h-3.5 w-3.5" />
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
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="name"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>{t('profileForm.fields.name')}</FormLabel>
                        <FormControl>
                          <Input placeholder={t('profileForm.fields.namePlaceholder')} {...field} />
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
                        <FormLabel>{t('profileForm.fields.descriptionLabel')}</FormLabel>
                        <FormControl>
                          <Textarea
                            placeholder={t('profileForm.fields.descriptionPlaceholder')}
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="personality"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>{t('profileForm.fields.personalityLabel')}</FormLabel>
                        <FormControl>
                          <Textarea
                            placeholder={t('profileForm.fields.personalityPlaceholder')}
                            className="min-h-[96px]"
                            {...field}
                          />
                        </FormControl>
                        <FormDescription>
                          {t('profileForm.fields.personalityHint')}
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="language"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>{t('profileForm.fields.language')}</FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value}>
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

                  <FormItem>
                    <FormLabel>{t('profileForm.fields.defaultEngine')}</FormLabel>
                    <Select
                      value={defaultEngine || '_none'}
                      onValueChange={(v) => {
                        setDefaultEngine(v === '_none' ? '' : v);
                      }}
                      disabled={
                        voiceSource === 'builtin' || editingProfile?.voice_type === 'preset'
                      }
                    >
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder={t('profileForm.fields.noPreference')} />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="_none">
                          {t('profileForm.fields.noPreference')}
                        </SelectItem>
                        {availableDefaultEngines.map((option) => (
                          <SelectItem key={option.value} value={option.value}>
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      {t('profileForm.fields.defaultEngineHint')}
                    </p>
                  </FormItem>

                  {editingProfileId && (
                    <div className="space-y-2">
                      <FormLabel>{t('profileForm.fields.defaultEffects')}</FormLabel>
                      <p className="text-xs text-muted-foreground">
                        {t('profileForm.fields.defaultEffectsHint')}
                      </p>
                      <EffectsChainEditor
                        value={profileEffectsChain}
                        onChange={(chain) => {
                          setProfileEffectsChain(chain);
                          setEffectsDirty(true);
                        }}
                        compact
                      />
                    </div>
                  )}
                </div>
              </div>

              <div className="flex gap-2 justify-end mt-6 pt-4 border-t">
                <Button type="button" variant="outline" onClick={() => handleOpenChange(false)}>
                  {t('common.cancel')}
                </Button>
                <Button
                  type="submit"
                  disabled={
                    createProfile.isPending || updateProfile.isPending || addSample.isPending
                  }
                >
                  {createProfile.isPending || updateProfile.isPending || addSample.isPending
                    ? t('profileForm.actions.saving')
                    : editingProfileId
                      ? t('profileForm.actions.saveChanges')
                      : t('profileForm.actions.createProfile')}
                </Button>
              </div>
            </form>
          </Form>
        </div>
      </DialogContent>
    </Dialog>
  );
}
