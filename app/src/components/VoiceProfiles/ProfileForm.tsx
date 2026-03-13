import { zodResolver } from '@hookform/resolvers/zod';
import { Edit2, Mic, Monitor, Upload, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
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
import { LANGUAGE_CODES, LANGUAGE_OPTIONS, type LanguageCode } from '@/lib/constants/languages';
import { useAudioPlayer } from '@/lib/hooks/useAudioPlayer';
import { useAudioRecording } from '@/lib/hooks/useAudioRecording';
import {
  useAddSample,
  useCreateProfile,
  useDeleteAvatar,
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

const baseProfileSchema = z.object({
  name: z.string().min(1, 'Name is required').max(100),
  description: z.string().max(500).optional(),
  language: z.enum(LANGUAGE_CODES as [LanguageCode, ...LanguageCode[]]),
  sampleFile: z.instanceof(File).optional(),
  referenceText: z.string().max(1000).optional(),
  avatarFile: z.instanceof(File).optional(),
});

const profileSchema = baseProfileSchema.refine(
  (data) => {
    // If sample file is provided, reference text is required
    if (data.sampleFile && (!data.referenceText || data.referenceText.trim().length === 0)) {
      return false;
    }
    return true;
  },
  {
    message: 'Reference text is required when adding a sample',
    path: ['referenceText'],
  },
);

type ProfileFormValues = z.infer<typeof profileSchema>;

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
  const uploadAvatar = useUploadAvatar();
  const deleteAvatar = useDeleteAvatar();
  const transcribe = useTranscription();
  const { toast } = useToast();
  const [sampleMode, setSampleMode] = useState<'upload' | 'record' | 'system'>('record');
  const [audioDuration, setAudioDuration] = useState<number | null>(null);
  const [isValidatingAudio, setIsValidatingAudio] = useState(false);
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
  const avatarInputRef = useRef<HTMLInputElement>(null);
  const { isPlaying, playPause, cleanup: cleanupAudio } = useAudioPlayer();
  const isCreating = !editingProfileId;
  const serverUrl = useServerStore((state) => state.serverUrl);

  const form = useForm<ProfileFormValues>({
    resolver: zodResolver(profileSchema),
    defaultValues: {
      name: '',
      description: '',
      language: 'en',
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
              message: `Audio is too long (${formatAudioDuration(duration)}). Maximum duration is ${formatAudioDuration(MAX_AUDIO_DURATION_SECONDS)}.`,
            });
          } else {
            form.clearErrors('sampleFile');
          }
        })
        .catch((error) => {
          console.error('Failed to get audio duration:', error);
          setAudioDuration(null);
          // For recordings, we auto-stop at max duration, so we can skip validation errors
          const isRecordedFile =
            selectedFile.name.startsWith('recording-') ||
            selectedFile.name.startsWith('system-audio-');
          if (!isRecordedFile) {
            form.setError('sampleFile', {
              type: 'manual',
              message: 'Failed to validate audio file. Please try a different file.',
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
  }, [selectedFile, form]);

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
        title: 'Recording complete',
        description: 'Audio has been recorded successfully.',
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
        title: 'System audio captured',
        description: 'Audio has been captured successfully.',
      });
    },
  });

  // Show recording errors
  useEffect(() => {
    if (recordingError) {
      toast({
        title: 'Recording error',
        description: recordingError,
        variant: 'destructive',
      });
    }
  }, [recordingError, toast]);

  // Show system audio recording errors
  useEffect(() => {
    if (systemRecordingError) {
      toast({
        title: 'System audio capture error',
        description: systemRecordingError,
        variant: 'destructive',
      });
    }
  }, [systemRecordingError, toast]);

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
        sampleFile: undefined,
        referenceText: undefined,
        avatarFile: undefined,
      });
    } else if (profileFormDraft && open) {
      // Restore from draft when opening in create mode
      form.reset({
        name: profileFormDraft.name,
        description: profileFormDraft.description,
        language: profileFormDraft.language as LanguageCode,
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
        sampleFile: undefined,
        referenceText: undefined,
        avatarFile: undefined,
      });
      setSampleMode('record');
      setAvatarPreview(null);
    }
  }, [editingProfile, profileFormDraft, open, form]);

  async function handleTranscribe() {
    const file = form.getValues('sampleFile');
    if (!file) {
      toast({
        title: 'No file selected',
        description: 'Please select an audio file first.',
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
        title: 'Transcription failed',
        description: error instanceof Error ? error.message : 'Failed to transcribe audio',
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
          title: 'Invalid file type',
          description: 'Please select an image file (PNG, JPG, or WebP)',
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
      form.setValue('avatarFile', file);
    }
  }

  async function handleRemoveAvatar() {
    if (editingProfileId && editingProfile?.avatar_path) {
      try {
        await deleteAvatar.mutateAsync(editingProfileId);
        toast({
          title: 'Avatar removed',
          description: 'Avatar image has been removed successfully.',
        });
      } catch (error) {
        toast({
          title: 'Failed to remove avatar',
          description: error instanceof Error ? error.message : 'Unknown error',
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
        // Editing: just update profile
        await updateProfile.mutateAsync({
          profileId: editingProfileId,
          data: {
            name: data.name,
            description: data.description,
            language: data.language,
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
              title: 'Avatar upload failed',
              description:
                avatarError instanceof Error ? avatarError.message : 'Failed to upload avatar',
              variant: 'destructive',
            });
          }
        }

        toast({
          title: 'Voice updated',
          description: `"${data.name}" has been updated successfully.`,
        });
      } else {
        // Creating: require sample file and reference text
        const sampleFile = form.getValues('sampleFile');
        const referenceText = form.getValues('referenceText');

        if (!sampleFile) {
          form.setError('sampleFile', {
            type: 'manual',
            message: 'Audio sample is required',
          });
          toast({
            title: 'Audio sample required',
            description: 'Please provide an audio sample to create the voice profile.',
            variant: 'destructive',
          });
          return;
        }

        if (!referenceText || referenceText.trim().length === 0) {
          form.setError('referenceText', {
            type: 'manual',
            message: 'Reference text is required',
          });
          toast({
            title: 'Reference text required',
            description: 'Please provide the reference text for the audio sample.',
            variant: 'destructive',
          });
          return;
        }

        // Validate audio duration before creating profile
        try {
          const duration = await getAudioDuration(sampleFile);
          if (duration > MAX_AUDIO_DURATION_SECONDS) {
            form.setError('sampleFile', {
              type: 'manual',
              message: `Audio is too long (${formatAudioDuration(duration)}). Maximum duration is ${formatAudioDuration(MAX_AUDIO_DURATION_SECONDS)}.`,
            });
            toast({
              title: 'Invalid audio file',
              description: `Audio duration is ${formatAudioDuration(duration)}, but maximum is ${formatAudioDuration(MAX_AUDIO_DURATION_SECONDS)}.`,
              variant: 'destructive',
            });
            return; // Prevent form submission
          }
        } catch (error) {
          form.setError('sampleFile', {
            type: 'manual',
            message: 'Failed to validate audio file. Please try a different file.',
          });
          toast({
            title: 'Validation error',
            description: error instanceof Error ? error.message : 'Failed to validate audio file',
            variant: 'destructive',
          });
          return; // Prevent form submission
        }

        // Creating: create profile, then add sample
        const profile = await createProfile.mutateAsync({
          name: data.name,
          description: data.description,
          language: data.language,
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
            title: 'Profile created',
            description: `"${data.name}" has been created with a sample.`,
          });
        } catch (sampleError) {
          // Profile was created but sample failed - still show error
          toast({
            title: 'Failed to add sample',
            description: `Profile "${data.name}" was created, but failed to add sample: ${sampleError instanceof Error ? sampleError.message : 'Unknown error'}`,
            variant: 'destructive',
          });
        }
      }

      // Clear draft and reset form on success
      setProfileFormDraft(null);
      form.reset();
      setEditingProfileId(null);
      setOpen(false);
    } catch (error) {
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to save profile',
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
      <DialogContent className="max-w-none w-screen h-screen left-0 top-0 translate-x-0 translate-y-0 rounded-none p-6 overflow-y-auto">
        <div className="max-w-5xl max-h-[85vh] mx-auto my-auto w-full flex flex-col">
          <DialogHeader>
            <DialogTitle className="text-2xl">
              {editingProfileId ? 'Edit Voice' : 'Clone voice'}
            </DialogTitle>
            <DialogDescription>
              {editingProfileId
                ? 'Update your voice profile details and manage samples.'
                : 'Create a new voice profile with an audio sample to clone the voice.'}
            </DialogDescription>
            {isCreating && profileFormDraft && (
              <div className="flex items-center gap-2 pt-2">
                <span className="text-xs text-muted-foreground">Draft restored</span>
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
                      sampleFile: undefined,
                      referenceText: '',
                    });
                    setSampleMode('record');
                  }}
                >
                  <X className="h-3 w-3 mr-1" />
                  Discard
                </Button>
              </div>
            )}
          </DialogHeader>

          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="flex-1 min-h-0 flex flex-col">
              <div className="grid gap-6 grid-cols-2 flex-1 overflow-y-auto min-h-0">
                {/* Left column: Sample management */}
                <div className="space-y-4 border-r pr-6">
                  {isCreating ? (
                    <>
                      <Tabs
                        className="pt-4"
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
                            Upload
                          </TabsTrigger>
                          <TabsTrigger value="record" className="flex items-center gap-2">
                            <Mic className="h-4 w-4 shrink-0" />
                            Record
                          </TabsTrigger>
                          {platform.metadata.isTauri && isSystemAudioSupported && (
                            <TabsTrigger value="system" className="flex items-center gap-2">
                              <Monitor className="h-4 w-4 shrink-0" />
                              System Audio
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
                            <FormLabel>Reference Text</FormLabel>
                            <FormControl>
                              <Textarea
                                placeholder="Enter the exact text spoken in the audio..."
                                className="min-h-[100px]"
                                {...field}
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    </>
                  ) : (
                    // Show sample list when editing
                    editingProfileId && (
                      <div>
                        <SampleList profileId={editingProfileId} />
                      </div>
                    )
                  )}
                </div>

                {/* Right column: Profile info */}
                <div className="space-y-4">
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
                                    alt="Avatar preview"
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
                        <FormLabel>Description (Optional)</FormLabel>
                        <FormControl>
                          <Textarea placeholder="Describe this voice..." {...field} />
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
                </div>
              </div>

              <div className="flex gap-2 justify-end mt-6 pt-4 border-t">
                <Button type="button" variant="outline" onClick={() => handleOpenChange(false)}>
                  Cancel
                </Button>
                <Button
                  type="submit"
                  disabled={
                    createProfile.isPending || updateProfile.isPending || addSample.isPending
                  }
                >
                  {createProfile.isPending || updateProfile.isPending || addSample.isPending
                    ? 'Saving...'
                    : editingProfileId
                      ? 'Save Changes'
                      : 'Create Profile'}
                </Button>
              </div>
            </form>
          </Form>
        </div>
      </DialogContent>
    </Dialog>
  );
}
