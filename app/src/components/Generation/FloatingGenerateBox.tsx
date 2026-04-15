import { useQuery } from '@tanstack/react-query';
import { useMatchRoute } from '@tanstack/react-router';
import { AnimatePresence, motion } from 'framer-motion';
import { CheckCircle, Loader2, SlidersHorizontal, Sparkles } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Form, FormControl, FormField, FormItem, FormMessage } from '@/components/ui/form';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { apiClient } from '@/lib/api/client';
import { getLanguageOptionsForEngine, type LanguageCode } from '@/lib/constants/languages';
import { useGenerationForm } from '@/lib/hooks/useGenerationForm';
import { useProfile, useProfiles } from '@/lib/hooks/useProfiles';
import { useStory } from '@/lib/hooks/useStories';
import { cn } from '@/lib/utils/cn';
import { useGenerationStore } from '@/stores/generationStore';
import { useStoryStore } from '@/stores/storyStore';
import { useUIStore } from '@/stores/uiStore';
import { EngineModelSelector } from './EngineModelSelector';
import { ParalinguisticInput } from './ParalinguisticInput';

interface FloatingGenerateBoxProps {
  isPlayerOpen?: boolean;
  showVoiceSelector?: boolean;
}

export function FloatingGenerateBox({
  isPlayerOpen = false,
  showVoiceSelector = false,
}: FloatingGenerateBoxProps) {
  const selectedProfileId = useUIStore((state) => state.selectedProfileId);
  const setSelectedProfileId = useUIStore((state) => state.setSelectedProfileId);
  const setSelectedEngine = useUIStore((state) => state.setSelectedEngine);
  const { data: selectedProfile } = useProfile(selectedProfileId || '');
  const { data: profiles } = useProfiles();
  const [isExpanded, setIsExpanded] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const reuseEffectsChainRef = useRef<import('@/lib/api/types').EffectConfig[] | null>(null);
  const matchRoute = useMatchRoute();
  const isStoriesRoute = matchRoute({ to: '/stories' });
  const selectedStoryId = useStoryStore((state) => state.selectedStoryId);
  const trackEditorHeight = useStoryStore((state) => state.trackEditorHeight);
  const { data: currentStory } = useStory(selectedStoryId);
  const addPendingStoryAdd = useGenerationStore((s) => s.addPendingStoryAdd);
  const reuseParams = useGenerationStore((s) => s.reuseParams);
  const setReuseParams = useGenerationStore((s) => s.setReuseParams);

  // Fetch effect presets for the dropdown
  const { data: effectPresets } = useQuery({
    queryKey: ['effectPresets'],
    queryFn: () => apiClient.listEffectPresets(),
  });

  // Fetch suggested params for the selected profile
  const { data: suggestedParams } = useQuery({
    queryKey: ['suggestedParams', selectedProfileId],
    queryFn: () => apiClient.getSuggestedParams(selectedProfileId!),
    enabled: !!selectedProfileId,
  });

  // Calculate if track editor is visible (on stories route with items)
  const hasTrackEditor = isStoriesRoute && currentStory && currentStory.items.length > 0;

  const { form, handleSubmit, isPending } = useGenerationForm({
    onSuccess: async (generationId) => {
      setIsExpanded(false);
      // Defer the story add until TTS completes -- useGenerationProgress handles it
      if (isStoriesRoute && selectedStoryId && generationId) {
        addPendingStoryAdd(generationId, selectedStoryId);
      }
    },
    getEffectsChain: () => {
      if (!selectedPresetId) return undefined;
      // Profile's own effects chain (no matching preset)
      if (selectedPresetId === '_profile') {
        return selectedProfile?.effects_chain ?? undefined;
      }
      // Effects chain reused from history (no matching preset)
      if (selectedPresetId === '_reuse') {
        return reuseEffectsChainRef.current ?? undefined;
      }
      if (!effectPresets) return undefined;
      const preset = effectPresets.find((p) => p.id === selectedPresetId);
      return preset?.effects_chain;
    },
  });

  // Click away handler to collapse the box
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as HTMLElement;

      // Don't collapse if clicking inside the container
      if (containerRef.current?.contains(target)) {
        return;
      }

      // Don't collapse if clicking on a Select dropdown (which renders in a portal)
      if (
        target.closest('[role="listbox"]') ||
        target.closest('[data-radix-popper-content-wrapper]')
      ) {
        return;
      }

      setIsExpanded(false);
    }

    if (isExpanded) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isExpanded]);

  // Set first voice as default if none selected
  useEffect(() => {
    if (!selectedProfileId && profiles && profiles.length > 0) {
      setSelectedProfileId(profiles[0].id);
    }
  }, [selectedProfileId, profiles, setSelectedProfileId]);

  // Sync engine selection to global store so ProfileList can filter
  const watchedEngine = form.watch('engine');
  useEffect(() => {
    if (watchedEngine) {
      setSelectedEngine(watchedEngine);
    }
  }, [watchedEngine, setSelectedEngine]);

  // Sync generation form language, engine, and effects with selected profile
  useEffect(() => {
    if (selectedProfile?.language) {
      form.setValue('language', selectedProfile.language as LanguageCode);
    }
    // Auto-switch engine if profile has a default
    if (selectedProfile?.default_engine) {
      form.setValue(
        'engine',
        selectedProfile.default_engine as
          | 'qwen'
          | 'luxtts'
          | 'chatterbox'
          | 'chatterbox_turbo'
          | 'tada'
          | 'kokoro',
      );
    }
    // Pre-fill effects from profile defaults
    if (
      selectedProfile?.effects_chain &&
      selectedProfile.effects_chain.length > 0 &&
      effectPresets
    ) {
      // Try to match against a known preset
      const profileChainJson = JSON.stringify(selectedProfile.effects_chain);
      const matchingPreset = effectPresets.find(
        (p) => JSON.stringify(p.effects_chain) === profileChainJson,
      );
      if (matchingPreset) {
        setSelectedPresetId(matchingPreset.id);
      } else {
        // No matching preset — use special value to pass profile chain directly
        setSelectedPresetId('_profile');
      }
    } else if (
      selectedProfile &&
      (!selectedProfile.effects_chain || selectedProfile.effects_chain.length === 0)
    ) {
      setSelectedPresetId(null);
    }
  }, [selectedProfile, effectPresets, form]);

  // Auto-resize textarea based on content (only when expanded)
  useEffect(() => {
    if (!isExpanded) {
      // Reset textarea height after collapse animation completes
      const timeoutId = setTimeout(() => {
        const textarea = textareaRef.current;
        if (textarea) {
          textarea.style.height = '32px';
          textarea.style.overflowY = 'hidden';
        }
      }, 200); // Wait for animation to complete
      return () => clearTimeout(timeoutId);
    }

    const textarea = textareaRef.current;
    if (!textarea) return;

    const adjustHeight = () => {
      textarea.style.height = 'auto';
      const scrollHeight = textarea.scrollHeight;
      const minHeight = 100; // Expanded minimum
      const maxHeight = 300; // Max height in pixels
      const targetHeight = Math.max(minHeight, Math.min(scrollHeight, maxHeight));
      textarea.style.height = `${targetHeight}px`;

      // Show scrollbar if content exceeds max height
      if (scrollHeight > maxHeight) {
        textarea.style.overflowY = 'auto';
      } else {
        textarea.style.overflowY = 'hidden';
      }
    };

    // Small delay to let framer animation complete
    const timeoutId = setTimeout(() => {
      adjustHeight();
    }, 200);

    // Adjust on mount and when value changes
    adjustHeight();

    // Watch for input changes
    textarea.addEventListener('input', adjustHeight);

    return () => {
      clearTimeout(timeoutId);
      textarea.removeEventListener('input', adjustHeight);
    };
  }, [isExpanded]);

  // Apply params from history "Reuse" button
  useEffect(() => {
    if (!reuseParams) return;
    form.setValue('text', reuseParams.text);
    if (reuseParams.language) form.setValue('language', reuseParams.language as LanguageCode);
    if (reuseParams.engine)
      form.setValue(
        'engine',
        reuseParams.engine as
          | 'qwen'
          | 'qwen_custom_voice'
          | 'luxtts'
          | 'chatterbox'
          | 'chatterbox_turbo'
          | 'tada'
          | 'kokoro',
      );
    if (reuseParams.temperature != null) form.setValue('temperature', reuseParams.temperature);
    if (reuseParams.top_k != null) form.setValue('top_k', Math.round(reuseParams.top_k));
    if (reuseParams.top_p != null) form.setValue('top_p', reuseParams.top_p);
    if (reuseParams.repetition_penalty != null)
      form.setValue('repetition_penalty', reuseParams.repetition_penalty);
    if (reuseParams.speed != null) form.setValue('speed', reuseParams.speed);
    // Apply effects chain if present
    if (reuseParams.effects_chain && reuseParams.effects_chain.length > 0) {
      reuseEffectsChainRef.current = reuseParams.effects_chain;
      if (effectPresets) {
        const chainJson = JSON.stringify(reuseParams.effects_chain);
        const matchingPreset = effectPresets.find(
          (p) => JSON.stringify(p.effects_chain) === chainJson,
        );
        if (matchingPreset) {
          setSelectedPresetId(matchingPreset.id);
        } else {
          // No matching preset — use sentinel so getEffectsChain returns the stored chain
          setSelectedPresetId('_reuse');
        }
      } else {
        setSelectedPresetId('_reuse');
      }
    } else {
      reuseEffectsChainRef.current = null;
    }
    setIsExpanded(true);
    // Consume the params so this effect doesn't re-fire
    setReuseParams(null);
  }, [reuseParams]); // eslint-disable-line react-hooks/exhaustive-deps

  function applySuggestedParams() {
    if (!suggestedParams) return;
    if (suggestedParams.temperature != null) form.setValue('temperature', suggestedParams.temperature);
    if (suggestedParams.top_k != null) form.setValue('top_k', Math.round(suggestedParams.top_k));
    if (suggestedParams.top_p != null) form.setValue('top_p', suggestedParams.top_p);
    if (suggestedParams.repetition_penalty != null) form.setValue('repetition_penalty', suggestedParams.repetition_penalty);
    if (suggestedParams.speed != null) form.setValue('speed', suggestedParams.speed);
  }

  async function onSubmit(data: Parameters<typeof handleSubmit>[0]) {
    await handleSubmit(data, selectedProfileId);
  }

  return (
    <motion.div
      ref={containerRef}
      className={cn(
        'fixed right-auto',
        isStoriesRoute
          ? // Position aligned with story list: after sidebar + padding, width 360px
            'left-[calc(5rem+2rem)] w-[360px]'
          : 'left-[calc(5rem+2rem)] right-8 lg:right-auto lg:w-[calc((100%-5rem-4rem)/2-1rem)]',
      )}
      style={{
        // On stories route: offset by track editor height when visible
        // On other routes: offset by audio player height when visible
        bottom: hasTrackEditor
          ? `${trackEditorHeight + 24}px`
          : isPlayerOpen
            ? 'calc(7rem + 1.5rem)'
            : '1.5rem',
      }}
    >
      <motion.div
        className="bg-background/30 backdrop-blur-2xl border border-accent/20 rounded-[2rem] shadow-2xl hover:bg-background/40 hover:border-accent/20 transition-all duration-300 p-3"
        transition={{ duration: 0.6, ease: 'easeInOut' }}
      >
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)}>
            <div className="flex gap-2">
              <motion.div className="flex-1" transition={{ duration: 0.3, ease: 'easeOut' }}>
                <FormField
                  control={form.control}
                  name="text"
                  render={({ field }) => (
                    <FormItem>
                      <FormControl>
                        <motion.div
                          animate={{
                            height: isExpanded ? 'auto' : '32px',
                          }}
                          transition={{ duration: 0.15, ease: 'easeOut' }}
                          style={{ overflow: 'hidden' }}
                        >
                          {(form.watch('engine') === 'chatterbox_turbo' || form.watch('engine') === 'qwen') ? (
                            <ParalinguisticInput
                              value={field.value}
                              onChange={field.onChange}
                              placeholder={
                                isStoriesRoute && currentStory
                                  ? `Generate speech for "${currentStory.name}"... (type / for effects)`
                                  : selectedProfile
                                    ? `Type / for effects like [laugh], [sigh]...`
                                    : 'Select a voice profile above...'
                              }
                              className="px-3 py-2 resize-none bg-transparent border-none focus-visible:ring-0 focus-visible:ring-offset-0 focus:outline-none focus:ring-0 outline-none ring-0 rounded-2xl text-sm w-full"
                              style={{
                                minHeight: isExpanded ? '100px' : '32px',
                                maxHeight: '300px',
                                overflowY: 'auto',
                              }}
                              disabled={!selectedProfileId}
                              onClick={() => setIsExpanded(true)}
                              onFocus={() => setIsExpanded(true)}
                            />
                          ) : (
                            <Textarea
                              {...field}
                              ref={(node: HTMLTextAreaElement | null) => {
                                textareaRef.current = node;
                                if (typeof field.ref === 'function') {
                                  field.ref(node);
                                }
                              }}
                              placeholder={
                                isStoriesRoute && currentStory
                                  ? `Generate speech for "${currentStory.name}"...`
                                  : selectedProfile
                                    ? `Generate speech using ${selectedProfile.name}...`
                                    : 'Select a voice profile above...'
                              }
                              className="resize-none bg-transparent border-none focus-visible:ring-0 focus-visible:ring-offset-0 focus:outline-none focus:ring-0 outline-none ring-0 rounded-2xl text-sm placeholder:text-muted-foreground/60 w-full"
                              style={{
                                minHeight: isExpanded ? '100px' : '32px',
                                maxHeight: '300px',
                              }}
                              disabled={!selectedProfileId}
                              onClick={() => setIsExpanded(true)}
                              onFocus={() => setIsExpanded(true)}
                            />
                          )}
                        </motion.div>
                      </FormControl>
                      <FormMessage className="text-xs" />
                    </FormItem>
                  )}
                />
              </motion.div>

              <div className="relative shrink-0 flex flex-col items-center gap-1">
                {/* Settings / Advanced popover */}
                <Popover open={showAdvanced} onOpenChange={setShowAdvanced}>
                  <PopoverTrigger asChild>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      aria-label="Advanced settings"
                      className={cn(
                        'h-7 w-7 rounded-full transition-all duration-200',
                        showAdvanced
                          ? 'bg-accent/20 text-accent'
                          : 'text-muted-foreground hover:text-accent hover:bg-accent/10',
                      )}
                    >
                      <SlidersHorizontal className="h-3.5 w-3.5" />
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent
                    side="top"
                    align="end"
                    sideOffset={8}
                    className="w-72 space-y-2.5 rounded-2xl border border-accent/20 bg-background/80 backdrop-blur-xl p-4"
                  >
                    <p className="text-xs font-medium text-muted-foreground mb-3">Advanced settings</p>

                    {/* Row 1: Temperature + Speed */}
                    <div className="grid grid-cols-2 gap-3">
                      <FormField
                        control={form.control}
                        name="temperature"
                        render={({ field }) => (
                          <FormItem className="space-y-1">
                            <div className="flex items-center justify-between">
                              <label className="text-xs text-muted-foreground/70">Temp</label>
                              <span className="text-xs text-muted-foreground tabular-nums">
                                {field.value?.toFixed(2) ?? '—'}
                              </span>
                            </div>
                            <FormControl>
                              <Slider
                                min={0}
                                max={2}
                                step={0.05}
                                value={field.value !== undefined ? [field.value] : [1]}
                                onValueChange={([v]) => field.onChange(v)}
                                className="h-3"
                              />
                            </FormControl>
                          </FormItem>
                        )}
                      />
                      <FormField
                        control={form.control}
                        name="speed"
                        render={({ field }) => (
                          <FormItem className="space-y-1">
                            <div className="flex items-center justify-between">
                              <label className="text-xs text-muted-foreground/70">Speed</label>
                              <span className="text-xs text-muted-foreground tabular-nums">
                                {field.value?.toFixed(2) ?? '—'}
                              </span>
                            </div>
                            <FormControl>
                              <Slider
                                min={0.5}
                                max={2}
                                step={0.05}
                                value={field.value !== undefined ? [field.value] : [1]}
                                onValueChange={([v]) => field.onChange(v)}
                                className="h-3"
                              />
                            </FormControl>
                          </FormItem>
                        )}
                      />
                    </div>

                    {/* Row 2: Top-K + Top-P */}
                    <div className="grid grid-cols-2 gap-3">
                      <FormField
                        control={form.control}
                        name="top_k"
                        render={({ field }) => (
                          <FormItem className="space-y-1">
                            <div className="flex items-center justify-between">
                              <label className="text-xs text-muted-foreground/70">Top-K</label>
                              <span className="text-xs text-muted-foreground tabular-nums">
                                {field.value !== undefined ? field.value : '—'}
                              </span>
                            </div>
                            <FormControl>
                              <Slider
                                min={0}
                                max={200}
                                step={1}
                                value={field.value !== undefined ? [field.value] : [50]}
                                onValueChange={([v]) => field.onChange(v)}
                                className="h-3"
                              />
                            </FormControl>
                          </FormItem>
                        )}
                      />
                      <FormField
                        control={form.control}
                        name="top_p"
                        render={({ field }) => (
                          <FormItem className="space-y-1">
                            <div className="flex items-center justify-between">
                              <label className="text-xs text-muted-foreground/70">Top-P</label>
                              <span className="text-xs text-muted-foreground tabular-nums">
                                {field.value?.toFixed(2) ?? '—'}
                              </span>
                            </div>
                            <FormControl>
                              <Slider
                                min={0}
                                max={1}
                                step={0.01}
                                value={field.value !== undefined ? [field.value] : [0.9]}
                                onValueChange={([v]) => field.onChange(v)}
                                className="h-3"
                              />
                            </FormControl>
                          </FormItem>
                        )}
                      />
                    </div>

                    {/* Row 3: Repetition Penalty (half-width, left column) */}
                    <div className="grid grid-cols-2 gap-3">
                      <FormField
                        control={form.control}
                        name="repetition_penalty"
                        render={({ field }) => (
                          <FormItem className="space-y-1">
                            <div className="flex items-center justify-between">
                              <label className="text-xs text-muted-foreground/70">Rep. Penalty</label>
                              <span className="text-xs text-muted-foreground tabular-nums">
                                {field.value?.toFixed(2) ?? '—'}
                              </span>
                            </div>
                            <FormControl>
                              <Slider
                                min={0.5}
                                max={2}
                                step={0.01}
                                value={field.value !== undefined ? [field.value] : [1]}
                                onValueChange={([v]) => field.onChange(v)}
                                className="h-3"
                              />
                            </FormControl>
                          </FormItem>
                        )}
                      />
                    </div>

                    {/* Row 5: Humanize text + intensity */}
                    <div className="flex items-center gap-2">
                      <FormField
                        control={form.control}
                        name="humanize_text"
                        render={({ field }) => (
                          <FormItem className="space-y-0">
                            <FormControl>
                              <div className="flex items-center gap-1.5">
                                <Checkbox
                                  id="humanize_text_adv"
                                  checked={!!field.value}
                                  onCheckedChange={field.onChange}
                                />
                                <label
                                  htmlFor="humanize_text_adv"
                                  className="text-xs text-muted-foreground/70 cursor-pointer select-none"
                                >
                                  Humanize
                                </label>
                              </div>
                            </FormControl>
                          </FormItem>
                        )}
                      />
                      <FormField
                        control={form.control}
                        name="humanize_intensity"
                        render={({ field }) => (
                          <FormItem className="flex-1 space-y-0">
                            <FormControl>
                              <Select
                                value={field.value ?? 'medium'}
                                onValueChange={field.onChange}
                                disabled={!form.watch('humanize_text')}
                              >
                                <SelectTrigger className="h-7 text-xs bg-card border-border rounded-full hover:bg-background/50 transition-all">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="light" className="text-xs">Light</SelectItem>
                                  <SelectItem value="medium" className="text-xs">Medium</SelectItem>
                                  <SelectItem value="heavy" className="text-xs">Heavy</SelectItem>
                                </SelectContent>
                              </Select>
                            </FormControl>
                          </FormItem>
                        )}
                      />
                    </div>

                    {/* Row 6: Inject breaths + Jitter */}
                    <div className="grid grid-cols-2 gap-3 items-center">
                      <FormField
                        control={form.control}
                        name="inject_breaths"
                        render={({ field }) => (
                          <FormItem className="space-y-0">
                            <FormControl>
                              <div className="flex items-center gap-1.5">
                                <Checkbox
                                  id="inject_breaths_adv"
                                  checked={!!field.value}
                                  onCheckedChange={field.onChange}
                                />
                                <label
                                  htmlFor="inject_breaths_adv"
                                  className="text-xs text-muted-foreground/70 cursor-pointer select-none"
                                >
                                  Inject breaths
                                </label>
                              </div>
                            </FormControl>
                          </FormItem>
                        )}
                      />
                      <FormField
                        control={form.control}
                        name="jitter_ms"
                        render={({ field }) => (
                          <FormItem className="space-y-1">
                            <div className="flex items-center justify-between">
                              <label className="text-xs text-muted-foreground/70">Jitter</label>
                              <span className="text-xs text-muted-foreground tabular-nums">
                                {field.value !== undefined ? `${field.value}ms` : '—'}
                              </span>
                            </div>
                            <FormControl>
                              <Slider
                                min={0}
                                max={50}
                                step={1}
                                value={field.value !== undefined ? [field.value] : [0]}
                                onValueChange={([v]) => field.onChange(v)}
                                className="h-3"
                              />
                            </FormControl>
                          </FormItem>
                        )}
                      />
                    </div>
                  </PopoverContent>
                </Popover>

                {/* Generate button */}
                <div className="group relative">
                  <Button
                    type="submit"
                    disabled={isPending || !selectedProfileId}
                    className="h-10 w-10 rounded-full bg-accent hover:bg-accent/90 hover:scale-105 text-accent-foreground shadow-lg hover:shadow-accent/50 transition-all duration-200"
                    size="icon"
                    aria-label={
                      isPending
                        ? 'Generating...'
                        : !selectedProfileId
                          ? 'Select a voice profile first'
                          : 'Generate speech'
                    }
                  >
                    {isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Sparkles className="h-4 w-4" />
                    )}
                  </Button>
                  <span className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 whitespace-nowrap rounded-md bg-popover px-3 py-1.5 text-xs text-popover-foreground border border-border opacity-0 transition-opacity group-hover:opacity-100 z-[9999]">
                    {isPending
                      ? 'Generating...'
                      : !selectedProfileId
                        ? 'Select a voice profile first'
                        : 'Generate speech'}
                  </span>
                </div>
              </div>
            </div>

            <AnimatePresence>
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
                className=" mt-3"
              >
                {/* Suggested params banner */}
                {suggestedParams && (
                  <div className="flex items-center gap-2 mb-2 px-1 py-1 rounded-xl bg-green-500/10 border border-green-500/20">
                    <CheckCircle className="h-3 w-3 text-green-500 shrink-0 ml-1" />
                    <span className="text-xs text-green-500 flex-1">
                      {suggestedParams?.n_samples
                        ? `Based on ${suggestedParams.n_samples} rating${suggestedParams.n_samples === 1 ? '' : 's'}`
                        : 'Proven params for this voice'}
                    </span>
                    <button
                      type="button"
                      className="text-xs text-green-500 font-medium hover:text-green-400 transition-colors px-1.5 py-0.5 rounded-lg hover:bg-green-500/10"
                      onClick={applySuggestedParams}
                    >
                      Apply
                    </button>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  {showVoiceSelector && (
                    <div className="flex-1">
                      <Select
                        value={selectedProfileId || ''}
                        onValueChange={(value) => setSelectedProfileId(value || null)}
                      >
                        <SelectTrigger className="h-8 text-xs bg-card border-border rounded-full hover:bg-background/50 transition-all w-full">
                          <SelectValue placeholder="Select a voice..." />
                        </SelectTrigger>
                        <SelectContent>
                          {profiles?.map((profile) => (
                            <SelectItem key={profile.id} value={profile.id} className="text-xs">
                              {profile.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  <FormField
                    control={form.control}
                    name="language"
                    render={({ field }) => {
                      const engineLangs = getLanguageOptionsForEngine(
                        form.watch('engine') || 'qwen',
                      );
                      return (
                        <FormItem className="flex-1 space-y-0">
                          <Select onValueChange={field.onChange} value={field.value}>
                            <FormControl>
                              <SelectTrigger className="h-8 text-xs bg-card border-border rounded-full hover:bg-background/50 transition-all">
                                <SelectValue />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              {engineLangs.map((lang) => (
                                <SelectItem key={lang.value} value={lang.value} className="text-xs">
                                  {lang.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <FormMessage className="text-xs" />
                        </FormItem>
                      );
                    }}
                  />

                  <FormItem className="flex-1 space-y-0">
                    <EngineModelSelector form={form} compact />
                  </FormItem>

                  <FormItem className="flex-1 space-y-0">
                    <Select
                      value={selectedPresetId || 'none'}
                      onValueChange={(value) => {
                        if (value !== '_reuse') reuseEffectsChainRef.current = null;
                        setSelectedPresetId(value === 'none' ? null : value);
                      }}
                    >
                      <SelectTrigger className="h-8 text-xs bg-card border-border rounded-full hover:bg-background/50 transition-all">
                        <SelectValue placeholder="No effects" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none" className="text-xs">
                          No effects
                        </SelectItem>
                        {selectedProfile?.effects_chain &&
                          selectedProfile.effects_chain.length > 0 && (
                            <SelectItem value="_profile" className="text-xs">
                              Profile default
                            </SelectItem>
                          )}
                        {selectedPresetId === '_reuse' && (
                          <SelectItem value="_reuse" className="text-xs">
                            From history
                          </SelectItem>
                        )}
                        {effectPresets?.map((preset) => (
                          <SelectItem key={preset.id} value={preset.id} className="text-xs">
                            {preset.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </FormItem>
                </div>

              </motion.div>
            </AnimatePresence>
          </form>
        </Form>
      </motion.div>
    </motion.div>
  );
}
