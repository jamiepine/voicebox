import { useEffect, useState } from 'react';
import { ChevronDown, ChevronUp, Loader2, Mic } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
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
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { getLanguageOptionsForEngine, type LanguageCode } from '@/lib/constants/languages';
import { useGenerationForm } from '@/lib/hooks/useGenerationForm';
import { useProfile } from '@/lib/hooks/useProfiles';
import { useUIStore } from '@/stores/uiStore';
import { EngineModelSelector, applyEngineSelection, getEngineDescription } from './EngineModelSelector';
import { ParalinguisticInput } from './ParalinguisticInput';

function getEngineSelectValue(engine: string): string {
  if (engine === 'qwen') return 'qwen:1.7B';
  if (engine === 'qwen_custom_voice') return 'qwen_custom_voice:1.7B';
  if (engine === 'tada') return 'tada:1B';
  return engine;
}

export function GenerationForm() {
  const selectedProfileId = useUIStore((state) => state.selectedProfileId);
  const { data: selectedProfile } = useProfile(selectedProfileId || '');

  const { form, handleSubmit, isPending } = useGenerationForm();
  const [advancedOpen, setAdvancedOpen] = useState(false);

  useEffect(() => {
    if (!selectedProfile) {
      return;
    }

    if (selectedProfile.language) {
      form.setValue('language', selectedProfile.language as LanguageCode);
    }

    const preferredEngine = selectedProfile.default_engine || selectedProfile.preset_engine;
    if (preferredEngine) {
      applyEngineSelection(form, getEngineSelectValue(preferredEngine));
    }
  }, [form, selectedProfile]);

  async function onSubmit(data: Parameters<typeof handleSubmit>[0]) {
    await handleSubmit(data, selectedProfileId);
  }

  const engine = form.watch('engine');
  const humanizeText = form.watch('humanize_text');
  const showParalinguistic = engine === 'chatterbox_turbo' || engine === 'qwen';
  const showSpeed = engine === 'qwen' || engine === 'qwen_custom_voice';

  return (
    <Card>
      <CardHeader>
        <CardTitle>Generate Speech</CardTitle>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <div>
              <FormLabel>Voice Profile</FormLabel>
              {selectedProfile ? (
                <div className="mt-2 p-3 border rounded-md bg-muted/50 flex items-center gap-2">
                  <Mic className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">{selectedProfile.name}</span>
                  <span className="text-sm text-muted-foreground">{selectedProfile.language}</span>
                </div>
              ) : (
                <div className="mt-2 p-3 border border-dashed rounded-md text-sm text-muted-foreground">
                  Click on a profile card above to select a voice profile
                </div>
              )}
            </div>

            <FormField
              control={form.control}
              name="text"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Text to Speak</FormLabel>
                  <FormControl>
                    {showParalinguistic ? (
                      <ParalinguisticInput
                        value={field.value}
                        onChange={field.onChange}
                        placeholder="Enter text... type / for effects like [laugh], [sigh]"
                        className="min-h-[150px] rounded-md border border-input bg-background px-3 py-2"
                      />
                    ) : (
                      <Textarea
                        placeholder="Enter the text you want to generate..."
                        className="min-h-[150px]"
                        {...field}
                      />
                    )}
                  </FormControl>
                  <FormDescription>
                    {showParalinguistic ? (
                      <>
                        Max 5000 characters. Type / to insert sound effects.
                        {engine === 'qwen' && (
                          <span className="block mt-0.5 text-muted-foreground/70">
                            Tags like [laugh] route to Chatterbox Turbo internally.
                          </span>
                        )}
                      </>
                    ) : (
                      'Max 5000 characters'
                    )}
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            {(engine === 'qwen' || engine === 'qwen_custom_voice') && (
              <FormField
                control={form.control}
                name="instruct"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Delivery Instructions (optional)</FormLabel>
                    <FormControl>
                      <Textarea
                        placeholder="e.g. Speak slowly with emphasis, Warm and friendly tone, Professional and authoritative..."
                        className="min-h-[80px]"
                        {...field}
                      />
                    </FormControl>
                    <FormDescription>
                      Natural language instructions to control speech delivery (tone, emotion,
                      pace). Max 500 characters
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            )}

            <div className="grid gap-4 md:grid-cols-3">
              <FormItem>
                <FormLabel>Model</FormLabel>
                <EngineModelSelector form={form} selectedProfile={selectedProfile} />
                <FormDescription>
                  {getEngineDescription(engine || 'qwen')}
                </FormDescription>
              </FormItem>

              <FormField
                control={form.control}
                name="language"
                render={({ field }) => {
                  const engineLangs = getLanguageOptionsForEngine(engine || 'qwen');
                  return (
                    <FormItem>
                      <FormLabel>Language</FormLabel>
                      <Select onValueChange={field.onChange} value={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {engineLangs.map((lang) => (
                            <SelectItem key={lang.value} value={lang.value}>
                              {lang.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  );
                }}
              />

              <FormField
                control={form.control}
                name="seed"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Seed (optional)</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        placeholder="Random"
                        {...field}
                        onChange={(e) =>
                          field.onChange(e.target.value ? parseInt(e.target.value, 10) : undefined)
                        }
                      />
                    </FormControl>
                    <FormDescription>For reproducible results</FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            {/* Advanced section */}
            <div className="border rounded-md">
              <button
                type="button"
                onClick={() => setAdvancedOpen((o) => !o)}
                className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-left hover:bg-muted/50 transition-colors rounded-md"
              >
                <span>Advanced</span>
                {advancedOpen ? (
                  <ChevronUp className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                )}
              </button>

              {advancedOpen && (
                <div className="px-4 pb-4 space-y-5 border-t">
                  {/* Sampling Parameters */}
                  <div className="space-y-4 pt-4">
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Sampling
                    </p>

                    <FormField
                      control={form.control}
                      name="temperature"
                      render={({ field }) => (
                        <FormItem>
                          <div className="flex items-center justify-between">
                            <FormLabel className="text-sm">Temperature</FormLabel>
                            <span className="text-sm text-muted-foreground tabular-nums">
                              {field.value ?? '—'}
                            </span>
                          </div>
                          <FormControl>
                            <Slider
                              min={0}
                              max={2}
                              step={0.1}
                              value={field.value !== undefined ? [field.value] : []}
                              onValueChange={([v]) => field.onChange(v)}
                            />
                          </FormControl>
                          <FormDescription>0.0 – 2.0 · default ~0.9</FormDescription>
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="top_p"
                      render={({ field }) => (
                        <FormItem>
                          <div className="flex items-center justify-between">
                            <FormLabel className="text-sm">Top-P</FormLabel>
                            <span className="text-sm text-muted-foreground tabular-nums">
                              {field.value ?? '—'}
                            </span>
                          </div>
                          <FormControl>
                            <Slider
                              min={0}
                              max={1}
                              step={0.05}
                              value={field.value !== undefined ? [field.value] : []}
                              onValueChange={([v]) => field.onChange(v)}
                            />
                          </FormControl>
                          <FormDescription>0.0 – 1.0</FormDescription>
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="repetition_penalty"
                      render={({ field }) => (
                        <FormItem>
                          <div className="flex items-center justify-between">
                            <FormLabel className="text-sm">Repetition Penalty</FormLabel>
                            <span className="text-sm text-muted-foreground tabular-nums">
                              {field.value ?? '—'}
                            </span>
                          </div>
                          <FormControl>
                            <Slider
                              min={0.5}
                              max={3}
                              step={0.05}
                              value={field.value !== undefined ? [field.value] : []}
                              onValueChange={([v]) => field.onChange(v)}
                            />
                          </FormControl>
                          <FormDescription>0.5 – 3.0</FormDescription>
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="top_k"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel className="text-sm">Top-K</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              min={0}
                              max={5000}
                              step={1}
                              placeholder="Default"
                              value={field.value ?? ''}
                              onChange={(e) =>
                                field.onChange(
                                  e.target.value ? parseInt(e.target.value, 10) : undefined,
                                )
                              }
                            />
                          </FormControl>
                          <FormDescription>0 – 5000</FormDescription>
                        </FormItem>
                      )}
                    />

                    {showSpeed && (
                      <FormField
                        control={form.control}
                        name="speed"
                        render={({ field }) => (
                          <FormItem>
                            <div className="flex items-center justify-between">
                              <FormLabel className="text-sm">Speed</FormLabel>
                              <span className="text-sm text-muted-foreground tabular-nums">
                                {field.value !== undefined ? `${field.value}×` : '—'}
                              </span>
                            </div>
                            <FormControl>
                              <Slider
                                min={0.5}
                                max={2}
                                step={0.1}
                                value={field.value !== undefined ? [field.value] : []}
                                onValueChange={([v]) => field.onChange(v)}
                              />
                            </FormControl>
                            <FormDescription>0.5× – 2.0×</FormDescription>
                          </FormItem>
                        )}
                      />
                    )}
                  </div>

                  {/* Humanization */}
                  <div className="space-y-4">
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Humanization
                    </p>

                    <FormField
                      control={form.control}
                      name="humanize_text"
                      render={({ field }) => (
                        <FormItem>
                          <div className="flex items-center gap-2">
                            <FormControl>
                              <Checkbox
                                id="humanize_text"
                                checked={!!field.value}
                                onCheckedChange={field.onChange}
                              />
                            </FormControl>
                            <FormLabel htmlFor="humanize_text" className="text-sm font-normal cursor-pointer">
                              Humanize text
                            </FormLabel>
                          </div>
                          <FormDescription>
                            Pre-process text with LLM to add natural speech patterns
                          </FormDescription>
                        </FormItem>
                      )}
                    />

                    {humanizeText && (
                      <FormField
                        control={form.control}
                        name="humanize_intensity"
                        render={({ field }) => (
                          <FormItem className="pl-6">
                            <FormLabel className="text-sm">Intensity</FormLabel>
                            <div className="flex gap-3">
                              {(['light', 'medium', 'heavy'] as const).map((level) => (
                                <label
                                  key={level}
                                  className="flex items-center gap-1.5 cursor-pointer text-sm"
                                >
                                  <input
                                    type="radio"
                                    name="humanize_intensity"
                                    value={level}
                                    checked={field.value === level}
                                    onChange={() => field.onChange(level)}
                                    className="accent-primary"
                                  />
                                  <span className="capitalize">{level}</span>
                                </label>
                              ))}
                            </div>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    )}

                    <FormField
                      control={form.control}
                      name="inject_breaths"
                      render={({ field }) => (
                        <FormItem>
                          <div className="flex items-center gap-2">
                            <FormControl>
                              <Checkbox
                                id="inject_breaths"
                                checked={!!field.value}
                                onCheckedChange={field.onChange}
                              />
                            </FormControl>
                            <FormLabel htmlFor="inject_breaths" className="text-sm font-normal cursor-pointer">
                              Inject breaths
                            </FormLabel>
                          </div>
                          <FormDescription>
                            Insert natural breath sounds between sentences
                          </FormDescription>
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="jitter_ms"
                      render={({ field }) => (
                        <FormItem>
                          <div className="flex items-center justify-between">
                            <FormLabel className="text-sm">Timing jitter</FormLabel>
                            <span className="text-sm text-muted-foreground tabular-nums">
                              {field.value !== undefined ? `${field.value} ms` : '—'}
                            </span>
                          </div>
                          <FormControl>
                            <Slider
                              min={0}
                              max={50}
                              step={1}
                              value={field.value !== undefined ? [field.value] : []}
                              onValueChange={([v]) => field.onChange(v)}
                            />
                          </FormControl>
                          <FormDescription>
                            Random timing offset per chunk (0 – 50 ms)
                          </FormDescription>
                        </FormItem>
                      )}
                    />
                  </div>
                </div>
              )}
            </div>

            <Button type="submit" className="w-full" disabled={isPending || !selectedProfileId}>
              {isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                'Generate Speech'
              )}
            </Button>
          </form>
        </Form>
      </CardContent>
    </Card>
  );
}
