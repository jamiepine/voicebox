import { Loader2, Mic } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
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
import { Textarea } from '@/components/ui/textarea';
import { getLanguageOptionsForEngine } from '@/lib/constants/languages';
import { useGenerationForm } from '@/lib/hooks/useGenerationForm';
import { useProfile } from '@/lib/hooks/useProfiles';
import { useUIStore } from '@/stores/uiStore';

export function GenerationForm() {
  const selectedProfileId = useUIStore((state) => state.selectedProfileId);
  const { data: selectedProfile } = useProfile(selectedProfileId || '');

  const { form, handleSubmit, isPending } = useGenerationForm();

  async function onSubmit(data: Parameters<typeof handleSubmit>[0]) {
    await handleSubmit(data, selectedProfileId);
  }

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
                    <Textarea
                      placeholder="Enter the text you want to generate..."
                      className="min-h-[150px]"
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>Max 5000 characters</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            {form.watch('engine') === 'qwen' && (
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
                <Select
                  value={
                    form.watch('engine') === 'luxtts'
                      ? 'luxtts'
                      : form.watch('engine') === 'chatterbox'
                        ? 'chatterbox'
                        : form.watch('engine') === 'chatterbox_turbo'
                          ? 'chatterbox_turbo'
                          : `qwen:${form.watch('modelSize') || '1.7B'}`
                  }
                  onValueChange={(value) => {
                    if (value === 'luxtts') {
                      form.setValue('engine', 'luxtts');
                      form.setValue('language', 'en');
                    } else if (value === 'chatterbox') {
                      form.setValue('engine', 'chatterbox');
                    } else if (value === 'chatterbox_turbo') {
                      form.setValue('engine', 'chatterbox_turbo');
                      form.setValue('language', 'en');
                    } else {
                      const [, modelSize] = value.split(':');
                      form.setValue('engine', 'qwen');
                      form.setValue('modelSize', modelSize as '1.7B' | '0.6B');
                    }
                  }}
                >
                  <FormControl>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    <SelectItem value="qwen:1.7B">Qwen3-TTS 1.7B</SelectItem>
                    <SelectItem value="qwen:0.6B">Qwen3-TTS 0.6B</SelectItem>
                    <SelectItem value="luxtts">LuxTTS</SelectItem>
                    <SelectItem value="chatterbox">Chatterbox</SelectItem>
                    <SelectItem value="chatterbox_turbo">Chatterbox Turbo</SelectItem>
                  </SelectContent>
                </Select>
                <FormDescription>
                  {form.watch('engine') === 'luxtts'
                    ? 'Fast, English-focused'
                    : form.watch('engine') === 'chatterbox'
                      ? '23 languages, incl. Hebrew'
                      : form.watch('engine') === 'chatterbox_turbo'
                        ? 'English, [laugh] [cough] tags'
                        : 'Multi-language, two sizes'}
                </FormDescription>
              </FormItem>

              <FormField
                control={form.control}
                name="language"
                render={({ field }) => {
                  const engineLangs = getLanguageOptionsForEngine(form.watch('engine') || 'qwen');
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
