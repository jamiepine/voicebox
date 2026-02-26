import { useEffect } from 'react';
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
import { LANGUAGE_OPTIONS, type LanguageCode } from '@/lib/constants/languages';
import { useGenerationForm } from '@/lib/hooks/useGenerationForm';
import { useProfile } from '@/lib/hooks/useProfiles';
import { useUIStore } from '@/stores/uiStore';

export function GenerationForm() {
  const selectedProfileId = useUIStore((state) => state.selectedProfileId);
  const { data: selectedProfile } = useProfile(selectedProfileId || '');

  const { form, handleSubmit, isPending } = useGenerationForm();

  // Auto-sync language when selected profile changes
  useEffect(() => {
    if (selectedProfile?.language) {
      const profileLang = selectedProfile.language as LanguageCode;
      if (LANGUAGE_OPTIONS.some((l) => l.value === profileLang)) {
        form.setValue('language', profileLang);
      }
    }
  }, [selectedProfile?.language, form]);

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
                    Natural language instructions to control speech delivery (tone, emotion, pace).
                    Max 500 characters
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="grid gap-4 md:grid-cols-3">
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

              <FormField
                control={form.control}
                name="modelSize"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Model Size</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="1.7B">Qwen TTS 1.7B (Higher Quality)</SelectItem>
                        <SelectItem value="0.6B">Qwen TTS 0.6B (Faster)</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormDescription>Larger models produce better quality</FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
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

            <Button
              type="submit"
              className="w-full"
              disabled={isPending || !selectedProfileId}
            >
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
