import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Slider } from '@/components/ui/slider';
import { useServerStore } from '@/stores/serverStore';

export function GenerationSettings() {
  const maxChunkChars = useServerStore((state) => state.maxChunkChars);
  const setMaxChunkChars = useServerStore((state) => state.setMaxChunkChars);
  const crossfadeMs = useServerStore((state) => state.crossfadeMs);
  const setCrossfadeMs = useServerStore((state) => state.setCrossfadeMs);
  const normalizeAudio = useServerStore((state) => state.normalizeAudio);
  const setNormalizeAudio = useServerStore((state) => state.setNormalizeAudio);
  const autoplayOnGenerate = useServerStore((state) => state.autoplayOnGenerate);
  const setAutoplayOnGenerate = useServerStore((state) => state.setAutoplayOnGenerate);

  return (
    <Card role="region" aria-label="Generation Settings" tabIndex={0}>
      <CardHeader>
        <CardTitle>Generation Settings</CardTitle>
        <CardDescription>
          Controls for long text generation. These settings apply to all engines.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label htmlFor="maxChunkChars" className="text-sm font-medium leading-none">
                Auto-chunking limit
              </label>
              <span className="text-sm tabular-nums text-muted-foreground">
                {maxChunkChars} chars
              </span>
            </div>
            <Slider
              id="maxChunkChars"
              value={[maxChunkChars]}
              onValueChange={([value]) => setMaxChunkChars(value)}
              min={100}
              max={5000}
              step={50}
              aria-label="Auto-chunking character limit"
            />
            <p className="text-sm text-muted-foreground">
              Long text is split into chunks at sentence boundaries before generating. Lower values
              can improve quality for long outputs.
            </p>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label htmlFor="crossfadeMs" className="text-sm font-medium leading-none">
                Chunk crossfade
              </label>
              <span className="text-sm tabular-nums text-muted-foreground">
                {crossfadeMs === 0 ? 'Cut' : `${crossfadeMs}ms`}
              </span>
            </div>
            <Slider
              id="crossfadeMs"
              value={[crossfadeMs]}
              onValueChange={([value]) => setCrossfadeMs(value)}
              min={0}
              max={200}
              step={10}
              aria-label="Chunk crossfade duration"
            />
            <p className="text-sm text-muted-foreground">
              Blends audio between chunks to smooth transitions. Set to 0 for a hard cut.
            </p>
          </div>

          <div className="flex items-start gap-3">
            <Checkbox
              id="normalizeAudio"
              checked={normalizeAudio}
              onCheckedChange={setNormalizeAudio}
              className="mt-[6px]"
            />
            <div className="space-y-1">
              <label
                htmlFor="normalizeAudio"
                className="text-sm font-medium leading-none cursor-pointer"
              >
                Normalize audio
              </label>
              <p className="text-sm text-muted-foreground">
                Adjusts output volume to a consistent level across generations.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3">
            <Checkbox
              id="autoplayOnGenerate"
              checked={autoplayOnGenerate}
              onCheckedChange={setAutoplayOnGenerate}
              className="mt-[6px]"
            />
            <div className="space-y-1">
              <label
                htmlFor="autoplayOnGenerate"
                className="text-sm font-medium leading-none cursor-pointer"
              >
                Autoplay on generate
              </label>
              <p className="text-sm text-muted-foreground">
                Automatically play audio when a generation completes.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
