import { FolderOpen, Languages, Mic, Zap } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Toggle } from '@/components/ui/toggle';
import { useGenerationSettings } from '@/lib/hooks/useSettings';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';
import { SettingRow, SettingSection } from './SettingRow';

export function GenerationPage() {
  const { t } = useTranslation();
  const platform = usePlatform();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const { settings, update } = useGenerationSettings();
  const persistedMaxChunkChars = settings?.max_chunk_chars ?? 800;
  const persistedCrossfadeMs = settings?.crossfade_ms ?? 50;
  const normalizeAudio = settings?.normalize_audio ?? true;
  const autoplayOnGenerate = settings?.autoplay_on_generate ?? true;
  // Slider mirrors persist on commit (pointer-up / keyboard-release) only —
  // onValueChange would fire a PATCH for every pointer-move pixel and round-
  // trip mid-drag failures could leave persisted state out of sync with UI.
  const [maxChunkChars, setMaxChunkChars] = useState(persistedMaxChunkChars);
  const [crossfadeMs, setCrossfadeMs] = useState(persistedCrossfadeMs);
  useEffect(() => setMaxChunkChars(persistedMaxChunkChars), [persistedMaxChunkChars]);
  useEffect(() => setCrossfadeMs(persistedCrossfadeMs), [persistedCrossfadeMs]);
  const [opening, setOpening] = useState(false);
  const [generationsPath, setGenerationsPath] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${serverUrl}/health/filesystem`)
      .then((res) => res.json())
      .then((data) => {
        const genDir = data.directories?.find((d: { path: string }) =>
          d.path.includes('generations'),
        );
        if (genDir?.path) setGenerationsPath(genDir.path);
      })
      .catch(() => {});
  }, [serverUrl]);

  const openGenerationsFolder = useCallback(async () => {
    if (!generationsPath) return;
    setOpening(true);
    try {
      await platform.filesystem.openPath(generationsPath);
    } catch (e) {
      console.error('Failed to open generations folder:', e);
    } finally {
      setOpening(false);
    }
  }, [platform, generationsPath]);

  return (
    <div className="flex gap-8 items-start max-w-5xl">
      <div className="flex-1 min-w-0 max-w-2xl space-y-8">
      <SettingSection
        title={t('settings.generation.title')}
        description={t('settings.generation.description')}
      >
        <SettingRow
          title={t('settings.generation.chunkLimit.title')}
          description={t('settings.generation.chunkLimit.description')}
          action={
            <span className="text-sm tabular-nums text-muted-foreground">
              {t('settings.generation.chunkLimit.value', { chars: maxChunkChars })}
            </span>
          }
        >
          <Slider
            id="maxChunkChars"
            value={[maxChunkChars]}
            onValueChange={([value]) => setMaxChunkChars(value)}
            onValueCommit={([value]) => update({ max_chunk_chars: value })}
            min={100}
            max={5000}
            step={50}
            aria-label={t('settings.generation.chunkLimit.title')}
          />
        </SettingRow>

        <SettingRow
          title={t('settings.generation.crossfade.title')}
          description={t('settings.generation.crossfade.description')}
          action={
            <span className="text-sm tabular-nums text-muted-foreground">
              {crossfadeMs === 0
                ? t('settings.generation.crossfade.cut')
                : t('settings.generation.crossfade.ms', { ms: crossfadeMs })}
            </span>
          }
        >
          <Slider
            id="crossfadeMs"
            value={[crossfadeMs]}
            onValueChange={([value]) => setCrossfadeMs(value)}
            onValueCommit={([value]) => update({ crossfade_ms: value })}
            min={0}
            max={200}
            step={10}
            aria-label={t('settings.generation.crossfade.title')}
          />
        </SettingRow>

        <SettingRow
          title={t('settings.generation.normalize.title')}
          description={t('settings.generation.normalize.description')}
          htmlFor="normalizeAudio"
          action={
            <Toggle
              id="normalizeAudio"
              checked={normalizeAudio}
              onCheckedChange={(v) => update({ normalize_audio: v })}
            />
          }
        />

        <SettingRow
          title={t('settings.generation.autoplay.title')}
          description={t('settings.generation.autoplay.description')}
          htmlFor="autoplayOnGenerate"
          action={
            <Toggle
              id="autoplayOnGenerate"
              checked={autoplayOnGenerate}
              onCheckedChange={(v) => update({ autoplay_on_generate: v })}
            />
          }
        />

        <SettingRow
          title={t('settings.generation.folder.title')}
          description={generationsPath ?? t('settings.generation.folder.description')}
          action={
            <Button
              variant="outline"
              size="sm"
              onClick={openGenerationsFolder}
              disabled={opening || !generationsPath}
            >
              <FolderOpen className="h-3.5 w-3.5 mr-1.5" />
              {t('settings.generation.folder.open')}
            </Button>
          }
        />
      </SettingSection>
      </div>

      <aside className="hidden lg:block w-[280px] shrink-0 space-y-6 sticky top-0">
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">{t('settings.generation.sidebar.aboutTitle')}</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {t('settings.generation.sidebar.aboutBody')}
          </p>
        </div>

        <div className="space-y-3">
          <h3 className="text-sm font-semibold">{t('settings.generation.sidebar.differencesTitle')}</h3>
          <ul className="space-y-3 text-sm text-muted-foreground">
            <li className="flex gap-2.5">
              <Mic className="h-4 w-4 shrink-0 mt-0.5 text-accent" />
              <span className="leading-relaxed">
                <span className="text-foreground font-medium">
                  {t('settings.generation.sidebar.clone.title')}
                </span>{' '}
                {t('settings.generation.sidebar.clone.body')}
              </span>
            </li>
            <li className="flex gap-2.5">
              <Languages className="h-4 w-4 shrink-0 mt-0.5 text-accent" />
              <span className="leading-relaxed">
                <span className="text-foreground font-medium">
                  {t('settings.generation.sidebar.engines.title')}
                </span>{' '}
                {t('settings.generation.sidebar.engines.body')}
              </span>
            </li>
            <li className="flex gap-2.5">
              <Zap className="h-4 w-4 shrink-0 mt-0.5 text-accent" />
              <span className="leading-relaxed">
                <span className="text-foreground font-medium">{t('settings.generation.sidebar.agentReady.title')}</span>{' '}
                {t('settings.generation.sidebar.agentReady.body')}
              </span>
            </li>
          </ul>
        </div>
      </aside>
    </div>
  );
}
