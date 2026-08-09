import type { HttpHandler } from 'msw';
import { HttpResponse, http } from 'msw';
import type {
  CaptureResponse,
  CaptureSettings,
  EffectPresetResponse,
  GenerationSettings,
  HistoryResponse,
  ModelStatus,
  StoryDetailResponse,
  StoryResponse,
  VoiceProfileResponse,
} from '@/lib/api/types';
import { buildCaptureReadiness, buildCaptureSettings, buildGenerationSettings } from '../fixtures';

/**
 * Happy-path handlers for one domain each. Tests compose what they need:
 *   worker.use(...profileHandlers([buildProfile()]), ...historyHandlers([]))
 * Anything not stubbed fails loudly via onUnhandledRequest: 'error'.
 */

export function profileHandlers(profiles: VoiceProfileResponse[]): HttpHandler[] {
  return [
    http.get('*/profiles', () => HttpResponse.json(profiles)),
    http.get('*/profiles/presets/:engine', () => HttpResponse.json([])),
    http.get('*/profiles/:id', ({ params }) => {
      const profile = profiles.find((p) => p.id === params.id);
      return profile ? HttpResponse.json(profile) : new HttpResponse(null, { status: 404 });
    }),
    http.get('*/profiles/:id/channels', () => HttpResponse.json([])),
    http.get('*/profiles/:id/samples', () => HttpResponse.json([])),
    http.get('*/channels', () => HttpResponse.json([])),
  ];
}

export function historyHandlers(items: HistoryResponse[]): HttpHandler[] {
  return [
    http.get('*/history', () => HttpResponse.json({ items, total: items.length })),
    http.get('*/history/:id', ({ params }) => {
      const item = items.find((i) => i.id === params.id);
      return item ? HttpResponse.json(item) : new HttpResponse(null, { status: 404 });
    }),
  ];
}

export function captureHandlers(
  items: CaptureResponse[],
  settings: CaptureSettings = buildCaptureSettings(),
): HttpHandler[] {
  return [
    http.get('*/captures', () => HttpResponse.json({ items, total: items.length })),
    http.get('*/capture/readiness', () => HttpResponse.json(buildCaptureReadiness())),
    http.get('*/settings/captures', () => HttpResponse.json(settings)),
    http.put('*/settings/captures', async ({ request }) => {
      const update = (await request.json()) as Partial<CaptureSettings>;
      return HttpResponse.json({ ...settings, ...update });
    }),
  ];
}

export function settingsHandlers(
  generation: GenerationSettings = buildGenerationSettings(),
): HttpHandler[] {
  return [
    http.get('*/settings/generation', () => HttpResponse.json(generation)),
    http.put('*/settings/generation', async ({ request }) => {
      const update = (await request.json()) as Partial<GenerationSettings>;
      return HttpResponse.json({ ...generation, ...update });
    }),
  ];
}

export function modelHandlers(models: ModelStatus[]): HttpHandler[] {
  return [
    http.get('*/models/status', () => HttpResponse.json({ models })),
    http.get('*/models/cache-dir', () => HttpResponse.json({ cache_dir: '/tmp/models' })),
  ];
}

export function storyHandlers(
  stories: StoryResponse[],
  details: StoryDetailResponse[] = [],
): HttpHandler[] {
  return [
    http.get('*/stories', () => HttpResponse.json(stories)),
    http.get('*/stories/:id', ({ params }) => {
      const detail = details.find((d) => d.id === params.id);
      return detail ? HttpResponse.json(detail) : new HttpResponse(null, { status: 404 });
    }),
  ];
}

export function effectsHandlers(presets: EffectPresetResponse[]): HttpHandler[] {
  return [
    http.get('*/effects/available', () => HttpResponse.json({ effects: [] })),
    http.get('*/effects/presets', () => HttpResponse.json(presets)),
  ];
}

export function taskHandlers(): HttpHandler[] {
  return [http.get('*/tasks/active', () => HttpResponse.json({ downloads: [], generations: [] }))];
}
