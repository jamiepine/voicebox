import { HttpResponse, http } from 'msw';
import { expect, it } from 'vitest';
import type { VoiceProfileResponse } from '@/lib/api/types';
import { useGenerationStore } from '@/stores/generationStore';
import { useUIStore } from '@/stores/uiStore';
import { buildGeneration, buildModelStatus, buildProfile } from '@/test/msw/fixtures';
import {
  captureHandlers,
  effectsHandlers,
  historyHandlers,
  modelHandlers,
  profileHandlers,
  settingsHandlers,
  storyHandlers,
  taskHandlers,
} from '@/test/msw/handlers';
import { worker } from '@/test/msw/worker';
import { renderRoute } from '@/test/render';
import { sseController } from '@/test/sse';

/**
 * FloatingGenerateBox calls useMatchRoute, so it needs router context; the
 * SSE completion loop (useGenerationProgress) lives in the router's root
 * layout. Mounting the index route exercises the real wiring for both.
 * History handlers are registered per test so requests can be counted.
 */
function stubAppRequests(profiles: VoiceProfileResponse[]) {
  worker.use(
    ...profileHandlers(profiles),
    ...captureHandlers([]),
    ...settingsHandlers(),
    ...modelHandlers([buildModelStatus()]),
    ...storyHandlers([]),
    ...effectsHandlers([]),
    ...taskHandlers(),
  );
}

it('renders the generate box wired to the selected profile', async () => {
  const profile = buildProfile({ name: 'Ada Lovelace' });
  stubAppRequests([profile]);
  worker.use(...historyHandlers([]));
  useUIStore.getState().setSelectedProfileId(profile.id);

  const screen = await renderRoute('/');

  await expect
    .element(screen.getByPlaceholder('Generate speech using Ada Lovelace…'))
    .toBeVisible();
  await expect.element(screen.getByRole('button', { name: 'Generate speech' })).toBeEnabled();
  expect(useUIStore.getState().selectedProfileId).toBe(profile.id);
});

it('posts to /generate on submit and tracks the pending generation', async () => {
  const profile = buildProfile({ name: 'Ada Lovelace' });
  const generation = buildGeneration({
    profile_id: profile.id,
    status: 'generating',
    audio_path: undefined,
  });
  const generateBodies: unknown[] = [];
  const sse = sseController();
  stubAppRequests([profile]);
  worker.use(
    ...historyHandlers([]),
    http.post('*/generate', async ({ request }) => {
      generateBodies.push(await request.json());
      return HttpResponse.json(generation);
    }),
    http.get('*/generate/:id/status', () => sse.response()),
  );
  useUIStore.getState().setSelectedProfileId(profile.id);

  const screen = await renderRoute('/');

  const input = screen.getByPlaceholder('Generate speech using Ada Lovelace…');
  await input.fill('Hello from the browser test');
  await screen.getByRole('button', { name: 'Generate speech' }).click();

  await expect.poll(() => generateBodies.length).toBe(1);
  expect(generateBodies[0]).toMatchObject({
    profile_id: profile.id,
    text: 'Hello from the browser test',
    language: 'en',
    engine: 'qwen',
  });
  await expect
    .poll(() => useGenerationStore.getState().pendingGenerationIds.has(generation.id))
    .toBe(true);
  // The form resets as soon as the request is accepted.
  await expect.element(input).toHaveValue('');
  sse.close();
});

it('clears pending state and refetches history when SSE reports completion', async () => {
  const profile = buildProfile({ name: 'Ada Lovelace' });
  const generation = buildGeneration({
    profile_id: profile.id,
    status: 'generating',
    audio_path: undefined,
  });
  const sse = sseController();
  let sseConnections = 0;
  let historyGets = 0;
  stubAppRequests([profile]);
  worker.use(
    http.get('*/history', () => {
      historyGets += 1;
      return HttpResponse.json({ items: [], total: 0 });
    }),
    http.post('*/generate', () => HttpResponse.json(generation)),
    http.get('*/generate/:id/status', () => {
      sseConnections += 1;
      return sse.response();
    }),
    // Autoplay is off via settingsHandlers, but keep audio stubbed so a
    // completion-triggered player fetch could never fail the run loudly.
    http.get(
      '*/audio/:id',
      () =>
        new HttpResponse(new Blob([new Uint8Array(64)]), {
          headers: { 'Content-Type': 'audio/wav' },
        }),
    ),
  );
  useUIStore.getState().setSelectedProfileId(profile.id);

  const screen = await renderRoute('/');

  await screen.getByPlaceholder('Generate speech using Ada Lovelace…').fill('Progress please');
  await screen.getByRole('button', { name: 'Generate speech' }).click();

  await expect
    .poll(() => useGenerationStore.getState().pendingGenerationIds.has(generation.id))
    .toBe(true);
  await expect.poll(() => sseConnections).toBe(1);
  // Initial mount fetch + post-submit invalidation — wait for both so the
  // final count increase can only come from the SSE completion refetch.
  await expect.poll(() => historyGets).toBe(2);

  sse.push({ data: { id: generation.id, status: 'generating' } });
  sse.push({ data: { id: generation.id, status: 'completed', duration: 1.5 } });

  await expect.poll(() => useGenerationStore.getState().pendingGenerationIds.size).toBe(0);
  await expect.poll(() => historyGets).toBe(3);
  sse.close();
});

it('disables the input and generate button when no profile is selected', async () => {
  stubAppRequests([]);
  worker.use(...historyHandlers([]));

  const screen = await renderRoute('/');

  await expect
    .element(screen.getByRole('button', { name: 'Select a voice profile first' }))
    .toBeDisabled();
  await expect.element(screen.getByPlaceholder('Select a voice profile above…')).toBeDisabled();
});

it('does not post to /generate when the text is empty', async () => {
  const profile = buildProfile({ name: 'Ada Lovelace' });
  let generateCalls = 0;
  stubAppRequests([profile]);
  worker.use(
    ...historyHandlers([]),
    http.post('*/generate', () => {
      generateCalls += 1;
      return HttpResponse.json(buildGeneration());
    }),
  );
  useUIStore.getState().setSelectedProfileId(profile.id);

  const screen = await renderRoute('/');

  const button = screen.getByRole('button', { name: 'Generate speech' });
  await expect.element(button).toBeEnabled();
  await button.click();

  // Validation rejects empty text before any request is made — give a
  // would-be submission ample time to surface, then assert it never did.
  await new Promise((resolve) => setTimeout(resolve, 300));
  expect(generateCalls).toBe(0);
  expect(useGenerationStore.getState().pendingGenerationIds.size).toBe(0);
});
