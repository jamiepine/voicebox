import { http } from 'msw';
import { expect, it } from 'vitest';
import { buildModelStatus, buildProfile } from './msw/fixtures';
import {
  captureHandlers,
  effectsHandlers,
  historyHandlers,
  modelHandlers,
  profileHandlers,
  settingsHandlers,
  storyHandlers,
  taskHandlers,
} from './msw/handlers';
import { worker } from './msw/worker';
import { renderRoute } from './render';
import { sseController } from './sse';

function useHappyPathHandlers() {
  worker.use(
    ...profileHandlers([buildProfile({ name: 'Ada Lovelace' })]),
    ...historyHandlers([]),
    ...captureHandlers([]),
    ...settingsHandlers(),
    ...modelHandlers([buildModelStatus()]),
    ...storyHandlers([]),
    ...effectsHandlers([]),
    ...taskHandlers(),
  );
}

it('renders the /voices route with the full app chrome', async () => {
  useHappyPathHandlers();

  const screen = await renderRoute('/voices');

  await expect.element(screen.getByText('Ada Lovelace')).toBeVisible();
});

it('feeds EventSource through the SSE controller', async () => {
  const sse = sseController();
  worker.use(http.get('*/generate/:id/status', () => sse.response()));

  const source = new EventSource('/generate/gen-1/status');
  const statuses: string[] = [];
  source.onmessage = (message) => {
    statuses.push((JSON.parse(message.data) as { status: string }).status);
  };
  await new Promise((resolve) => {
    source.onopen = resolve;
  });

  sse.push({ data: { status: 'generating' } });
  sse.push({ data: { status: 'completed' } });

  await expect.poll(() => statuses).toEqual(['generating', 'completed']);
  source.close();
  sse.close();
});
