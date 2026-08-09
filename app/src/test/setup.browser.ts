import { afterEach, beforeAll } from 'vitest';
import { cleanup } from 'vitest-browser-react';
import { worker } from './msw/worker';
import { drainQueryClients } from './render';

beforeAll(async () => {
  await worker.start({ onUnhandledRequest: 'error', quiet: true });
  return () => worker.stop();
});

// Registered after setup.ts, so this runs first (afterEach is LIFO):
// unmount → cancel in-flight queries → reset handlers, then setup.ts
// restores stores and mocks.
afterEach(async () => {
  await cleanup();
  await drainQueryClients();
  worker.resetHandlers();
});
