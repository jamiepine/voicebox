import { setupWorker } from 'msw/browser';
import { serverHandlers } from './handlers/server';

/**
 * Browser-mode MSW worker. Individual tests layer route-specific handlers
 * on top with `worker.use(...)`; `setup.browser.ts` resets them after each
 * test. Only the health/baseline handlers are registered globally.
 */
export const worker = setupWorker(...serverHandlers);
