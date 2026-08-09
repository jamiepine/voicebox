import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createMemoryHistory, createRouter, RouterProvider } from '@tanstack/react-router';
import type { ReactNode } from 'react';
import { render } from 'vitest-browser-react';
import { PlatformProvider } from '@/platform/PlatformContext';
import { routeTree } from '@/router';
import { createMockPlatform, type MockPlatform } from './mockPlatform';

export function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      // Retries and interval refetching are disabled so tests are
      // deterministic — polling components get their data exactly once.
      queries: {
        retry: false,
        refetchInterval: false,
        refetchOnWindowFocus: false,
        gcTime: Number.POSITIVE_INFINITY,
      },
      mutations: { retry: false },
    },
  });
}

export interface RenderWithProvidersOptions {
  platform?: MockPlatform;
  queryClient?: QueryClient;
}

// Every client handed to a render is drained on teardown so in-flight
// queries can't fire after MSW handlers reset (noisy unhandled-request
// errors between tests).
const activeQueryClients: QueryClient[] = [];

export async function drainQueryClients(): Promise<void> {
  for (const client of activeQueryClients) {
    await client.cancelQueries();
    client.clear();
  }
  activeQueryClients.length = 0;
}

export async function renderWithProviders(ui: ReactNode, options: RenderWithProvidersOptions = {}) {
  const platform = options.platform ?? createMockPlatform();
  const queryClient = options.queryClient ?? createTestQueryClient();
  activeQueryClients.push(queryClient);

  const result = await render(
    <QueryClientProvider client={queryClient}>
      <PlatformProvider platform={platform}>{ui}</PlatformProvider>
    </QueryClientProvider>,
  );

  // Object.assign keeps the render result's prototype methods (locators)
  // intact — spreading would drop them.
  return Object.assign(result, { platform, queryClient });
}

/**
 * Mount the real route tree at `route` over memory history — full app chrome
 * (sidebar, frame, toasts) included. A throwaway router per call keeps route
 * state from leaking between tests.
 */
export async function renderRoute(route: string, options: RenderWithProvidersOptions = {}) {
  const router = createRouter({
    routeTree,
    history: createMemoryHistory({ initialEntries: [route] }),
  });
  const result = await renderWithProviders(<RouterProvider router={router} />, options);
  return Object.assign(result, { router });
}
