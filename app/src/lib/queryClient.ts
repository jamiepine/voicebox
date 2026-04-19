import { QueryClient } from '@tanstack/react-query';

/**
 * Shared QueryClient instance used across the app.
 *
 * Extracted into its own side-effect-free module so it can be imported from
 * both the React bootstrap (main.tsx) and non-React code (stores, utilities)
 * without pulling in ReactDOM or other bootstrap side effects.
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      gcTime: 1000 * 60 * 10, // 10 minutes (formerly cacheTime)
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});
