import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import ReactDOM from 'react-dom/client';
// import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import App from '@/App';
// Import CSS from app directory using alias so Tailwind can scan the source files
import '@/index.css';
import { PlatformProvider } from '@/platform/PlatformContext';
import { tauriPlatform } from './platform';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      gcTime: 1000 * 60 * 10, // 10 minutes
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <PlatformProvider platform={tauriPlatform}>
        <App />
        {/* <ReactQueryDevtools initialIsOpen={false} /> */}
      </PlatformProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
