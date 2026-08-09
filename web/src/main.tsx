import { QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '../../app/src/App';
import '../../app/src/index.css';
import '../../app/src/i18n';
import { queryClient } from '../../app/src/lib/queryClient';
import { PlatformProvider } from '../../app/src/platform/PlatformContext';
import { webPlatform } from './platform';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <PlatformProvider platform={webPlatform}>
        <App />
      </PlatformProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
