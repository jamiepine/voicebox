import path from 'node:path';
import react from '@vitejs/plugin-react';
import { playwright } from '@vitest/browser-playwright';
import { defineConfig } from 'vitest/config';
import { changelogPlugin } from './app/plugins/changelog';

const appSrc = path.resolve(__dirname, 'app/src');

const shared = {
  plugins: [react(), changelogPlugin(__dirname)],
  resolve: {
    alias: { '@': appSrc },
  },
};

export default defineConfig({
  ...shared,
  test: {
    projects: [
      {
        ...shared,
        test: {
          name: 'unit',
          environment: 'happy-dom',
          include: ['app/src/**/*.test.{ts,tsx}'],
          exclude: ['app/src/**/*.browser.test.{ts,tsx}'],
          setupFiles: ['app/src/test/setup.ts'],
        },
      },
      {
        ...shared,
        publicDir: path.resolve(__dirname, 'app/src/test/public'),
        // Pre-bundle everything the app pulls in so the dep optimizer never
        // reloads mid-run (a cold cache otherwise fails the first CI run).
        optimizeDeps: {
          include: [
            'react',
            'react-dom/client',
            'react/jsx-runtime',
            'react/jsx-dev-runtime',
            '@tanstack/react-query',
            '@tanstack/react-router',
            'zustand',
            'zustand/middleware',
            'i18next',
            'react-i18next',
            'i18next-browser-languagedetector',
            'framer-motion',
            'motion/react',
            'lucide-react',
            'wavesurfer.js',
            '@dnd-kit/core',
            '@dnd-kit/sortable',
            '@dnd-kit/utilities',
            'react-hook-form',
            '@hookform/resolvers/zod',
            'zod',
            'clsx',
            'tailwind-merge',
            'class-variance-authority',
            'date-fns',
            'msw',
            'vitest-browser-react',
          ],
        },
        test: {
          name: 'browser',
          include: ['app/src/**/*.browser.test.{ts,tsx}'],
          setupFiles: ['app/src/test/setup.ts', 'app/src/test/setup.browser.ts'],
          browser: {
            enabled: true,
            headless: true,
            provider: playwright(),
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
});
