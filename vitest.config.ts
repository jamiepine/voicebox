import path from 'node:path';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: {
      // Mirror the Vite alias so imports like "@/i18n" resolve correctly.
      '@': path.resolve(__dirname, 'app/src'),
    },
  },
  test: {
    // Run tests from both app and web workspaces.
    include: ['app/src/**/*.test.ts', 'app/src/**/*.test.tsx', 'web/src/**/*.test.ts'],
    environment: 'node',
    globals: true,
  },
});
