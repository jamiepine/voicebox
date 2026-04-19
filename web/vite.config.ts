import path from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { changelogPlugin } from '../app/plugins/changelog';

export default defineConfig({
  plugins: [react(), tailwindcss(), changelogPlugin(path.resolve(__dirname, '..'))],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '../app/src'),
    },
  },
  build: {
    outDir: 'dist',
  },
});
