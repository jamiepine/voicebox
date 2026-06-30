import path from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { changelogPlugin } from './plugins/changelog';

export default defineConfig({
  plugins: [tailwindcss(), react(), changelogPlugin(path.resolve(__dirname, '..'))],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
