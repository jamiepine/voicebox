import fs from 'node:fs';
import path from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { changelogPlugin } from '../app/plugins/changelog';

function resolveWorkspaceDependency(name: string) {
  const isolatedPath = path.resolve(__dirname, '../app/node_modules', name);
  return fs.existsSync(isolatedPath)
    ? isolatedPath
    : path.resolve(__dirname, '../node_modules', name);
}

export default defineConfig({
  plugins: [react(), tailwindcss(), changelogPlugin(path.resolve(__dirname, '..'))],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '../app/src'),
      react: resolveWorkspaceDependency('react'),
      'react-dom': resolveWorkspaceDependency('react-dom'),
      '@tanstack/react-query': resolveWorkspaceDependency('@tanstack/react-query'),
      '@tanstack/react-query-devtools': resolveWorkspaceDependency(
        '@tanstack/react-query-devtools',
      ),
      zustand: resolveWorkspaceDependency('zustand'),
    },
    dedupe: ['react', 'react-dom', '@tanstack/react-query', 'zustand'],
  },
  root: path.resolve(__dirname),
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
    // Watch files in the app directory for changes
    watch: {
      ignored: ['!**/../app/**', '**/target/**'],
    },
  },
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    target: 'es2021',
    minify: !process.env.TAURI_DEBUG,
    sourcemap: !!process.env.TAURI_DEBUG,
    outDir: 'dist',
  },
});
