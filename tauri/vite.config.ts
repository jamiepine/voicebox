import path from 'node:path';
import { execSync } from 'node:child_process';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

function getGitHash(): string {
  try {
    return execSync('git rev-parse --short HEAD', { encoding: 'utf8' }).trim();
  } catch {
    return 'unknown';
  }
}

function getGitCommitCount(): number {
  try {
    return parseInt(execSync('git rev-list --count HEAD', { encoding: 'utf8' }).trim(), 10);
  } catch {
    return 0;
  }
}

export default defineConfig({
  define: {
    __GIT_HASH__: JSON.stringify(getGitHash()),
    __GIT_COMMIT_COUNT__: getGitCommitCount(),
  },
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '../app/src'),
      react: path.resolve(__dirname, '../app/node_modules/react'),
      'react-dom': path.resolve(__dirname, '../app/node_modules/react-dom'),
      '@tanstack/react-query': path.resolve(__dirname, '../app/node_modules/@tanstack/react-query'),
      '@tanstack/react-query-devtools': path.resolve(
        __dirname,
        '../app/node_modules/@tanstack/react-query-devtools',
      ),
      zustand: path.resolve(__dirname, '../app/node_modules/zustand'),
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
      ignored: ['!**/../app/**'],
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
