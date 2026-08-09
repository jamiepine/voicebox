import { defineConfig, devices } from '@playwright/test';

/**
 * E2E suite driving the web build (same app as Tauri, browser platform)
 * against a real CPU backend with the fake TTS engine. Each worker gets
 * its own uvicorn on its own port with its own data dir — see fixtures.ts.
 *
 * PW_DEV=1 targets `bun run dev:web` (port 5173) instead of the preview
 * build for faster local iteration.
 */
const DEV = !!process.env.PW_DEV;
const PORT = DEV ? 5173 : 4173;

export default defineConfig({
  testDir: './specs',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['html', { open: 'never' }], ['github']] : 'list',
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: DEV
      ? 'bun run dev:web'
      : 'bun run build:web && cd web && bunx vite preview --port 4173 --strictPort',
    cwd: '..',
    port: PORT,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
