import { type ChildProcess, spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test as base } from '@playwright/test';

const REPO_ROOT = path.resolve(__dirname, '..');
const BASE_PORT = 18100;

export interface BackendFixture {
  url: string;
  dataDir: string;
}

async function waitForHealth(url: string, timeoutMs = 60_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${url}/health`);
      if (res.ok) return;
    } catch {
      // not up yet
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`backend at ${url} did not become healthy within ${timeoutMs}ms`);
}

export const test = base.extend<Record<never, never>, { backend: BackendFixture }>({
  backend: [
    async ({}, use, workerInfo) => {
      const port = BASE_PORT + workerInfo.workerIndex;
      const url = `http://127.0.0.1:${port}`;
      // uvicorn resolves the data dir from cwd, so a temp cwd isolates
      // each worker's SQLite and audio files completely.
      const dataDir = mkdtempSync(path.join(tmpdir(), `voicebox-e2e-${workerInfo.workerIndex}-`));
      const python = process.env.VOICEBOX_PYTHON ?? path.join(REPO_ROOT, '.venv-ci/bin/python');

      const proc: ChildProcess = spawn(
        python,
        ['-m', 'uvicorn', 'backend.main:app', '--port', String(port), '--log-level', 'warning'],
        {
          cwd: dataDir,
          env: {
            ...process.env,
            PYTHONPATH: REPO_ROOT,
            VOICEBOX_FAKE_TTS: '1',
            VOICEBOX_CORS_ORIGINS:
              'http://localhost:4173,http://127.0.0.1:4173,http://localhost:5173',
          },
          stdio: ['ignore', 'pipe', 'pipe'],
        },
      );
      const logs: Buffer[] = [];
      proc.stdout?.on('data', (chunk) => logs.push(chunk));
      proc.stderr?.on('data', (chunk) => logs.push(chunk));

      try {
        await waitForHealth(url);
      } catch (error) {
        proc.kill('SIGKILL');
        throw new Error(`${(error as Error).message}\nbackend log:\n${Buffer.concat(logs)}`);
      }

      await use({ url, dataDir });

      proc.kill('SIGTERM');
      await new Promise((resolve) => {
        proc.once('exit', resolve);
        setTimeout(resolve, 5000);
      });
      rmSync(dataDir, { recursive: true, force: true });
    },
    { scope: 'worker' },
  ],

  // Point the app's persisted server store at this worker's backend before
  // any page script runs.
  page: async ({ page, backend }, use) => {
    await page.addInitScript((serverUrl) => {
      window.localStorage.setItem(
        'voicebox-server',
        JSON.stringify({
          state: {
            serverUrl,
            isConnected: false,
            mode: 'local',
            keepServerRunningOnClose: false,
            customModelsDir: null,
          },
          version: 0,
        }),
      );
    }, backend.url);
    await use(page);
  },
});

export { expect } from '@playwright/test';
