import { mockIPC } from '@tauri-apps/api/mocks';
import { afterEach, beforeEach, expect, it } from 'vitest';
import App from '@/App';
import { createMockPlatform } from '@/test/mockPlatform';
import { buildModelStatus, buildProfile } from '@/test/msw/fixtures';
import {
  captureHandlers,
  effectsHandlers,
  historyHandlers,
  modelHandlers,
  profileHandlers,
  settingsHandlers,
  storyHandlers,
  taskHandlers,
} from '@/test/msw/handlers';
import { worker } from '@/test/msw/worker';
import { renderWithProviders } from '@/test/render';

const originalUrl = window.location.href;

// useChordSync and the permission gates call the Tauri IPC modules directly,
// outside the Platform abstraction. There is no Tauri runtime in the test
// browser, so `invoke`/`listen` would reject with a TypeError that some
// callers (e.g. useChordSync's `listen('dictate:warm-request')`) never get a
// chance to handle, surfacing as unhandled rejections. mockIPC installs the
// official in-memory IPC shim; `shouldMockEvents` covers listen/emit too.
//
// Reinstalled per test for a fresh listener map, but never cleared: the
// harness unmounts components after this file's afterEach, and those unmount
// cleanups still `unlisten` through the shim. The per-file iframe throws the
// window state away anyway.
beforeEach(() => {
  mockIPC(
    (cmd) => {
      // Permission checks treat the result as a trusted boolean — grant
      // them so no permission banners pop over the UI under test.
      if (cmd.startsWith('check_')) return true;
      return null;
    },
    { shouldMockEvents: true },
  );
});

afterEach(() => {
  window.history.replaceState(null, '', originalUrl);
  delete window.__voiceboxServerStartedByApp;
});

/**
 * Everything the index route (MainEditor + app chrome) fetches on mount.
 * Unstubbed requests fail the test loudly, so this is the full route budget.
 */
function useHappyPathHandlers() {
  worker.use(
    ...profileHandlers([buildProfile({ name: 'Ada Lovelace' })]),
    ...historyHandlers([]),
    ...captureHandlers([]),
    ...settingsHandlers(),
    ...modelHandlers([buildModelStatus()]),
    ...storyHandlers([]),
    ...effectsHandlers([]),
    ...taskHandlers(),
  );
}

// App reads window.location at render time: `?view=dictate` picks the pill
// window, and the router matches the real browser path. Point the URL at the
// state under test before mounting; afterEach restores the runner's URL.
function setAppUrl(path: string) {
  window.history.replaceState(null, '', path);
}

it('skips the startup gate outside Tauri and renders the router', async () => {
  useHappyPathHandlers();
  setAppUrl('/');

  const screen = await renderWithProviders(<App />);

  // Index route is MainEditor — the profile list proves the router mounted.
  await expect.element(screen.getByText('Ada Lovelace')).toBeVisible();

  // Web mode assumes an external server: no lifecycle management at all.
  expect(screen.platform.lifecycle.startServer).not.toHaveBeenCalled();
  expect(screen.platform.lifecycle.setupWindowCloseHandler).not.toHaveBeenCalled();
});

it('skips server auto-start in Tauri dev mode and still reaches the router', async () => {
  // App gates auto-start on `import.meta.env.PROD`, which is false under
  // vitest. The reachable Tauri branch is therefore the dev one: window
  // close handler installed, auto-start skipped, serverReady forced true.
  //
  // The PROD-only branches — `lifecycle.startServer`, the health-check
  // polling fallback, and the startup-error screen with its Retry button —
  // are unreachable here without mocking import.meta.env, so they are
  // intentionally not covered.
  useHappyPathHandlers();
  setAppUrl('/');
  const platform = createMockPlatform({ metadata: { isTauri: true } });

  const screen = await renderWithProviders(<App />, { platform });

  await expect.element(screen.getByText('Ada Lovelace')).toBeVisible();

  expect(platform.lifecycle.startServer).not.toHaveBeenCalled();
  expect(platform.lifecycle.setupWindowCloseHandler).toHaveBeenCalled();
  // Startup syncs the keep-server-running setting into Rust.
  expect(platform.lifecycle.setKeepServerRunning).toHaveBeenCalledWith(expect.any(Boolean));
  // Auto-updater runs its mount check in Tauri.
  expect(platform.updater.checkForUpdates).toHaveBeenCalled();
  // Dev mode records that the app does not own the server process.
  expect(window.__voiceboxServerStartedByApp).toBe(false);
});

it('renders the dictate pill window for ?view=dictate without booting the main app', async () => {
  // No route handlers on purpose: the dictate view must not touch any of the
  // main app's endpoints, and an unhandled request would fail the test.
  setAppUrl('/?view=dictate');

  const screen = await renderWithProviders(<App />);

  // DictateWindow forces the document transparent so the Tauri window takes
  // the pill's shape — the observable signal that it mounted without
  // throwing under the non-Tauri mock platform.
  await expect.poll(() => document.body.style.background).toBe('transparent');

  // The pill starts hidden: the wrapper renders but contains no CapturePill.
  const wrapper = screen.container.firstElementChild as HTMLElement;
  expect(wrapper.className).toContain('h-screen');
  expect(wrapper.childElementCount).toBe(0);

  // The startup gate never ran — no server lifecycle calls from this window.
  expect(screen.platform.lifecycle.startServer).not.toHaveBeenCalled();
  expect(screen.platform.lifecycle.setupWindowCloseHandler).not.toHaveBeenCalled();
});
