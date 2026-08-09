import { vi } from 'vitest';
import type { Platform, UpdateStatus } from '@/platform/types';

export interface MockPlatform extends Platform {
  /** Push a new updater status to all subscribers, as the real updater would. */
  emitUpdateStatus(status: UpdateStatus): void;
}

export interface MockPlatformOverrides {
  filesystem?: Partial<Platform['filesystem']>;
  updater?: Partial<Platform['updater']>;
  audio?: Partial<Platform['audio']>;
  lifecycle?: Partial<Platform['lifecycle']>;
  metadata?: Partial<Platform['metadata']>;
}

const INITIAL_UPDATE_STATUS: UpdateStatus = {
  checking: false,
  available: false,
  downloading: false,
  installing: false,
  readyToInstall: false,
};

export const TEST_SERVER_URL = 'http://127.0.0.1:17493';

/**
 * A fully spy-able Platform. Every method is a vi.fn with a benign default
 * (browser-like: no system audio, isTauri false), so tests can assert calls
 * or override behavior per section via `overrides`.
 */
export function createMockPlatform(overrides: MockPlatformOverrides = {}): MockPlatform {
  let updateStatus = { ...INITIAL_UPDATE_STATUS };
  const subscribers = new Set<(status: UpdateStatus) => void>();

  return {
    filesystem: {
      saveFile: vi.fn(async (filename: string) => filename),
      openPath: vi.fn(async () => {}),
      pickDirectory: vi.fn(async () => null),
      ...overrides.filesystem,
    },
    updater: {
      checkForUpdates: vi.fn(async () => {}),
      downloadAndInstall: vi.fn(async () => {}),
      restartAndInstall: vi.fn(async () => {}),
      getStatus: vi.fn(() => ({ ...updateStatus })),
      subscribe: vi.fn((callback: (status: UpdateStatus) => void) => {
        subscribers.add(callback);
        callback(updateStatus);
        return () => {
          subscribers.delete(callback);
        };
      }),
      ...overrides.updater,
    },
    audio: {
      isSystemAudioSupported: vi.fn(async () => false),
      startSystemAudioCapture: vi.fn(async () => {}),
      stopSystemAudioCapture: vi.fn(async () => new Blob()),
      listOutputDevices: vi.fn(async () => []),
      playToDevices: vi.fn(async () => {}),
      stopPlayback: vi.fn(),
      ...overrides.audio,
    },
    lifecycle: {
      startServer: vi.fn(async () => TEST_SERVER_URL),
      stopServer: vi.fn(async () => {}),
      restartServer: vi.fn(async () => TEST_SERVER_URL),
      setKeepServerRunning: vi.fn(async () => {}),
      setBackendOverride: vi.fn(async () => {}),
      setupWindowCloseHandler: vi.fn(async () => {}),
      subscribeToServerLogs: vi.fn(() => () => {}),
      ...overrides.lifecycle,
    },
    metadata: {
      getVersion: vi.fn(async () => '0.0.0-test'),
      isTauri: false,
      ...overrides.metadata,
    },
    emitUpdateStatus(status: UpdateStatus) {
      updateStatus = { ...status };
      for (const callback of subscribers) callback(updateStatus);
    },
  };
}
