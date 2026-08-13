/**
 * Web platform implementation
 *
 * Voicebox's primary target is the Tauri desktop app; the web build
 * (app/ served as a static SPA by the Python backend) had no platform
 * implementation after the Tauri refactor, which crashed the UI with
 * "usePlatform must be used within PlatformProvider".
 *
 * This implementation provides browser-safe no-ops for the desktop-only
 * features (updater, system audio capture, server lifecycle) and a
 * download-based saveFile so audio export keeps working in the browser.
 */

import type {
  Platform,
  PlatformAudio,
  PlatformFilesystem,
  PlatformLifecycle,
  PlatformMetadata,
  PlatformUpdater,
  UpdateStatus,
} from './types';

const noop = () => {};

const EMPTY_UPDATE_STATUS: UpdateStatus = {
  checking: false,
  available: false,
  downloading: false,
  installing: false,
  readyToInstall: false,
};

const webFilesystem: PlatformFilesystem = {
  async saveFile(filename: string, blob: Blob) {
    // Browser download via object URL — file filters are a desktop concept
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  },

  async openPath(path: string) {
    window.open(path, '_blank', 'noopener');
  },

  async pickDirectory() {
    // Browsers can't return a persistent directory path
    return null;
  },
};

const webUpdater: PlatformUpdater = {
  async checkForUpdates() {},
  async downloadAndInstall() {},
  async restartAndInstall() {},
  getStatus: () => ({ ...EMPTY_UPDATE_STATUS }),
  subscribe: () => noop,
};

const webAudio: PlatformAudio = {
  async isSystemAudioSupported() {
    return false;
  },
  async startSystemAudioCapture() {
    throw new Error('System audio capture is not supported in the browser.');
  },
  async stopSystemAudioCapture() {
    throw new Error('System audio capture is not supported in the browser.');
  },
  async listOutputDevices() {
    return [];
  },
  async playToDevices() {
    // Web playback uses the default <audio> element path
  },
  stopPlayback() {},
};

const webLifecycle: PlatformLifecycle = {
  async startServer() {
    throw new Error('Not running in Tauri environment');
  },
  async stopServer() {},
  async restartServer() {
    throw new Error('Not running in Tauri environment');
  },
  async setKeepServerRunning() {},
  async setBackendOverride() {},
  async setupWindowCloseHandler() {},
  subscribeToServerLogs: () => noop,
};

const webMetadata: PlatformMetadata = {
  async getVersion() {
    return 'web';
  },
  isTauri: false,
};

export const webPlatform: Platform = {
  filesystem: webFilesystem,
  updater: webUpdater,
  audio: webAudio,
  lifecycle: webLifecycle,
  metadata: webMetadata,
};
