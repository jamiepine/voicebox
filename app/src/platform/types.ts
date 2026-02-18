/**
 * Platform abstraction types
 * These interfaces define the contract that platform implementations must fulfill
 */

export interface FileFilter {
  name: string;
  extensions: string[];
}

export interface PlatformFilesystem {
  saveFile(filename: string, blob: Blob, filters?: FileFilter[]): Promise<void>;
}

export interface UpdateStatus {
  checking: boolean;
  available: boolean;
  version?: string;
  downloading: boolean;
  installing: boolean;
  readyToInstall: boolean;
  error?: string;
  downloadProgress?: number; // 0-100 percentage
  downloadedBytes?: number;
  totalBytes?: number;
}

export interface PlatformUpdater {
  checkForUpdates(): Promise<void>;
  downloadAndInstall(): Promise<void>;
  restartAndInstall(): Promise<void>;
  getStatus(): UpdateStatus;
  subscribe(callback: (status: UpdateStatus) => void): () => void;
}

export interface AudioDevice {
  id: string;
  name: string;
  is_default: boolean;
}

export interface PlatformAudio {
  isSystemAudioSupported(): boolean;
  startSystemAudioCapture(maxDurationSecs: number): Promise<void>;
  stopSystemAudioCapture(): Promise<Blob>;
  listOutputDevices(): Promise<AudioDevice[]>;
  listInputDevices(): Promise<AudioDevice[]>;
  playToDevices(audioData: Uint8Array, deviceIds: string[]): Promise<void>;
  stopPlayback(): void;
}

export interface PlatformLifecycle {
  startServer(remote?: boolean): Promise<string>;
  stopServer(): Promise<void>;
  setKeepServerRunning(keep: boolean): Promise<void>;
  setupWindowCloseHandler(): Promise<void>;
  onServerReady?: () => void;
}

export interface PlatformMetadata {
  getVersion(): Promise<string>;
  /** Short git hash + commit count injected at build time, e.g. "abc1234 #264" */
  getBuildInfo(): string;
  isTauri: boolean;
}

export interface Platform {
  filesystem: PlatformFilesystem;
  updater: PlatformUpdater;
  audio: PlatformAudio;
  lifecycle: PlatformLifecycle;
  metadata: PlatformMetadata;
}
