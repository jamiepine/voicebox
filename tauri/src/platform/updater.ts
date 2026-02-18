import { relaunch } from '@tauri-apps/plugin-process';
import { check, type Update } from '@tauri-apps/plugin-updater';
import type { PlatformUpdater, UpdateStatus } from '@/platform/types';

// Check if we're on Windows (NSIS installer handles restart automatically)
const isWindows = () => {
  return navigator.userAgent.includes('Windows');
};

class TauriUpdater implements PlatformUpdater {
  private status: UpdateStatus = {
    checking: false,
    available: false,
    downloading: false,
    installing: false,
    readyToInstall: false,
  };

  private update: Update | null = null;
  private subscribers: Set<(status: UpdateStatus) => void> = new Set();

  private notifySubscribers() {
    for (const callback of this.subscribers) {
      callback(this.status);
    }
  }

  subscribe(callback: (status: UpdateStatus) => void): () => void {
    this.subscribers.add(callback);
    // Immediately call with current status
    callback(this.status);
    return () => {
      this.subscribers.delete(callback);
    };
  }

  getStatus(): UpdateStatus {
    return { ...this.status };
  }

  async checkForUpdates(): Promise<void> {
    try {
      this.status = { ...this.status, checking: true, error: undefined };
      this.notifySubscribers();

      const foundUpdate = await check();

      if (foundUpdate?.available) {
        this.update = foundUpdate;
        this.status = {
          checking: false,
          available: true,
          version: foundUpdate.version,
          downloading: false,
          installing: false,
          readyToInstall: false,
        };
      } else {
        this.status = {
          checking: false,
          available: false,
          downloading: false,
          installing: false,
          readyToInstall: false,
        };
      }
      this.notifySubscribers();
    } catch (error) {
      this.status = {
        checking: false,
        available: false,
        downloading: false,
        installing: false,
        readyToInstall: false,
        error: error instanceof Error ? error.message : 'Failed to check for updates',
      };
      this.notifySubscribers();
    }
  }

  async downloadAndInstall(): Promise<void> {
    if (!this.update) return;

    try {
      this.status = { ...this.status, downloading: true, error: undefined };
      this.notifySubscribers();

      let downloadedBytes = 0;
      let totalBytes = 0;

      await this.update.download((event) => {
        switch (event.event) {
          case 'Started':
            totalBytes = event.data.contentLength || 0;
            downloadedBytes = 0;
            this.status = {
              ...this.status,
              downloading: true,
              totalBytes,
              downloadedBytes: 0,
              downloadProgress: 0,
            };
            this.notifySubscribers();
            break;
          case 'Progress': {
            downloadedBytes += event.data.chunkLength;
            const progress =
              totalBytes > 0 ? Math.round((downloadedBytes / totalBytes) * 100) : undefined;
            this.status = {
              ...this.status,
              downloadedBytes,
              downloadProgress: progress,
            };
            this.notifySubscribers();
            break;
          }
          case 'Finished':
            this.status = {
              ...this.status,
              downloading: false,
              readyToInstall: true,
              downloadProgress: 100,
            };
            this.notifySubscribers();
            break;
        }
      });
    } catch (error) {
      this.status = {
        ...this.status,
        downloading: false,
        installing: false,
        readyToInstall: false,
        downloadProgress: undefined,
        downloadedBytes: undefined,
        totalBytes: undefined,
        error: error instanceof Error ? error.message : 'Failed to download update',
      };
      this.notifySubscribers();
    }
  }

  async restartAndInstall(): Promise<void> {
    if (!this.update) return;

    try {
      this.status = { ...this.status, installing: true, error: undefined };
      this.notifySubscribers();

      await this.update.install();

      // On Windows with NSIS, the installer handles the restart automatically.
      // On macOS/Linux, we need to manually relaunch.
      if (!isWindows()) {
        await relaunch();
      }
    } catch (error) {
      this.status = {
        ...this.status,
        installing: false,
        error: error instanceof Error ? error.message : 'Failed to install update',
      };
      this.notifySubscribers();
    }
  }
}

export const tauriUpdater = new TauriUpdater();
