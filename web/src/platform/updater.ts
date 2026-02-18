import type { PlatformUpdater, UpdateStatus } from '@/platform/types';

class WebUpdater implements PlatformUpdater {
  private status: UpdateStatus = {
    checking: false,
    available: false,
    downloading: false,
    installing: false,
    readyToInstall: false,
  };

  private subscribers: Set<(status: UpdateStatus) => void> = new Set();

  private notifySubscribers() {
    for (const callback of this.subscribers) {
      callback(this.status);
    }
  }

  subscribe(callback: (status: UpdateStatus) => void): () => void {
    this.subscribers.add(callback);
    callback(this.status);
    return () => {
      this.subscribers.delete(callback);
    };
  }

  getStatus(): UpdateStatus {
    return { ...this.status };
  }

  async checkForUpdates(): Promise<void> {
    // Web apps don't need client-side updates
    // Updates are handled by redeploying the web app
  }

  async downloadAndInstall(): Promise<void> {
    // No-op for web
  }

  async restartAndInstall(): Promise<void> {
    // No-op for web
  }
}

export const webUpdater = new WebUpdater();
