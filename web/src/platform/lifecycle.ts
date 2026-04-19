import type { PlatformLifecycle, ServerLogEntry } from '@/platform/types';

class WebLifecycle implements PlatformLifecycle {
  onServerReady?: () => void;

  async startServer(_remote = false, _modelsDir?: string | null): Promise<string> {
    // Web assumes server is running externally
    // Return a default URL - this should be configured via env vars
    const serverUrl = import.meta.env.VITE_SERVER_URL || 'http://localhost:17493';
    this.onServerReady?.();
    return serverUrl;
  }

  async stopServer(): Promise<void> {
    // No-op for web - server is managed externally
  }

  async restartServer(_modelsDir?: string | null): Promise<string> {
    // No-op for web - server is managed externally
    return import.meta.env.VITE_SERVER_URL || 'http://localhost:17493';
  }

  async setKeepServerRunning(_keep: boolean): Promise<void> {
    // No-op for web
  }

  async setupWindowCloseHandler(): Promise<void> {
    // No-op for web - no window close handling needed
  }

  subscribeToServerLogs(_callback: (_entry: ServerLogEntry) => void): () => void {
    // No-op for web - server logs are not available
    return () => {};
  }
}

export const webLifecycle = new WebLifecycle();
