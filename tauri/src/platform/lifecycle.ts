import { invoke } from '@tauri-apps/api/core';
import { listen, emit } from '@tauri-apps/api/event';
import type { PlatformLifecycle } from '@/platform/types';

class TauriLifecycle implements PlatformLifecycle {
  onServerReady?: () => void;

  async startServer(remote = false): Promise<string> {
    try {
      const result = await invoke<string>('start_server', { remote });
      console.log('Server started:', result);
      this.onServerReady?.();
      return result;
    } catch (error) {
      console.error('Failed to start server:', error);
      throw error;
    }
  }

  async stopServer(): Promise<void> {
    try {
      await invoke('stop_server');
      console.log('Server stopped');
    } catch (error) {
      console.error('Failed to stop server:', error);
      throw error;
    }
  }

  async setKeepServerRunning(keepRunning: boolean): Promise<void> {
    try {
      await invoke('set_keep_server_running', { keepRunning });
    } catch (error) {
      console.error('Failed to set keep server running setting:', error);
    }
  }

  async setupWindowCloseHandler(): Promise<void> {
    try {
      // Listen for window close request from Rust
      await listen<null>('window-close-requested', async () => {
        // Import store here to avoid circular dependency
        const { useServerStore } = await import('@/stores/serverStore');
        const keepRunning = useServerStore.getState().keepServerRunningOnClose;

        // Check if server was started by this app instance
        const serverStartedByApp = window.__voiceboxServerStartedByApp ?? false;

        if (!keepRunning && serverStartedByApp) {
          // Stop server before closing (only if we started it)
          try {
            await this.stopServer();
          } catch (error) {
            console.error('Failed to stop server on close:', error);
          }
        }

        // Emit event back to Rust to allow close
        await emit('window-close-allowed');
      });
    } catch (error) {
      console.error('Failed to setup window close handler:', error);
    }
  }
}

export const tauriLifecycle = new TauriLifecycle();
