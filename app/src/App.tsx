import { useEffect, useState } from 'react';
import voiceboxLogo from '@/assets/voicebox-logo.png';
import { AudioPlayer } from '@/components/AudioPlayer/AudioPlayer';
import { GenerationForm } from '@/components/Generation/GenerationForm';
import { HistoryTable } from '@/components/History/HistoryTable';
import { ConnectionForm } from '@/components/ServerSettings/ConnectionForm';
import { ModelManagement } from '@/components/ServerSettings/ModelManagement';
import { ServerStatus } from '@/components/ServerSettings/ServerStatus';
import { UpdateStatus } from '@/components/ServerSettings/UpdateStatus';
import ShinyText from '@/components/ShinyText';
import { Sidebar } from '@/components/Sidebar';
import { TitleBarDragRegion } from '@/components/TitleBarDragRegion';
import { Toaster } from '@/components/ui/toaster';
import { ProfileList } from '@/components/VoiceProfiles/ProfileList';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';
import { MODEL_DISPLAY_NAMES, useRestoreActiveTasks } from '@/lib/hooks/useRestoreActiveTasks';
import {
  isMacOS,
  isTauri,
  setKeepServerRunning,
  setupWindowCloseHandler,
  startServer,
} from '@/lib/tauri';
import { useServerStore } from '@/stores/serverStore';

// Track if server is starting to prevent duplicate starts
let serverStarting = false;

const LOADING_MESSAGES = [
  'Warming up tensors...',
  'Calibrating synthesizer engine...',
  'Initializing voice models...',
  'Loading neural networks...',
  'Preparing audio pipelines...',
  'Optimizing waveform generators...',
  'Tuning frequency analyzers...',
  'Building voice embeddings...',
  'Configuring text-to-speech cores...',
  'Syncing audio buffers...',
  'Establishing model connections...',
  'Preprocessing training data...',
  'Validating voice samples...',
  'Compiling inference engines...',
  'Mapping phoneme sequences...',
  'Aligning prosody parameters...',
  'Activating speech synthesis...',
  'Fine-tuning acoustic models...',
  'Preparing voice cloning matrices...',
  'Initializing Qwen TTS framework...',
];

function App() {
  const [activeTab, setActiveTab] = useState('main');
  const [serverReady, setServerReady] = useState(false);
  const [loadingMessageIndex, setLoadingMessageIndex] = useState(0);

  // Monitor active downloads/generations and show toasts for them
  const activeDownloads = useRestoreActiveTasks();

  // Sync stored setting to Rust on startup
  useEffect(() => {
    if (isTauri()) {
      const keepRunning = useServerStore.getState().keepServerRunningOnClose;
      setKeepServerRunning(keepRunning).catch((error) => {
        console.error('Failed to sync initial setting to Rust:', error);
      });
    }
  }, []);

  // Setup window close handler and auto-start server when running in Tauri (production only)
  useEffect(() => {
    if (!isTauri()) {
      return;
    }

    // Setup window close handler to check setting and stop server if needed
    // This works in both dev and prod, but will only stop server if it was started by the app
    setupWindowCloseHandler().catch((error) => {
      console.error('Failed to setup window close handler:', error);
    });

    // Only auto-start server in production mode
    // In dev mode, user runs server separately
    if (!import.meta.env?.PROD) {
      console.log('Dev mode: Skipping auto-start of server (run it separately)');
      setServerReady(true); // Mark as ready so UI doesn't show loading screen
      // Mark that server was not started by app (so we don't try to stop it on close)
      // @ts-expect-error - adding property to window
      window.__voiceboxServerStartedByApp = false;
      return;
    }

    // Auto-start server in production
    if (serverStarting) {
      return;
    }

    serverStarting = true;
    console.log('Production mode: Starting bundled server...');

    startServer(false)
      .then((serverUrl) => {
        console.log('Server is ready at:', serverUrl);
        // Update the server URL in the store with the dynamically assigned port
        useServerStore.getState().setServerUrl(serverUrl);
        setServerReady(true);
        // Mark that we started the server (so we know to stop it on close)
        // @ts-expect-error - adding property to window
        window.__voiceboxServerStartedByApp = true;
      })
      .catch((error) => {
        console.error('Failed to auto-start server:', error);
        serverStarting = false;
        // @ts-expect-error - adding property to window
        window.__voiceboxServerStartedByApp = false;
      });

    // Cleanup: stop server on actual unmount (not StrictMode remount)
    // Note: Window close is handled separately in Tauri Rust code
    return () => {
      // Window close event handles server shutdown based on setting
      serverStarting = false;
    };
  }, []);

  // Cycle through loading messages every 3 seconds
  useEffect(() => {
    if (!isTauri() || serverReady) {
      return;
    }

    const interval = setInterval(() => {
      setLoadingMessageIndex((prev) => (prev + 1) % LOADING_MESSAGES.length);
    }, 3000);

    return () => clearInterval(interval);
  }, [serverReady]);

  // Show loading screen while server is starting in Tauri
  if (isTauri() && !serverReady) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center pt-12">
        <TitleBarDragRegion />
        <div className="text-center space-y-6">
          <div className="flex justify-center relative">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-48 h-48 rounded-full bg-accent/20 blur-3xl" />
            </div>
            <img
              src={voiceboxLogo}
              alt="Voicebox"
              className="w-48 h-48 object-contain animate-fade-in-scale relative z-10"
            />
          </div>
          <div className="animate-fade-in-delayed">
            <ShinyText
              text={LOADING_MESSAGES[loadingMessageIndex]}
              className="text-lg font-medium text-muted-foreground"
              speed={2}
              color="hsl(var(--muted-foreground))"
              shineColor="hsl(var(--foreground))"
            />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-background flex flex-col overflow-hidden pt-12">
      <TitleBarDragRegion />
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <Sidebar activeTab={activeTab} onTabChange={setActiveTab} isMacOS={isMacOS()} />

        <main className="flex-1 ml-20 overflow-hidden flex flex-col">
          <div className="container mx-auto px-8 max-w-[1800px] h-full overflow-hidden flex flex-col">
            {activeTab === 'settings' ? (
              <div className="space-y-4 overflow-y-auto flex flex-col">
                <div className="grid gap-4 md:grid-cols-2">
                  <ConnectionForm />
                  <ServerStatus />
                </div>
                {isTauri() && <UpdateStatus />}
                <ModelManagement />
                <div className="py-8 text-center text-sm text-muted-foreground">
                  Created by{' '}
                  <a
                    href="https://github.com/jamiepine"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent hover:underline"
                  >
                    Jamie Pine
                  </a>
                </div>
              </div>
            ) : (
              // Main view: Profiles top left, Generator bottom left, History right
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full min-h-0 overflow-hidden">
                {/* Left Column */}
                <div className="flex flex-col gap-6 min-h-0 overflow-y-auto pb-32">
                  {/* Profiles - Top Left */}
                  <div className="shrink-0 flex flex-col">
                    <ProfileList />
                  </div>

                  {/* Generator - Bottom Left */}
                  <div className="shrink-0">
                    <GenerationForm />
                  </div>
                </div>

                {/* Right Column - History */}
                <div className="flex flex-col min-h-0 overflow-hidden">
                  <HistoryTable />
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Audio Player - always visible except on settings */}
      {activeTab !== 'settings' && <AudioPlayer />}

      {/* Show download toasts for any active downloads (from anywhere) */}
      {activeDownloads.map((download) => {
        const displayName = MODEL_DISPLAY_NAMES[download.model_name] || download.model_name;
        return (
          <DownloadToastRestorer
            key={download.model_name}
            modelName={download.model_name}
            displayName={displayName}
          />
        );
      })}

      <Toaster />
    </div>
  );
}

/**
 * Component that restores a download toast for a specific model.
 */
function DownloadToastRestorer({
  modelName,
  displayName,
}: {
  modelName: string;
  displayName: string;
}) {
  // Use the download toast hook to restore the toast
  useModelDownloadToast({
    modelName,
    displayName,
    enabled: true,
  });

  return null;
}

export default App;
