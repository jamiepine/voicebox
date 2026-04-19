import { useQuery, useQueryClient } from '@tanstack/react-query';
import { AlertCircle, Cpu, Download, Loader2, RotateCw, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { apiClient } from '@/lib/api/client';
import type { CudaDownloadProgress, HealthResponse } from '@/lib/api/types';
import { useServerHealth } from '@/lib/hooks/useServer';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';
import { SettingRow, SettingSection } from './SettingRow';

type RestartPhase = 'idle' | 'stopping' | 'waiting' | 'ready';

function AppleLogo({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.8-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83M13 3.5c.73-.83 1.94-1.46 2.94-1.5.13 1.17-.34 2.35-1.04 3.19-.69.85-1.83 1.51-2.95 1.42-.15-1.15.41-2.35 1.05-3.11z" />
    </svg>
  );
}

function GpuIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="4" y="6" width="16" height="12" rx="2" />
      <path d="M2 10h2M2 14h2M20 10h2M20 14h2" />
      <path d="M9 10h6M9 14h4" />
    </svg>
  );
}

function GpuInfoCard({ health }: { health: HealthResponse }) {
  const hasGpu = health.gpu_available && health.gpu_type;

  // Parse GPU name from type string like "CUDA (NVIDIA RTX 4090)" or "MPS (Apple M2 Pro)"
  const gpuName = hasGpu
    ? health.gpu_type!.replace(/^(CUDA|ROCm|MPS|Metal|XPU|DirectML)\s*\((.+)\)$/, '$2') ||
      health.gpu_type!
    : null;
  const gpuBackend = hasGpu ? health.gpu_type!.replace(/\s*\(.+\)$/, '') : null;
  const isApple = gpuBackend === 'MPS' || gpuBackend === 'Metal';
  const showBackendVariant = health.backend_variant && health.backend_variant !== 'cpu';

  return (
    <div className="rounded-lg border border-border/60 p-4">
      <div className="flex items-center gap-3">
        {hasGpu ? (
          isApple ? (
            <AppleLogo className="h-5 w-5 shrink-0 text-muted-foreground" />
          ) : (
            <GpuIcon className="h-5 w-5 shrink-0 text-accent" />
          )
        ) : (
          <Cpu className="h-5 w-5 shrink-0 text-muted-foreground" />
        )}
        <div className="flex-1 min-w-0 space-y-0.5">
          <div className="text-sm font-medium">{hasGpu ? gpuName : 'CPU Only'}</div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            {hasGpu ? (
              <>
                <span>{gpuBackend}</span>
                {showBackendVariant && (
                  <>
                    <span className="text-border">|</span>
                    <span className="uppercase">{health.backend_variant}</span>
                  </>
                )}
                {health.vram_used_mb != null && health.vram_used_mb > 0 && (
                  <>
                    <span className="text-border">|</span>
                    <span>{health.vram_used_mb.toFixed(0)} MB VRAM</span>
                  </>
                )}
              </>
            ) : (
              <span>No GPU acceleration detected</span>
            )}
          </div>
        </div>
        {hasGpu && (
          <div className="flex items-center gap-2 rounded-full border border-accent/30 px-2.5 py-0.5">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent/60" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-accent shadow-[0_0_4px_1px_hsl(var(--accent)/0.4)]" />
            </span>
            <span className="text-[10px] font-medium text-muted-foreground">Active</span>
          </div>
        )}
      </div>
    </div>
  );
}

export function GpuPage() {
  const platform = usePlatform();
  const queryClient = useQueryClient();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const { data: health } = useServerHealth();

  const [restartPhase, setRestartPhase] = useState<RestartPhase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<CudaDownloadProgress | null>(null);
  const healthPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const {
    data: cudaStatus,
    isLoading: _cudaStatusLoading,
    refetch: refetchCudaStatus,
  } = useQuery({
    queryKey: ['cuda-status', serverUrl],
    queryFn: () => apiClient.getCudaStatus(),
    refetchInterval: (query) => (query.state.status === 'pending' ? false : 10000),
    retry: 1,
    enabled: !!health,
  });

  const isCurrentlyCuda = health?.backend_variant === 'cuda';
  const cudaAvailable = cudaStatus?.available ?? false;
  const cudaDownloading = cudaStatus?.downloading ?? false;

  useEffect(() => {
    return () => {
      if (healthPollRef.current) {
        clearInterval(healthPollRef.current);
        healthPollRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!cudaDownloading || !serverUrl) return;

    const eventSource = new EventSource(`${serverUrl}/backend/cuda-progress`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as CudaDownloadProgress;
        setDownloadProgress(data);

        if (data.status === 'complete') {
          eventSource.close();
          setDownloadProgress(null);
          refetchCudaStatus();
        } else if (data.status === 'error') {
          eventSource.close();
          setError(data.error || 'Download failed');
          setDownloadProgress(null);
          refetchCudaStatus();
        }
      } catch (e) {
        console.error('Error parsing CUDA progress event:', e);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [cudaDownloading, serverUrl, refetchCudaStatus]);

  const clearHealthPolling = useCallback(() => {
    if (healthPollRef.current) {
      clearInterval(healthPollRef.current);
      healthPollRef.current = null;
    }
  }, []);

  const startHealthPolling = useCallback(() => {
    clearHealthPolling();

    healthPollRef.current = setInterval(async () => {
      try {
        const result = await apiClient.getHealth();
        if (result.status === 'healthy') {
          clearHealthPolling();
          setRestartPhase('ready');
          queryClient.invalidateQueries();
          setTimeout(() => setRestartPhase('idle'), 2000);
        }
      } catch {
        // Server still down, keep polling
      }
    }, 1000);
  }, [queryClient, clearHealthPolling]);

  const restartServerWithPolling = useCallback(
    async (errorMessage: string) => {
      setRestartPhase('stopping');
      try {
        await platform.lifecycle.restartServer();
        setRestartPhase('waiting');
        startHealthPolling();
      } catch (e: unknown) {
        clearHealthPolling();
        setRestartPhase('idle');
        throw new Error(e instanceof Error ? e.message : errorMessage);
      }
    },
    [platform, startHealthPolling, clearHealthPolling],
  );

  const handleDownload = async () => {
    setError(null);
    try {
      await apiClient.downloadCudaBackend();
      refetchCudaStatus();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to start download';
      if (msg.includes('already downloaded')) {
        refetchCudaStatus();
      } else {
        setError(msg);
      }
    }
  };

  const handleRestart = async () => {
    setError(null);
    try {
      await restartServerWithPolling('Restart failed');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Restart failed');
    }
  };

  const handleSwitchToCpu = async () => {
    setError(null);
    setRestartPhase('stopping');
    try {
      await apiClient.deleteCudaBackend();
      await restartServerWithPolling('Failed to switch to CPU');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to switch to CPU');
      refetchCudaStatus();
    }
  };

  const handleDelete = async () => {
    setError(null);
    try {
      await apiClient.deleteCudaBackend();
      refetchCudaStatus();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to delete CUDA backend');
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / k ** i).toFixed(1)} ${sizes[i]}`;
  };

  if (!health) return null;

  const hasNativeGpu =
    health.gpu_available &&
    !isCurrentlyCuda &&
    health.gpu_type &&
    !health.gpu_type.includes('CUDA');

  return (
    <div className="space-y-8 max-w-2xl">
      <GpuInfoCard health={health} />

      {/* CUDA section — only when no native GPU and not already on CUDA */}
      {!hasNativeGpu && !isCurrentlyCuda && (
        <SettingSection
          title="CUDA Backend"
          description="NVIDIA GPU acceleration via a downloadable CUDA backend."
        >
          {/* Download progress */}
          {cudaDownloading && downloadProgress && (
            <SettingRow title="Downloading CUDA backend...">
              <div className="space-y-1.5">
                <Progress value={downloadProgress.progress} className="h-2" />
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>
                    {downloadProgress.filename ||
                      (cudaAvailable ? 'Updating...' : 'Downloading...')}
                  </span>
                  <span>
                    {downloadProgress.total > 0
                      ? `${formatBytes(downloadProgress.current)} / ${formatBytes(downloadProgress.total)}`
                      : `${downloadProgress.progress.toFixed(1)}%`}
                  </span>
                </div>
              </div>
            </SettingRow>
          )}

          {/* Restart in progress */}
          {restartPhase !== 'idle' && (
            <SettingRow
              title={
                restartPhase === 'ready'
                  ? 'Server restarted successfully'
                  : restartPhase === 'waiting'
                    ? 'Restarting server...'
                    : 'Stopping server...'
              }
              action={<Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
            />
          )}

          {/* Error */}
          {error && (
            <SettingRow title="Error">
              <div className="flex items-center gap-2 text-sm text-destructive">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            </SettingRow>
          )}

          {/* Actions */}
          {restartPhase === 'idle' && !cudaDownloading && (
            <>
              {!cudaAvailable && !isCurrentlyCuda && (
                <SettingRow
                  title="Download CUDA backend"
                  description="~2.4 GB download. Requires an NVIDIA GPU with CUDA support."
                  action={
                    <Button onClick={handleDownload} size="sm">
                      <Download className="h-3.5 w-3.5 mr-1.5" />
                      Download
                    </Button>
                  }
                />
              )}

              {cudaAvailable && !isCurrentlyCuda && platform.metadata.isTauri && (
                <SettingRow
                  title="Switch to CUDA backend"
                  description="CUDA backend is downloaded and ready. Restart to enable."
                  action={
                    <Button onClick={handleRestart} size="sm">
                      <RotateCw className="h-3.5 w-3.5 mr-1.5" />
                      Restart
                    </Button>
                  }
                />
              )}

              {isCurrentlyCuda && platform.metadata.isTauri && (
                <SettingRow
                  title="Switch to CPU backend"
                  description="Disable GPU acceleration. You can re-download CUDA later."
                  action={
                    <Button onClick={handleSwitchToCpu} variant="outline" size="sm">
                      <RotateCw className="h-3.5 w-3.5 mr-1.5" />
                      Switch
                    </Button>
                  }
                />
              )}

              {cudaAvailable && !isCurrentlyCuda && (
                <SettingRow
                  title="Remove CUDA backend"
                  description="Delete the downloaded CUDA binary to free disk space."
                  action={
                    <Button
                      onClick={handleDelete}
                      variant="ghost"
                      size="sm"
                      className="text-muted-foreground hover:text-destructive"
                    >
                      <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                      Remove
                    </Button>
                  }
                />
              )}
            </>
          )}
        </SettingSection>
      )}

      <p className="text-xs text-muted-foreground/60 leading-relaxed">
        Voicebox automatically detects and uses the best available GPU on your system. On Apple
        Silicon Macs, the MLX backend runs natively on the Neural Engine and GPU via Metal
        Performance Shaders (MPS), with no additional setup required. On Windows and Linux with
        NVIDIA GPUs, you can download an optional CUDA backend for hardware-accelerated inference.
        AMD ROCm, Intel XPU, and DirectML are also supported where available through PyTorch. When
        no GPU is detected, Voicebox falls back to CPU — all engines still work, just slower.
      </p>
    </div>
  );
}
