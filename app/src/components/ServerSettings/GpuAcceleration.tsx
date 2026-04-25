import { useQuery, useQueryClient } from '@tanstack/react-query';
import { AlertCircle, Download, Loader2, RotateCw, Trash2 } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { apiClient } from '@/lib/api/client';
import type { CudaDownloadProgress, RocmDownloadProgress } from '@/lib/api/types';
import { useServerHealth } from '@/lib/hooks/useServer';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';

type RestartPhase = 'idle' | 'stopping' | 'waiting' | 'ready';

export function GpuAcceleration() {
  const platform = usePlatform();
  const queryClient = useQueryClient();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const { data: health } = useServerHealth();

  const [restartPhase, setRestartPhase] = useState<RestartPhase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<CudaDownloadProgress | null>(null);
  const [rocmDownloadProgress, setRocmDownloadProgress] = useState<RocmDownloadProgress | null>(
    null,
  );
  const healthPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Query CUDA backend status
  const {
    data: cudaStatus,
    isLoading: _cudaStatusLoading,
    refetch: refetchCudaStatus,
  } = useQuery({
    queryKey: ['cuda-status', serverUrl],
    queryFn: () => apiClient.getCudaStatus(),
    refetchInterval: (query) => (query.state.status === 'pending' ? false : 10000),
    retry: 1,
    enabled: !!health, // Only fetch when backend is reachable
  });

  // Query ROCm backend status
  const {
    data: rocmStatus,
    isLoading: _rocmStatusLoading,
    refetch: refetchRocmStatus,
  } = useQuery({
    queryKey: ['rocm-status', serverUrl],
    queryFn: () => apiClient.getRocmStatus(),
    refetchInterval: (query) => (query.state.status === 'pending' ? false : 10000),
    retry: 1,
    enabled: !!health, // Only fetch when backend is reachable
  });

  // Derived state
  const isCurrentlyCuda = health?.backend_variant === 'cuda';
  const isCurrentlyRocm = health?.backend_variant === 'rocm';
  const cudaAvailable = cudaStatus?.available ?? false;
  const cudaDownloading = cudaStatus?.downloading ?? false;
  const rocmAvailable = rocmStatus?.available ?? false;
  const rocmDownloading = rocmStatus?.downloading ?? false;

  // Clean up health poll on unmount
  useEffect(() => {
    return () => {
      if (healthPollRef.current) {
        clearInterval(healthPollRef.current);
        healthPollRef.current = null;
      }
    };
  }, []);

  // SSE progress tracking during CUDA download
  useEffect(() => {
    if (!cudaDownloading || !serverUrl) {
      return;
    }

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

  // SSE progress tracking during ROCm download
  useEffect(() => {
    if (!rocmDownloading || !serverUrl) {
      return;
    }

    const eventSource = new EventSource(`${serverUrl}/backend/rocm-progress`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as RocmDownloadProgress;
        setRocmDownloadProgress(data);

        if (data.status === 'complete') {
          eventSource.close();
          setRocmDownloadProgress(null);
          refetchRocmStatus();
        } else if (data.status === 'error') {
          eventSource.close();
          setError(data.error || 'Download failed');
          setRocmDownloadProgress(null);
          refetchRocmStatus();
        }
      } catch (e) {
        console.error('Error parsing ROCm progress event:', e);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [rocmDownloading, serverUrl, refetchRocmStatus]);

  // Start aggressive health polling during restart
  const startHealthPolling = useCallback(() => {
    if (healthPollRef.current) return;

    healthPollRef.current = setInterval(async () => {
      try {
        const result = await apiClient.getHealth();
        if (result.status === 'healthy') {
          // Server is back up
          if (healthPollRef.current) {
            clearInterval(healthPollRef.current);
            healthPollRef.current = null;
          }
          setRestartPhase('ready');
          // Invalidate all queries to refresh UI
          queryClient.invalidateQueries();
          // Reset after a moment
          setTimeout(() => setRestartPhase('idle'), 2000);
        }
      } catch {
        // Server still down, keep polling
      }
    }, 1000);
  }, [queryClient]);

  const handleDownloadCuda = async () => {
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

  const handleDownloadRocm = async () => {
    setError(null);
    try {
      await apiClient.downloadRocmBackend();
      refetchRocmStatus();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to start download';
      if (msg.includes('already downloaded')) {
        refetchRocmStatus();
      } else {
        setError(msg);
      }
    }
  };

  const handleRestart = async () => {
    setError(null);
    setRestartPhase('stopping');

    try {
      setRestartPhase('waiting');
      startHealthPolling();
      await platform.lifecycle.restartServer();
      // Invoke resolved — server is likely ready. Stop polling and refresh.
      if (healthPollRef.current) {
        clearInterval(healthPollRef.current);
        healthPollRef.current = null;
      }
      setRestartPhase('ready');
      queryClient.invalidateQueries();
      setTimeout(() => setRestartPhase('idle'), 2000);
    } catch (e: unknown) {
      setRestartPhase('idle');
      if (healthPollRef.current) {
        clearInterval(healthPollRef.current);
        healthPollRef.current = null;
      }
      setError(e instanceof Error ? e.message : 'Restart failed');
    }
  };

  const handleSwitchToCpuFromCuda = async () => {
    setError(null);
    setRestartPhase('stopping');

    try {
      // Tell Rust launcher to skip GPU binary detection on next start.
      // We cannot delete an active .exe on Windows, so we override instead.
      await platform.lifecycle.setBackendOverride('cpu');
      setRestartPhase('waiting');
      startHealthPolling();
      await platform.lifecycle.restartServer();
      if (healthPollRef.current) {
        clearInterval(healthPollRef.current);
        healthPollRef.current = null;
      }
      setRestartPhase('ready');
      queryClient.invalidateQueries();
      setTimeout(() => setRestartPhase('idle'), 2000);
    } catch (e: unknown) {
      setRestartPhase('idle');
      if (healthPollRef.current) {
        clearInterval(healthPollRef.current);
        healthPollRef.current = null;
      }
      setError(e instanceof Error ? e.message : 'Failed to switch to CPU');
      refetchCudaStatus();
    }
  };

  const handleSwitchToCpuFromRocm = async () => {
    setError(null);
    setRestartPhase('stopping');

    try {
      // Tell Rust launcher to skip GPU binary detection on next start.
      // We cannot delete an active .exe on Windows, so we override instead.
      await platform.lifecycle.setBackendOverride('cpu');
      setRestartPhase('waiting');
      startHealthPolling();
      await platform.lifecycle.restartServer();
      if (healthPollRef.current) {
        clearInterval(healthPollRef.current);
        healthPollRef.current = null;
      }
      setRestartPhase('ready');
      queryClient.invalidateQueries();
      setTimeout(() => setRestartPhase('idle'), 2000);
    } catch (e: unknown) {
      setRestartPhase('idle');
      if (healthPollRef.current) {
        clearInterval(healthPollRef.current);
        healthPollRef.current = null;
      }
      setError(e instanceof Error ? e.message : 'Failed to switch to CPU');
      refetchRocmStatus();
    }
  };

  const handleDeleteCuda = async () => {
    setError(null);
    try {
      await apiClient.deleteCudaBackend();
      refetchCudaStatus();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to delete CUDA backend');
    }
  };

  const handleDeleteRocm = async () => {
    setError(null);
    try {
      await apiClient.deleteRocmBackend();
      refetchRocmStatus();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to delete ROCm backend');
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / k ** i).toFixed(1)} ${sizes[i]}`;
  };

  // Don't render until health data is available
  if (!health) return null;

  // If the system already has native GPU (MPS, ROCm active, etc.), only show info - no download needed
  const hasNativeGpu =
    health.gpu_available &&
    !isCurrentlyCuda &&
    health.gpu_type &&
    !health.gpu_type.includes('CUDA');

  return (
    <Card>
      <CardHeader>
        <CardTitle>GPU Acceleration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* GPU status */}
        <div className="space-y-1">
          {health.gpu_available && health.gpu_type ? (
            <>
              <div className="text-sm font-medium">
                {health.gpu_type.replace(/^(CUDA|ROCm|MPS|Metal|XPU|DirectML)\s*\((.+)\)$/, '$2') ||
                  health.gpu_type}
              </div>
              <div className="text-sm text-muted-foreground">
                {health.gpu_type.replace(/\s*\(.+\)$/, '')}
                {health.vram_used_mb != null && health.vram_used_mb > 0
                  ? ` \u00b7 ${health.vram_used_mb.toFixed(0)} MB VRAM used`
                  : ''}
              </div>
            </>
          ) : (
            <>
              <div className="text-sm font-medium">CPU</div>
              <div className="text-sm text-muted-foreground">No GPU acceleration available</div>
            </>
          )}
        </div>

        {/* Currently running CUDA - show switch back to CPU */}
        {isCurrentlyCuda && platform.metadata.isTauri && (
          <>
            {restartPhase !== 'idle' ? (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-primary/5 border">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">
                  {restartPhase === 'stopping' && 'Stopping server...'}
                  {restartPhase === 'waiting' && 'Restarting server...'}
                  {restartPhase === 'ready' && 'Server restarted successfully!'}
                </span>
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Running with CUDA GPU acceleration. Switch back to CPU if needed (you can
                  re-download later).
                </p>
                <Button
                  onClick={handleSwitchToCpuFromCuda}
                  variant="outline"
                  className="w-full"
                  size="sm"
                >
                  <RotateCw className="h-4 w-4 mr-2" />
                  Switch to CPU Backend
                </Button>
              </div>
            )}
            {error && (
              <div className="flex items-center gap-2 text-sm text-destructive">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </>
        )}

        {/* Currently running ROCm - show switch back to CPU */}
        {isCurrentlyRocm && platform.metadata.isTauri && (
          <>
            {restartPhase !== 'idle' ? (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-primary/5 border">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">
                  {restartPhase === 'stopping' && 'Stopping server...'}
                  {restartPhase === 'waiting' && 'Restarting server...'}
                  {restartPhase === 'ready' && 'Server restarted successfully!'}
                </span>
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Running with ROCm GPU acceleration for AMD. Switch back to CPU if needed (you can
                  re-download later).
                </p>
                <Button
                  onClick={handleSwitchToCpuFromRocm}
                  variant="outline"
                  className="w-full"
                  size="sm"
                >
                  <RotateCw className="h-4 w-4 mr-2" />
                  Switch to CPU Backend
                </Button>
              </div>
            )}
            {error && (
              <div className="flex items-center gap-2 text-sm text-destructive">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </>
        )}

        {/* Backend download/manage sections - show when no native GPU and not currently running GPU */}
        {!hasNativeGpu && !isCurrentlyCuda && !isCurrentlyRocm && (
          <>
            {/* CUDA Section */}
            <div className="space-y-4">
              <div className="text-sm font-medium">NVIDIA (CUDA)</div>

              {/* CUDA Download progress */}
              {cudaDownloading && downloadProgress && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>
                        {downloadProgress.filename ||
                          (cudaAvailable
                            ? 'Updating CUDA backend...'
                            : 'Downloading CUDA backend...')}
                      </span>
                    </div>
                    {downloadProgress.total > 0 && (
                      <span className="text-muted-foreground">
                        {downloadProgress.progress.toFixed(1)}%
                      </span>
                    )}
                  </div>
                  {downloadProgress.total > 0 && (
                    <>
                      <Progress value={downloadProgress.progress} className="h-2" />
                      <div className="text-xs text-muted-foreground">
                        {formatBytes(downloadProgress.current)} /{' '}
                        {formatBytes(downloadProgress.total)}
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* CUDA Actions */}
              {restartPhase === 'idle' && !cudaDownloading && (
                <div className="space-y-2">
                  {!cudaAvailable && (
                    <div className="space-y-3">
                      <p className="text-sm text-muted-foreground">
                        Download the CUDA backend (~2.4 GB) for NVIDIA GPU acceleration. Requires an
                        NVIDIA GPU with CUDA support.
                      </p>
                      <Button onClick={handleDownloadCuda} className="w-full" size="sm">
                        <Download className="h-4 w-4 mr-2" />
                        Download CUDA Backend
                      </Button>
                    </div>
                  )}

                  {cudaAvailable && platform.metadata.isTauri && (
                    <div className="space-y-3">
                      <p className="text-sm text-muted-foreground">
                        CUDA backend is downloaded and ready. Restart the server to enable GPU
                        acceleration.
                      </p>
                      <Button onClick={handleRestart} className="w-full" size="sm">
                        <RotateCw className="h-4 w-4 mr-2" />
                        Switch to CUDA Backend
                      </Button>
                    </div>
                  )}

                  {cudaAvailable && (
                    <Button
                      onClick={handleDeleteCuda}
                      variant="ghost"
                      className="w-full text-muted-foreground hover:text-destructive"
                      size="sm"
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      Remove CUDA Backend
                    </Button>
                  )}
                </div>
              )}
            </div>

            {/* Divider */}
            <div className="border-t" />

            {/* ROCm Section */}
            <div className="space-y-4">
              <div className="text-sm font-medium">AMD (ROCm)</div>

              {/* ROCm Download progress */}
              {rocmDownloading && rocmDownloadProgress && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>
                        {rocmDownloadProgress.filename ||
                          (rocmAvailable
                            ? 'Updating ROCm backend...'
                            : 'Downloading ROCm backend...')}
                      </span>
                    </div>
                    {rocmDownloadProgress.total > 0 && (
                      <span className="text-muted-foreground">
                        {rocmDownloadProgress.progress.toFixed(1)}%
                      </span>
                    )}
                  </div>
                  {rocmDownloadProgress.total > 0 && (
                    <>
                      <Progress value={rocmDownloadProgress.progress} className="h-2" />
                      <div className="text-xs text-muted-foreground">
                        {formatBytes(rocmDownloadProgress.current)} /{' '}
                        {formatBytes(rocmDownloadProgress.total)}
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* ROCm Actions */}
              {restartPhase === 'idle' && !rocmDownloading && (
                <div className="space-y-2">
                  {!rocmAvailable && (
                    <div className="space-y-3">
                      <p className="text-sm text-muted-foreground">
                        Download the ROCm backend (~2-3 GB) for AMD GPU acceleration. Requires an
                        AMD Radeon GPU with ROCm support.
                      </p>
                      <Button onClick={handleDownloadRocm} className="w-full" size="sm">
                        <Download className="h-4 w-4 mr-2" />
                        Download AMD ROCm Backend
                      </Button>
                    </div>
                  )}

                  {rocmAvailable && platform.metadata.isTauri && (
                    <div className="space-y-3">
                      <p className="text-sm text-muted-foreground">
                        ROCm backend is downloaded and ready. Restart the server to enable AMD GPU
                        acceleration.
                      </p>
                      <Button onClick={handleRestart} className="w-full" size="sm">
                        <RotateCw className="h-4 w-4 mr-2" />
                        Switch to ROCm Backend
                      </Button>
                    </div>
                  )}

                  {rocmAvailable && (
                    <Button
                      onClick={handleDeleteRocm}
                      variant="ghost"
                      className="w-full text-muted-foreground hover:text-destructive"
                      size="sm"
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      Remove ROCm Backend
                    </Button>
                  )}
                </div>
              )}
            </div>

            {/* Restart in progress */}
            {restartPhase !== 'idle' && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-primary/5 border">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">
                  {restartPhase === 'stopping' && 'Stopping server...'}
                  {restartPhase === 'waiting' && 'Restarting server...'}
                  {restartPhase === 'ready' && 'Server restarted successfully!'}
                </span>
              </div>
            )}

            {/* Error display */}
            {error && (
              <div className="flex items-center gap-2 text-sm text-destructive">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
