import { CheckCircle2, Loader2, X, XCircle } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { toast } from '@/components/ui/use-toast';
import type { ModelProgress } from '@/lib/api/types';
import { apiClient } from '@/lib/api/client';
import { useServerStore } from '@/stores/serverStore';

const POLL_MS = 1000;

interface UseModelDownloadToastOptions {
  modelName: string;
  displayName: string;
  enabled?: boolean;
  onComplete?: () => void;
  onError?: () => void;
  onCancel?: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / k ** i).toFixed(1)} ${sizes[i]}`;
}

/**
 * Hook to show and update a toast notification with model download/load progress.
 * Polls the server every second — simpler than SSE, no connection-count issues.
 */
export function useModelDownloadToast({
  modelName,
  displayName,
  enabled = false,
  onComplete,
  onError,
  onCancel,
}: UseModelDownloadToastOptions) {
  const serverUrl = useServerStore((state) => state.serverUrl);

  const onCompleteRef = useRef(onComplete);
  const onErrorRef = useRef(onError);
  const onCancelRef = useRef(onCancel);
  const displayNameRef = useRef(displayName);
  onCompleteRef.current = onComplete;
  onErrorRef.current = onError;
  onCancelRef.current = onCancel;
  displayNameRef.current = displayName;

  useEffect(() => {
    if (!enabled || !serverUrl || !modelName) return;

    let stopped = false;

    // Create toast once — capture update/dismiss in closure-stable refs
    const { update: updateToast, dismiss: dismissToast } = toast({
      title: displayNameRef.current,
      description: (
        <div className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Starting...</span>
        </div>
      ),
      duration: Infinity,
    });

    const handleCancel = async () => {
      stopped = true;
      try { await apiClient.cancelModelDownload(modelName); } catch { /* ignore */ }
      dismissToast();
      onCancelRef.current?.();
    };

    const renderToast = (progress: ModelProgress | null) => {
      const hasTotal = !!progress && progress.total > 0;
      const progressPercent = hasTotal ? progress!.progress : 0;
      const progressText = hasTotal
        ? `${formatBytes(progress!.current)} / ${formatBytes(progress!.total)} (${progress!.progress.toFixed(1)}%)`
        : '';

      const status = progress?.status ?? 'downloading';
      const isTerminal = status === 'complete' || status === 'error';
      const showCancel = !isTerminal && status !== 'loading';

      let statusIcon: React.ReactNode = <Loader2 className="h-4 w-4 animate-spin" />;
      let statusText = 'Downloading...';

      if (status === 'complete') {
        statusIcon = <CheckCircle2 className="h-4 w-4 text-green-500" />;
        statusText = 'Download complete';
      } else if (status === 'error') {
        statusIcon = <XCircle className="h-4 w-4 text-destructive" />;
        statusText = `Error: ${progress?.error || 'Unknown error'}`;
      } else if (status === 'loading') {
        statusText = 'Loading model...';
      } else if (status === 'extracting') {
        statusText = 'Extracting...';
      } else {
        statusText = progress?.filename || 'Downloading...';
      }

      // biome-ignore lint: updateToast expects ToasterToast but id is captured in closure
      (updateToast as any)({
        title: (
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              {statusIcon}
              <span>{displayNameRef.current}</span>
            </div>
            {showCancel && (
              <Button size="sm" variant="ghost" className="h-5 w-5 p-0 shrink-0" onClick={handleCancel} title="Cancel">
                <X className="h-3 w-3" />
              </Button>
            )}
          </div>
        ),
        description: (
          <div className="space-y-2">
            <div className="text-sm">{statusText}</div>
            {status !== 'loading' && (
              hasTotal ? (
                <>
                  <Progress value={progressPercent} className="h-2" />
                  <div className="text-xs text-muted-foreground">{progressText}</div>
                </>
              ) : (
                <div className="h-2 w-full rounded-full bg-secondary overflow-hidden">
                  <div className="h-full w-1/3 rounded-full bg-primary animate-[indeterminate_1.5s_ease-in-out_infinite]" />
                </div>
              )
            )}
          </div>
        ),
        duration: isTerminal ? 4000 : Infinity,
        variant: status === 'error' ? 'destructive' : 'default',
      });
    };

    // Poll loop
    const poll = async () => {
      if (stopped) return;
      try {
        const res = await fetch(`${serverUrl}/models/progress-snapshot/${modelName}`);
        if (stopped) return;
        if (res.ok) {
          const data = await res.json();
          // 'idle' means no active download (finished, cancelled, or not started)
          if (data.status === 'idle') {
            stopped = true;
            dismissToast();
            return;
          }
          const progress: ModelProgress = data;
          renderToast(progress);

          if (progress.status === 'complete' || (progress.progress ?? 0) >= 100) {
            stopped = true;
            onCompleteRef.current?.();
            return;
          }
          if (progress.status === 'error') {
            stopped = true;
            onErrorRef.current?.();
            return;
          }
        }
      } catch {
        // server temporarily unavailable, keep polling
      }
      if (!stopped) intervalId = window.setTimeout(poll, POLL_MS);
    };

    let intervalId = window.setTimeout(poll, POLL_MS);

    return () => {
      stopped = true;
      clearTimeout(intervalId);
      dismissToast();
    };
  }, [enabled, serverUrl, modelName]);
}
