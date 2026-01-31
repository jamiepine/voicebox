import { CheckCircle2, Loader2, XCircle } from 'lucide-react';
import { useCallback, useEffect, useRef } from 'react';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/components/ui/use-toast';
import type { ModelProgress } from '@/lib/api/types';
import { useServerStore } from '@/stores/serverStore';

interface UseModelDownloadToastOptions {
  modelName: string;
  displayName: string;
  enabled?: boolean;
  onComplete?: () => void;
  onError?: () => void;
}

/**
 * Hook to show and update a toast notification with model download progress.
 * Subscribes to Server-Sent Events for real-time progress updates.
 */
export function useModelDownloadToast({
  modelName,
  displayName,
  enabled = false,
  onComplete,
  onError,
}: UseModelDownloadToastOptions) {
  const { toast } = useToast();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const toastIdRef = useRef<string | null>(null);
  // biome-ignore lint: Using any for toast update ref to handle complex toast types
  const toastUpdateRef = useRef<any>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const formatBytes = useCallback((bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / k ** i).toFixed(1)} ${sizes[i]}`;
  }, []);

  useEffect(() => {
    console.log('[useModelDownloadToast] useEffect triggered', {
      enabled,
      serverUrl,
      modelName,
      displayName,
    });

    if (!enabled || !serverUrl || !modelName) {
      console.log('[useModelDownloadToast] Not enabled, skipping');
      return;
    }

    console.log('[useModelDownloadToast] Creating toast and EventSource for:', modelName);

    // Create initial toast
    const toastResult = toast({
      title: displayName,
      description: (
        <div className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Connecting to download...</span>
        </div>
      ),
      duration: Infinity, // Don't auto-dismiss, we'll handle it manually
    });
    toastIdRef.current = toastResult.id;
    toastUpdateRef.current = toastResult.update;

    // Subscribe to progress updates via Server-Sent Events
    const eventSourceUrl = `${serverUrl}/models/progress/${modelName}`;
    console.log('[useModelDownloadToast] Creating EventSource to:', eventSourceUrl);
    const eventSource = new EventSource(eventSourceUrl);

    eventSource.onopen = () => {
      console.log('[useModelDownloadToast] EventSource connection opened for:', modelName);
    };

    eventSource.onmessage = (event) => {
      console.log('[useModelDownloadToast] Received SSE message:', event.data);
      try {
        const progress = JSON.parse(event.data) as ModelProgress;

        // Update toast with progress
        if (toastIdRef.current && toastUpdateRef.current) {
          const progressPercent = progress.total > 0 ? progress.progress : 0;
          const progressText =
            progress.total > 0
              ? `${formatBytes(progress.current)} / ${formatBytes(progress.total)} (${progress.progress.toFixed(1)}%)`
              : '';

          // Determine status icon and text
          let statusIcon: React.ReactNode = null;
          let statusText = 'Processing...';

          switch (progress.status) {
            case 'complete':
              statusIcon = <CheckCircle2 className="h-4 w-4 text-green-500" />;
              statusText = 'Download complete';
              break;
            case 'error':
              statusIcon = <XCircle className="h-4 w-4 text-destructive" />;
              statusText = `Error: ${progress.error || 'Unknown error'}`;
              break;
            case 'downloading':
              statusIcon = <Loader2 className="h-4 w-4 animate-spin" />;
              statusText = progress.filename || 'Downloading...';
              break;
            case 'extracting':
              statusIcon = <Loader2 className="h-4 w-4 animate-spin" />;
              statusText = 'Extracting...';
              break;
          }

          toastUpdateRef.current({
            title: (
              <div className="flex items-center gap-2">
                {statusIcon}
                <span>{displayName}</span>
              </div>
            ),
            description: (
              <div className="space-y-2">
                <div className="text-sm">{statusText}</div>
                {progress.total > 0 && (
                  <>
                    <Progress value={progressPercent} className="h-2" />
                    <div className="text-xs text-muted-foreground">{progressText}</div>
                  </>
                )}
              </div>
            ),
            duration: progress.status === 'complete' ? 5000 : Infinity,
            variant: progress.status === 'error' ? 'destructive' : 'default',
          });

          // Close connection and dismiss toast on completion or error
          // Also treat progress >= 100% as complete
          const isComplete = progress.status === 'complete' || progress.progress >= 100;
          const isError = progress.status === 'error';

          if (isComplete || isError) {
            console.log('[useModelDownloadToast] Download finished:', {
              isComplete,
              isError,
              progress: progress.progress,
            });
            eventSource.close();
            eventSourceRef.current = null;

            // Update toast to show completion state before callbacks
            if (isComplete && toastUpdateRef.current) {
              toastUpdateRef.current({
                title: (
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <span>{displayName}</span>
                  </div>
                ),
                description: 'Download complete',
                duration: 3000,
              });
            }

            // Call callbacks
            if (isComplete && onComplete) {
              console.log('[useModelDownloadToast] Download complete, calling onComplete callback');
              onComplete();
            } else if (isError && onError) {
              console.log('[useModelDownloadToast] Download error, calling onError callback');
              onError();
            }
          }
        }
      } catch (error) {
        console.error('Error parsing progress event:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('[useModelDownloadToast] SSE error for:', modelName, error);
      console.log('[useModelDownloadToast] EventSource readyState:', eventSource.readyState);
      eventSource.close();
      eventSourceRef.current = null;

      // Show error toast
      if (toastIdRef.current && toastUpdateRef.current) {
        toastUpdateRef.current({
          title: displayName,
          description: 'Failed to track download progress',
          variant: 'destructive',
          duration: 5000,
        });
        toastIdRef.current = null;
        toastUpdateRef.current = null;
      }
    };

    eventSourceRef.current = eventSource;

    // Cleanup on unmount or when disabled
    return () => {
      console.log('[useModelDownloadToast] Cleanup - closing EventSource for:', modelName);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      // Note: We don't dismiss the toast here as it might still be showing completion state
    };
  }, [enabled, serverUrl, modelName, displayName, toast, formatBytes, onComplete, onError]);

  return {
    isTracking: enabled && eventSourceRef.current !== null,
  };
}
