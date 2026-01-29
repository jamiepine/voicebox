import { useEffect, useRef } from 'react';
import { useToast } from '@/components/ui/use-toast';
import { useServerStore } from '@/stores/serverStore';
import { Progress } from '@/components/ui/progress';
import { Loader2, CheckCircle2, XCircle } from 'lucide-react';
import type { ModelProgress } from '@/lib/api/types';

interface UseModelDownloadToastOptions {
  modelName: string;
  displayName: string;
  enabled?: boolean;
}

/**
 * Hook to show and update a toast notification with model download progress.
 * Subscribes to Server-Sent Events for real-time progress updates.
 */
export function useModelDownloadToast({
  modelName,
  displayName,
  enabled = false,
}: UseModelDownloadToastOptions) {
  const { toast } = useToast();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const toastIdRef = useRef<string | null>(null);
  const toastUpdateRef = useRef<
    ((props: {
      title?: React.ReactNode;
      description?: React.ReactNode;
      duration?: number;
      variant?: 'default' | 'destructive';
      open?: boolean;
    }) => void) | null
  >(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  useEffect(() => {
    if (!enabled || !serverUrl || !modelName) {
      return;
    }

    // Create initial toast
    const toastResult = toast({
      title: displayName,
      description: 'Starting download...',
      duration: Infinity, // Don't auto-dismiss, we'll handle it manually
    });
    toastIdRef.current = toastResult.id;
    toastUpdateRef.current = toastResult.update;

    // Subscribe to progress updates via Server-Sent Events
    const eventSource = new EventSource(`${serverUrl}/models/progress/${modelName}`);

    eventSource.onmessage = (event) => {
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
              statusText = progress.filename ? `Downloading ${progress.filename}...` : 'Downloading...';
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
          if (progress.status === 'complete' || progress.status === 'error') {
            eventSource.close();
            eventSourceRef.current = null;

            // Auto-dismiss on completion after delay
            if (progress.status === 'complete') {
              setTimeout(() => {
                if (toastIdRef.current && toastUpdateRef.current) {
                  toastUpdateRef.current({
                    open: false,
                  });
                  toastIdRef.current = null;
                  toastUpdateRef.current = null;
                }
              }, 5000);
            }
          }
        }
      } catch (error) {
        console.error('Error parsing progress event:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE error:', error);
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
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      // Note: We don't dismiss the toast here as it might still be showing completion state
    };
  }, [enabled, serverUrl, modelName, displayName, toast]);

  return {
    isTracking: enabled && eventSourceRef.current !== null,
  };
}