import { useQuery } from '@tanstack/react-query';
import { CheckCircle2, Loader2, XCircle } from 'lucide-react';
import { useCallback, useEffect, useRef } from 'react';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { ActiveDownloadTask } from '@/lib/api/types';

interface UseModelDownloadToastOptions {
  modelName: string;
  displayName: string;
  enabled?: boolean;
  onComplete?: () => void;
  onError?: (error: string) => void;
}

const POLL_INTERVAL_MS = 1000;
// A task that never shows up in /tasks/active (e.g. it finished between the
// trigger and the first poll) is resolved after this many empty polls.
const MAX_POLLS_WITHOUT_TASK = 3;

// The toast update() props are typed for plain-string titles; rich titles are
// what the old SSE version passed too, through an untyped ref.
type ToastHandle = {
  // biome-ignore lint/suspicious/noExplicitAny: rich JSX titles/descriptions
  update: (props: any) => void;
  dismiss: () => void;
};

/**
 * Hook to show and update a toast notification with model download progress.
 *
 * Progress comes from the shared, polled ``['activeTasks']`` query rather than
 * one Server-Sent Events stream per model. Each EventSource holds an HTTP
 * connection open for the whole download, and the webview allows only six
 * per server: with six or more downloads active (queued ones included) every
 * other request -- model status, the Download button itself -- stalled
 * behind them, and dropped streams surfaced as "Failed to track download
 * progress". Polling one shared query costs a single short request per
 * interval no matter how many downloads are tracked.
 */
export function useModelDownloadToast({
  modelName,
  displayName,
  enabled = false,
  onComplete,
  onError,
}: UseModelDownloadToastOptions) {
  const { toast } = useToast();
  const toastRef = useRef<ToastHandle | null>(null);
  const seenTaskRef = useRef(false);
  const emptyPollsRef = useRef(0);
  const finishedRef = useRef(false);
  const callbacksRef = useRef({ onComplete, onError });
  callbacksRef.current = { onComplete, onError };

  const active = enabled && !!modelName;

  const { data: activeTasks, dataUpdatedAt } = useQuery({
    queryKey: ['activeTasks'],
    queryFn: () => apiClient.getActiveTasks(),
    enabled: active,
    refetchInterval: active ? POLL_INTERVAL_MS : false,
  });

  const formatBytes = useCallback((bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / k ** i).toFixed(1)} ${sizes[i]}`;
  }, []);

  // Open a fresh toast whenever tracking starts for a model.
  // biome-ignore lint/correctness/useExhaustiveDependencies: modelName restarts tracking for a new model
  useEffect(() => {
    if (!active) return;
    seenTaskRef.current = false;
    emptyPollsRef.current = 0;
    finishedRef.current = false;
    toastRef.current = toast({
      title: displayName,
      description: (
        <div className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Connecting to download...</span>
        </div>
      ),
      duration: Infinity,
    });
    // The toast is left in place on cleanup: it may be showing a final state.
  }, [active, modelName, displayName, toast]);

  const showProgress = useCallback(
    (task: ActiveDownloadTask) => {
      const total = task.total ?? 0;
      const current = task.current ?? 0;
      const percent = total > 0 ? (task.progress ?? 0) : 0;
      toastRef.current?.update({
        title: (
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>{displayName}</span>
          </div>
        ),
        description: (
          <div className="space-y-2">
            <div className="text-sm">{task.filename || 'Downloading...'}</div>
            {total > 0 && (
              <>
                <Progress value={percent} className="h-2" />
                <div className="text-xs text-muted-foreground">
                  {`${formatBytes(current)} / ${formatBytes(total)} (${percent.toFixed(1)}%)`}
                </div>
              </>
            )}
          </div>
        ),
        duration: Infinity,
      });
    },
    [displayName, formatBytes],
  );

  const finish = useCallback(
    (outcome: 'complete' | 'error' | 'gone', error?: string) => {
      finishedRef.current = true;
      const handle = toastRef.current;
      toastRef.current = null;
      if (outcome === 'complete') {
        handle?.update({
          title: (
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-green-500" />
              <span>{displayName}</span>
            </div>
          ),
          description: 'Download complete',
          duration: 3000,
        });
        callbacksRef.current.onComplete?.();
      } else if (outcome === 'error') {
        handle?.update({
          title: (
            <div className="flex items-center gap-2">
              <XCircle className="h-4 w-4 text-destructive" />
              <span>{displayName}</span>
            </div>
          ),
          description: 'Download failed. See Problems panel for details.',
          duration: 5000,
        });
        callbacksRef.current.onError?.(error || 'Unknown error');
      } else {
        // Cancelled or otherwise dropped without finishing.
        handle?.dismiss();
        callbacksRef.current.onComplete?.();
      }
    },
    [displayName],
  );

  // React to each poll result. dataUpdatedAt changes on every successful fetch,
  // so an unchanged task list still advances the empty-poll counter.
  // biome-ignore lint/correctness/useExhaustiveDependencies: dataUpdatedAt is the poll tick
  useEffect(() => {
    if (!active || finishedRef.current || !activeTasks) return;

    const task = activeTasks.downloads.find((d) => d.model_name === modelName);
    if (task) {
      seenTaskRef.current = true;
      emptyPollsRef.current = 0;
      if (task.status === 'error') {
        finish('error', task.error);
      } else {
        showProgress(task);
      }
      return;
    }

    emptyPollsRef.current += 1;
    if (!seenTaskRef.current && emptyPollsRef.current < MAX_POLLS_WITHOUT_TASK) return;

    // The task left the list: the server drops it on success, keeps it with
    // status "error" on failure, and also drops it when cancelled -- so check
    // the model's state to tell success from cancellation.
    finishedRef.current = true;
    apiClient
      .getModelStatus()
      .then((status) => {
        const model = status.models.find((m) => m.model_name === modelName);
        finish(model?.downloaded ? 'complete' : 'gone');
      })
      .catch(() => finish('gone'));
  }, [active, activeTasks, dataUpdatedAt, modelName, finish, showProgress]);

  return {
    isTracking: active && !finishedRef.current,
  };
}
