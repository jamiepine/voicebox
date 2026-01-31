import { Download, RefreshCw } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Progress } from '@/components/ui/progress';
import { ToastAction } from '@/components/ui/toast';
import { useToast } from '@/components/ui/use-toast';
import { usePlatform } from '@/platform/PlatformContext';
import type { UpdateStatus } from '@/platform/types';

// Re-export UpdateStatus for backwards compatibility
export type { UpdateStatus };

interface UseAutoUpdaterOptions {
  checkOnMount?: boolean;
  showToast?: boolean;
}

export function useAutoUpdater(options: boolean | UseAutoUpdaterOptions = false) {
  // Support both old boolean API and new options object
  const { checkOnMount, showToast } =
    typeof options === 'boolean'
      ? { checkOnMount: options, showToast: false }
      : { checkOnMount: options.checkOnMount ?? false, showToast: options.showToast ?? false };

  const platform = usePlatform();
  const { toast } = useToast();
  const [status, setStatus] = useState<UpdateStatus>(platform.updater.getStatus());
  const hasCheckedRef = useRef(false);
  const toastIdRef = useRef<string | null>(null);
  const toastUpdateRef = useRef<
    | ((props: {
        title?: React.ReactNode;
        description?: React.ReactNode;
        duration?: number;
        variant?: 'default' | 'destructive';
        open?: boolean;
        action?: React.ReactElement<typeof ToastAction>;
      }) => void)
    | null
  >(null);

  // Subscribe to updater status changes
  useEffect(() => {
    const unsubscribe = platform.updater.subscribe((newStatus) => {
      setStatus(newStatus);
    });
    return unsubscribe;
    // Empty dependency array - platform is stable from context
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform.updater.subscribe]);

  const checkForUpdates = useCallback(async () => {
    await platform.updater.checkForUpdates();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform.updater.checkForUpdates]);

  const downloadAndInstall = useCallback(async () => {
    await platform.updater.downloadAndInstall();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform.updater.downloadAndInstall]);

  const restartAndInstall = useCallback(async () => {
    await platform.updater.restartAndInstall();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform.updater.restartAndInstall]);

  // Check for updates on mount
  useEffect(() => {
    if (checkOnMount && platform.metadata.isTauri && !hasCheckedRef.current) {
      hasCheckedRef.current = true;
      checkForUpdates().catch((error) => {
        console.error('Auto update check failed:', error);
      });
    }
    // Empty dependency array - only run once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [platform.metadata.isTauricheckOnMountcheckForUpdates]);

  // Show toast when update is available
  useEffect(() => {
    if (
      !showToast ||
      !status.available ||
      status.downloading ||
      status.readyToInstall ||
      toastIdRef.current
    ) {
      return;
    }

    const handleUpdateNow = async () => {
      await downloadAndInstall();
    };

    const toastResult = toast({
      title: 'Update Available',
      description: `Version ${status.version} is ready to download.`,
      duration: Infinity,
      action: (
        <ToastAction altText="Update now" onClick={handleUpdateNow}>
          Update Now
        </ToastAction>
      ),
    });

    toastIdRef.current = toastResult.id;
    // Type assertion needed because update function has broader type than our ref
    toastUpdateRef.current = toastResult.update as typeof toastUpdateRef.current;
  }, [
    showToast,
    status.available,
    status.downloading,
    status.readyToInstall,
    status.version,
    downloadAndInstall,
    toast,
  ]);

  // Update toast when downloading
  useEffect(() => {
    if (!showToast || !status.downloading || !toastIdRef.current || !toastUpdateRef.current) {
      return;
    }

    const progressPercent = status.downloadProgress || 0;
    const progressText =
      status.downloadedBytes !== undefined &&
      status.totalBytes !== undefined &&
      status.totalBytes > 0
        ? `${(status.downloadedBytes / 1024 / 1024).toFixed(1)} MB / ${(status.totalBytes / 1024 / 1024).toFixed(1)} MB`
        : '';

    toastUpdateRef.current({
      title: (
        <div className="flex items-center gap-2">
          <Download className="h-4 w-4 animate-pulse" />
          <span>Downloading Update</span>
        </div>
      ),
      description: (
        <div className="space-y-2">
          <div className="text-sm">Version {status.version}</div>
          {progressPercent > 0 && (
            <>
              <Progress value={progressPercent} className="h-2" />
              {progressText && <div className="text-xs text-muted-foreground">{progressText}</div>}
            </>
          )}
        </div>
      ),
      duration: Infinity,
    });
  }, [
    showToast,
    status.downloading,
    status.downloadProgress,
    status.downloadedBytes,
    status.totalBytes,
    status.version,
  ]);

  // Update toast when ready to install
  useEffect(() => {
    if (!showToast || !status.readyToInstall || !toastIdRef.current || !toastUpdateRef.current) {
      return;
    }

    const handleRestartNow = async () => {
      await restartAndInstall();
    };

    toastUpdateRef.current({
      title: 'Update Ready',
      description: `Version ${status.version} has been downloaded and is ready to install.`,
      duration: Infinity,
      action: (
        <ToastAction altText="Restart now" onClick={handleRestartNow}>
          <RefreshCw className="h-3 w-3 mr-1" />
          Restart Now
        </ToastAction>
      ),
    });
  }, [showToast, status.readyToInstall, status.version, restartAndInstall]);

  // Handle errors in toast
  useEffect(() => {
    if (!showToast || !status.error || !toastIdRef.current || !toastUpdateRef.current) {
      return;
    }

    toastUpdateRef.current({
      title: 'Update Failed',
      description: status.error,
      variant: 'destructive',
      duration: 5000,
    });

    setTimeout(() => {
      toastIdRef.current = null;
      toastUpdateRef.current = null;
    }, 5000);
  }, [showToast, status.error]);

  return {
    status,
    checkForUpdates,
    downloadAndInstall,
    restartAndInstall,
  };
}
