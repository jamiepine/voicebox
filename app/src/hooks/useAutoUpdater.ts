import { useCallback, useEffect, useRef, useState } from 'react';
import { usePlatform } from '@/platform/PlatformContext';
import type { UpdateStatus } from '@/platform/types';

// Re-export UpdateStatus for backwards compatibility
export type { UpdateStatus };

interface UseAutoUpdaterOptions {
  checkOnMount?: boolean;
  showToast?: boolean;
}

export function useAutoUpdater(options: boolean | UseAutoUpdaterOptions = false) {
  const { checkOnMount } =
    typeof options === 'boolean' ? { checkOnMount: options } : { checkOnMount: options.checkOnMount ?? false };

  const platform = usePlatform();
  const [status, setStatus] = useState<UpdateStatus>(platform.updater.getStatus());
  const hasCheckedRef = useRef(false);

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

  useEffect(() => {
    if (checkOnMount && platform.metadata.isTauri && !hasCheckedRef.current) {
      hasCheckedRef.current = true;
      checkForUpdates().catch((error) => {
        console.error('Auto update check failed:', error);
      });
    }
  }, [checkOnMount, checkForUpdates, platform.metadata.isTauri]);

  return {
    status,
    checkForUpdates,
    downloadAndInstall,
    restartAndInstall,
  };
}
