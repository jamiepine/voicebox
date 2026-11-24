import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { ExternalLink, MicOff } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  MICROPHONE_BLOCKED_EVENT,
  requestMicrophoneAccess,
  useMicrophonePermissionStore,
} from '@/lib/hooks/useMicrophonePermission';
import { usePlatform } from '@/platform/PlatformContext';

function detectOs(isTauri: boolean): 'windows' | 'macos' | 'linux' | 'browser' {
  if (!isTauri) return 'browser';
  const ua = navigator.userAgent;
  if (ua.includes('Windows')) return 'windows';
  if (ua.includes('Mac')) return 'macos';
  return 'linux';
}

/**
 * Shared recovery surface for every in-app recording entry point. Opened by
 * useAudioRecording when getUserMedia is denied, and by the dictate window
 * through MICROPHONE_BLOCKED_EVENT.
 */
export function MicrophonePermissionDialog() {
  const { t } = useTranslation();
  const platform = usePlatform();
  const { blocked, setBlocked } = useMicrophonePermissionStore();
  const [retrying, setRetrying] = useState(false);
  const [stillBlocked, setStillBlocked] = useState(false);
  const os = detectOs(platform.metadata.isTauri);

  useEffect(() => {
    if (!platform.metadata.isTauri) return;
    let unlisten: UnlistenFn | null = null;
    listen(MICROPHONE_BLOCKED_EVENT, () => setBlocked(true))
      .then((fn) => {
        unlisten = fn;
      })
      .catch(() => {});
    return () => {
      if (unlisten) unlisten();
    };
  }, [platform.metadata.isTauri, setBlocked]);

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (open) return;
      setBlocked(false);
      setStillBlocked(false);
    },
    [setBlocked],
  );

  const handleRetry = useCallback(async () => {
    setRetrying(true);
    setStillBlocked(false);
    const granted = await requestMicrophoneAccess();
    setRetrying(false);
    if (granted) {
      handleOpenChange(false);
      return;
    }
    setStillBlocked(true);
  }, [handleOpenChange]);

  const handleOpenSettings = useCallback(async () => {
    try {
      await invoke('open_microphone_settings');
    } catch (err) {
      console.warn('[microphone] open settings failed:', err);
    }
  }, []);

  return (
    <Dialog open={blocked} onOpenChange={handleOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <MicOff className="h-5 w-5 text-amber-500" />
            {t('microphonePermission.title')}
          </DialogTitle>
          <DialogDescription className="leading-relaxed">
            {t(`microphonePermission.body.${os}`)}
          </DialogDescription>
        </DialogHeader>
        {stillBlocked && (
          <p className="text-sm text-amber-600 dark:text-amber-400">
            {t(`microphonePermission.stillBlocked.${os === 'macos' ? 'macos' : 'default'}`)}
          </p>
        )}
        <DialogFooter className="gap-2">
          {(os === 'windows' || os === 'macos') && (
            <Button variant="outline" onClick={handleOpenSettings} className="gap-1.5">
              <ExternalLink className="h-3.5 w-3.5" />
              {t('microphonePermission.openSettings')}
            </Button>
          )}
          <Button onClick={handleRetry} disabled={retrying}>
            {retrying ? t('microphonePermission.retrying') : t('microphonePermission.retry')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
