import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { AlertTriangle, ExternalLink } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { usePlatform } from '@/platform/PlatformContext';

/**
 * Tracks macOS Accessibility permission state. Without this permission the
 * global chord can still record, but the synthetic-⌘V paste silently drops —
 * so callers can surface an inline prompt instead of relying on the
 * system-level permission dialog (which only fires once, the first time the
 * app tries to post a keystroke).
 *
 * Triggered on three signals:
 * - app mount in Tauri
 * - `system:accessibility-missing` event from the dictate window's paste
 *   failure handler
 * - window focus (cheap way to re-check after the user flips the toggle in
 *   System Settings and alt-tabs back)
 */
export function useAccessibilityPermission() {
  const platform = usePlatform();
  const [needsPermission, setNeedsPermission] = useState(false);
  const [checking, setChecking] = useState(false);

  const recheck = useCallback(async (): Promise<boolean> => {
    if (!platform.metadata.isTauri) return true;
    setChecking(true);
    try {
      const trusted = await invoke<boolean>('check_accessibility_permission');
      setNeedsPermission(!trusted);
      return trusted;
    } catch (err) {
      console.warn('[accessibility] check failed:', err);
      return false;
    } finally {
      setChecking(false);
    }
  }, [platform.metadata.isTauri]);

  useEffect(() => {
    if (!platform.metadata.isTauri) return;
    recheck();
    const onFocus = () => {
      recheck();
    };
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [platform.metadata.isTauri, recheck]);

  useEffect(() => {
    if (!platform.metadata.isTauri) return;
    let unlisten: UnlistenFn | null = null;
    listen('system:accessibility-missing', () => {
      setNeedsPermission(true);
    })
      .then((fn) => {
        unlisten = fn;
      })
      .catch(() => {});
    return () => {
      if (unlisten) unlisten();
    };
  }, [platform.metadata.isTauri]);

  const openSettings = useCallback(async () => {
    try {
      await invoke('open_accessibility_settings');
    } catch (err) {
      console.warn('[accessibility] open settings failed:', err);
    }
  }, []);

  return { needsPermission, checking, recheck, openSettings };
}

/**
 * Inline notice rendered next to the auto-paste setting when macOS
 * Accessibility permission is missing. Returns null when the permission is
 * already granted.
 */
export function AccessibilityNotice() {
  const { t } = useTranslation();
  const { needsPermission, checking, recheck, openSettings } = useAccessibilityPermission();
  const [stillMissing, setStillMissing] = useState(false);

  const handleRecheck = useCallback(async () => {
    setStillMissing(false);
    const trusted = await recheck();
    if (!trusted) setStillMissing(true);
  }, [recheck]);

  if (!needsPermission) return null;

  return (
    <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3.5 py-3">
      <div className="flex items-start gap-3">
        <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-amber-500" />
        <div className="flex-1 min-w-0 space-y-1">
          <p className="text-sm font-medium text-foreground">
            {t('captures.permissions.accessibility.title')}
          </p>
          <p className="text-sm text-muted-foreground leading-relaxed">
            <Trans i18nKey="captures.permissions.accessibility.body" components={{ path: <span /> }} />
          </p>
          <div className="flex items-center gap-2 pt-1.5">
            <Button size="sm" onClick={openSettings} className="gap-1.5">
              <ExternalLink className="h-3.5 w-3.5" />
              {t('captures.permissions.accessibility.openSettings')}
            </Button>
            <Button variant="outline" size="sm" onClick={handleRecheck} disabled={checking}>
              {checking ? t('captures.permissions.accessibility.rechecking') : t('captures.permissions.accessibility.recheck')}
            </Button>
          </div>
          {stillMissing && !checking && (
            <p className="text-xs text-amber-600 dark:text-amber-400 pt-1">
              {t('captures.permissions.accessibility.stillMissing')}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
