import { invoke } from '@tauri-apps/api/core';
import { AlertTriangle, ExternalLink } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { usePlatform } from '@/platform/PlatformContext';

/**
 * Tracks macOS Input Monitoring permission state. Without it, `rdev::listen`
 * sees no key events and the chord engine never fires — but neither does
 * anything error-out visibly, so we surface an inline prompt next to the
 * hotkey toggle instead of leaving the user wondering why the shortcut is
 * dead.
 *
 * Re-checked on mount and on window focus (cheap way to pick up the user
 * flipping the toggle in System Settings and alt-tabbing back).
 */
export function useInputMonitoringPermission() {
  const platform = usePlatform();
  const [needsPermission, setNeedsPermission] = useState(false);
  const [checking, setChecking] = useState(false);

  const recheck = useCallback(async (): Promise<boolean> => {
    if (!platform.metadata.isTauri) return true;
    setChecking(true);
    try {
      const trusted = await invoke<boolean>('check_input_monitoring_permission');
      setNeedsPermission(!trusted);
      return trusted;
    } catch (err) {
      console.warn('[input-monitoring] check failed:', err);
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

  const openSettings = useCallback(async () => {
    try {
      await invoke('open_input_monitoring_settings');
    } catch (err) {
      console.warn('[input-monitoring] open settings failed:', err);
    }
  }, []);

  return { needsPermission, checking, recheck, openSettings };
}

/**
 * Inline notice rendered under the global-shortcut toggle when the user has
 * opted in but macOS Input Monitoring is not granted. Returns null when the
 * permission is present (or when the toggle is off and the notice would just
 * be noise).
 */
export function InputMonitoringNotice({ enabled }: { enabled: boolean }) {
  const { t } = useTranslation();
  const { needsPermission, checking, recheck, openSettings } =
    useInputMonitoringPermission();
  const [stillMissing, setStillMissing] = useState(false);

  const handleRecheck = useCallback(async () => {
    setStillMissing(false);
    const trusted = await recheck();
    if (!trusted) setStillMissing(true);
  }, [recheck]);

  if (!enabled || !needsPermission) return null;

  return (
    <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3.5 py-3">
      <div className="flex items-start gap-3">
        <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-amber-500" />
        <div className="flex-1 min-w-0 space-y-1">
          <p className="text-sm font-medium text-foreground">
            {t('captures.permissions.inputMonitoring.title')}
          </p>
          <p className="text-sm text-muted-foreground leading-relaxed">
            <Trans i18nKey="captures.permissions.inputMonitoring.body" components={{ path: <span /> }} />
          </p>
          <div className="flex items-center gap-2 pt-1.5">
            <Button size="sm" onClick={openSettings} className="gap-1.5">
              <ExternalLink className="h-3.5 w-3.5" />
              {t('captures.permissions.inputMonitoring.openSettings')}
            </Button>
            <Button variant="outline" size="sm" onClick={handleRecheck} disabled={checking}>
              {checking ? t('captures.permissions.inputMonitoring.rechecking') : t('captures.permissions.inputMonitoring.recheck')}
            </Button>
          </div>
          {stillMissing && !checking && (
            <p className="text-xs text-amber-600 dark:text-amber-400 pt-1">
              {t('captures.permissions.inputMonitoring.stillMissing')}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
