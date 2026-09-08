import { invoke } from '@tauri-apps/api/core';
import { AlertTriangle, ExternalLink } from 'lucide-react';
import { useCallback } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { usePlatform } from '@/platform/PlatformContext';

const isMacOSAgent = () => /Mac|iPhone|iPad/.test(navigator.userAgent);
const isWindowsAgent = () => /Win/.test(navigator.userAgent);

/**
 * Deep-link into the OS microphone privacy pane. Unlike the Accessibility and
 * Input Monitoring gates there is no `check_*` counterpart: WKWebView exposes
 * no permission-state query, so the recording UI learns about a denial by
 * classifying the `getUserMedia` rejection and only then offers this way out.
 */
export function useMicrophonePermissionHelp() {
  const platform = usePlatform();

  // Only macOS and Windows have a settings pane to deep-link into; the Rust
  // command errors elsewhere, so Linux must not be offered the button.
  const canOpenSettings = platform.metadata.isTauri && (isMacOSAgent() || isWindowsAgent());

  /**
   * Reveal the microphone pane in the OS settings app. No-op where there is
   * no such pane to open, which is the web build and Linux.
   */
  const openSettings = useCallback(async () => {
    if (!canOpenSettings) return;
    try {
      await invoke('open_microphone_settings');
    } catch (err) {
      console.warn('[microphone] open settings failed:', err);
    }
  }, [canOpenSettings]);

  return { canOpenSettings, openSettings };
}

/**
 * Inline notice shown in place of the record button once macOS (or the
 * browser) has refused the microphone. macOS never re-prompts after a denial,
 * so without this the user is stuck on an error toast with nothing to act on.
 */
export function MicrophonePermissionNotice({
  onRetry,
  showRestartHint = false,
}: {
  onRetry?: () => void;
  /**
   * A retry has already failed. Owned by the caller, not this component: a
   * retry re-renders the record surface, so state kept here would reset
   * before the hint could ever show.
   */
  showRestartHint?: boolean;
}) {
  const { t } = useTranslation();
  const platform = usePlatform();
  const { canOpenSettings, openSettings } = useMicrophonePermissionHelp();
  const isMacOS = isMacOSAgent();

  // Linux Tauri builds have no settings pane to point at, so they get generic
  // wording rather than the Windows steps.
  const bodyKey = !platform.metadata.isTauri
    ? 'captures.permissions.microphone.bodyWeb'
    : isMacOS
      ? 'captures.permissions.microphone.bodyMac'
      : isWindowsAgent()
        ? 'captures.permissions.microphone.bodyWindows'
        : 'captures.permissions.microphone.bodyLinux';

  return (
    <div className="w-full rounded-lg border border-amber-500/30 bg-amber-500/10 px-3.5 py-3">
      <div className="flex items-start gap-3">
        <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-amber-500" />
        <div className="flex-1 min-w-0 space-y-1">
          <p className="text-sm font-medium text-foreground">
            {t('captures.permissions.microphone.title')}
          </p>
          <p className="text-sm text-muted-foreground leading-relaxed">
            <Trans i18nKey={bodyKey} components={{ path: <span /> }} />
          </p>
          <div className="flex items-center gap-2 pt-1.5">
            {canOpenSettings && (
              <Button type="button" size="sm" onClick={openSettings} className="gap-1.5">
                <ExternalLink className="h-3.5 w-3.5" />
                {t('captures.permissions.microphone.openSettings')}
              </Button>
            )}
            {onRetry && (
              <Button type="button" variant="outline" size="sm" onClick={onRetry}>
                {t('captures.permissions.microphone.retry')}
              </Button>
            )}
          </div>
          {showRestartHint && isMacOS && canOpenSettings && (
            <p className="text-xs text-amber-600 dark:text-amber-400 pt-1">
              {t('captures.permissions.microphone.stillMissing')}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
