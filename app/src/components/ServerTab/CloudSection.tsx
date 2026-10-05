import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Cloud, Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { SettingRow, SettingSection } from './SettingRow';

// "Log in with browser" device pairing. The backend opens the system browser
// and completes the code exchange; here we just kick it off and poll status
// until the link goes live. The API key never touches the frontend.
export function CloudSection() {
  const { t } = useTranslation();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [polling, setPolling] = useState(false);

  const { data: status } = useQuery({
    queryKey: ['cloud-status'],
    queryFn: () => apiClient.getCloudStatus(),
    refetchInterval: polling ? 2000 : false,
  });

  const connected = status?.connected ?? false;

  // Once the browser flow completes, stop polling and celebrate.
  useEffect(() => {
    if (connected && polling) {
      setPolling(false);
      toast({
        title: t('settings.general.cloud.connectedToast.title'),
        description: t('settings.general.cloud.connectedToast.description', {
          name: status?.device_name ?? t('settings.general.cloud.account.thisDevice'),
        }),
      });
    }
  }, [connected, polling, status?.device_name, toast, t]);

  // Give up after two minutes so an abandoned browser flow doesn't leave the
  // button stuck on "Waiting for browser…". The backend state stays valid for
  // ten, so the user can simply start again.
  useEffect(() => {
    if (!polling) return;
    const timeoutId = window.setTimeout(() => {
      setPolling(false);
      toast({
        title: t('settings.general.cloud.signInTimedOut.title'),
        description: t('settings.general.cloud.signInTimedOut.description'),
        variant: 'destructive',
      });
    }, 120_000);
    return () => window.clearTimeout(timeoutId);
  }, [polling, toast, t]);

  const startLogin = useMutation({
    mutationFn: () => apiClient.startCloudLogin(),
    onSuccess: () => {
      setPolling(true);
      toast({
        title: t('settings.general.cloud.continueInBrowser.title'),
        description: t('settings.general.cloud.continueInBrowser.description'),
      });
    },
    onError: (error: Error) =>
      toast({
        title: t('settings.general.cloud.signInFailedTitle'),
        description: error.message,
        variant: 'destructive',
      }),
  });

  const disconnect = useMutation({
    mutationFn: () => apiClient.disconnectCloud(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cloud-status'] });
      toast({
        title: t('settings.general.cloud.disconnectedToast.title'),
        description: t('settings.general.cloud.disconnectedToast.description'),
      });
    },
    onError: (error: Error) =>
      toast({
        title: t('settings.general.cloud.disconnectFailedTitle'),
        description: error.message,
        variant: 'destructive',
      }),
  });

  const busy = startLogin.isPending || polling;

  return (
    <SettingSection
      title={t('settings.general.cloud.title')}
      description={t('settings.general.cloud.description')}
    >
      <SettingRow
        title={
          connected
            ? t('settings.general.cloud.account.connected')
            : t('settings.general.cloud.account.account')
        }
        description={
          connected
            ? t('settings.general.cloud.account.linkedAs', {
                name: status?.device_name ?? t('settings.general.cloud.account.thisDevice'),
                keyPrefix: status?.key_prefix ? ` · ${status.key_prefix}…` : '',
              })
            : t('settings.general.cloud.account.logInHint')
        }
        action={
          connected ? (
            <Button
              disabled={disconnect.isPending}
              onClick={() => disconnect.mutate()}
              size="sm"
              variant="outline"
            >
              {disconnect.isPending ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  {t('settings.general.cloud.actions.disconnecting')}
                </>
              ) : (
                t('settings.general.cloud.actions.disconnect')
              )}
            </Button>
          ) : (
            <Button disabled={busy} onClick={() => startLogin.mutate()} size="sm">
              {busy ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  {polling
                    ? t('settings.general.cloud.actions.waitingForBrowser')
                    : t('settings.general.cloud.actions.opening')}
                </>
              ) : (
                <>
                  <Cloud className="h-3.5 w-3.5 mr-1.5" />
                  {t('settings.general.cloud.actions.logInWithBrowser')}
                </>
              )}
            </Button>
          )
        }
      />

      {connected && (
        <SettingRow
          title={t('settings.general.cloud.manage.title')}
          description={t('settings.general.cloud.manage.description')}
        >
          <a
            className="text-sm text-accent hover:underline"
            href={status?.dashboard_url ?? 'https://voicebox.sh/account'}
            rel="noopener noreferrer"
            target="_blank"
          >
            {t('settings.general.cloud.manage.openDashboard')}
          </a>
        </SettingRow>
      )}
    </SettingSection>
  );
}
