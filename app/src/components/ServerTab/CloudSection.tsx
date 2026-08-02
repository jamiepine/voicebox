import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Cloud, Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { SettingRow, SettingSection } from './SettingRow';

// "Log in with browser" device pairing. The backend opens the system browser
// and completes the code exchange; here we just kick it off and poll status
// until the link goes live. The API key never touches the frontend.
export function CloudSection() {
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
        title: 'Connected to Voicebox Cloud',
        description: `Linked as ${status?.device_name ?? 'this device'}.`,
      });
    }
  }, [connected, polling, status?.device_name, toast]);

  // Give up after two minutes so an abandoned browser flow doesn't leave the
  // button stuck on "Waiting for browser…". The backend state stays valid for
  // ten, so the user can simply start again.
  useEffect(() => {
    if (!polling) return;
    const timeoutId = window.setTimeout(() => {
      setPolling(false);
      toast({
        title: 'Sign-in timed out',
        description: 'The browser sign-in was not completed. Try again.',
        variant: 'destructive',
      });
    }, 120_000);
    return () => window.clearTimeout(timeoutId);
  }, [polling, toast]);

  const startLogin = useMutation({
    mutationFn: () => apiClient.startCloudLogin(),
    onSuccess: () => {
      setPolling(true);
      toast({
        title: 'Continue in your browser',
        description: 'Authorize this device, then return here.',
      });
    },
    onError: (error: Error) =>
      toast({
        title: 'Could not start sign-in',
        description: error.message,
        variant: 'destructive',
      }),
  });

  const disconnect = useMutation({
    mutationFn: () => apiClient.disconnectCloud(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cloud-status'] });
      toast({
        title: 'Disconnected',
        description:
          'This device is no longer linked. The key stays valid until revoked in your account.',
      });
    },
    onError: (error: Error) =>
      toast({ title: 'Could not disconnect', description: error.message, variant: 'destructive' }),
  });

  const busy = startLogin.isPending || polling;

  return (
    <SettingSection
      title="Voicebox Cloud"
      description="End-to-end encrypted backup & sync across your devices."
    >
      <SettingRow
        title={connected ? 'Connected' : 'Account'}
        description={
          connected
            ? `Linked as ${status?.device_name ?? 'this device'}${
                status?.key_prefix ? ` · ${status.key_prefix}…` : ''
              }`
            : 'Log in to back up and sync your captures and generations.'
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
                  Disconnecting…
                </>
              ) : (
                'Disconnect'
              )}
            </Button>
          ) : (
            <Button disabled={busy} onClick={() => startLogin.mutate()} size="sm">
              {busy ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  {polling ? 'Waiting for browser…' : 'Opening…'}
                </>
              ) : (
                <>
                  <Cloud className="h-3.5 w-3.5 mr-1.5" />
                  Log in with browser
                </>
              )}
            </Button>
          )
        }
      />

      {connected && (
        <SettingRow
          title="Manage"
          description="Revoke this device, add API keys, or manage billing from your account."
        >
          <a
            className="text-sm text-accent hover:underline"
            href={status?.dashboard_url ?? 'https://voicebox.sh/account'}
            rel="noopener noreferrer"
            target="_blank"
          >
            Open account dashboard ↗
          </a>
        </SettingRow>
      )}
    </SettingSection>
  );
}
