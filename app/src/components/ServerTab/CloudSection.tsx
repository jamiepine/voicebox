import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Cloud, Copy, Loader2, RefreshCw, ShieldCheck } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { SettingRow, SettingSection } from './SettingRow';

// "Log in with browser" device pairing. The backend opens the system browser
// and completes the code exchange; here we just kick it off and poll status
// until the link goes live. The API key never touches the frontend.
//
// Once linked, the encrypted-backup rows drive the sync identity flows: this
// device registers as an encryption device, and either mints the account's
// master key (first device — the recovery phrase is force-displayed exactly
// once) or waits to be provisioned by another device / restored from the
// phrase. All crypto happens in the local backend; this UI only ever sees the
// phrase, and only at mint time.
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
      queryClient.invalidateQueries({ queryKey: ['cloud-sync-status'] });
      toast({
        title: 'Disconnected',
        description: 'This device is no longer linked. The key stays valid until revoked in your account.',
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
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
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

      {connected && <BackupRows />}

      {connected && (
        <SettingRow
          title="Manage"
          description="Revoke this device, add API keys, or manage billing from your account."
        >
          <a
            className="text-sm text-accent hover:underline"
            href="https://voicebox.sh/account"
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

function BackupRows() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [phrase, setPhrase] = useState<string | null>(null);
  const [phraseCopied, setPhraseCopied] = useState(false);
  const [restoreInput, setRestoreInput] = useState('');

  const { data: sync } = useQuery({
    queryKey: ['cloud-sync-status'],
    queryFn: () => apiClient.getCloudSyncStatus(),
  });

  const refreshSync = () => queryClient.invalidateQueries({ queryKey: ['cloud-sync-status'] });

  const setup = useMutation({
    mutationFn: () => apiClient.setupCloudSync(),
    onSuccess: (result) => {
      refreshSync();
      if (result.recovery_phrase) {
        // First device: the phrase exists only in this response. Force-display
        // it; the dialog can only be dismissed by confirming.
        setPhraseCopied(false);
        setPhrase(result.recovery_phrase);
      }
    },
    onError: (error: Error) =>
      toast({ title: 'Could not enable backup', description: error.message, variant: 'destructive' }),
  });

  const adopt = useMutation({
    mutationFn: () => apiClient.adoptCloudSync(),
    onSuccess: (result) => {
      refreshSync();
      if (result.status === 'ready') {
        toast({ title: 'Backup enabled', description: 'This device received its encryption key.' });
      } else {
        toast({
          title: 'Not approved yet',
          description: 'Approve this device from another synced device, then check again.',
        });
      }
    },
    onError: (error: Error) =>
      toast({ title: 'Could not check', description: error.message, variant: 'destructive' }),
  });

  const restore = useMutation({
    mutationFn: (recoveryPhrase: string) => apiClient.restoreCloudSync(recoveryPhrase),
    onSuccess: () => {
      refreshSync();
      setRestoreInput('');
      toast({ title: 'Backup restored', description: 'Your encryption key was recovered. Syncing is ready.' });
    },
    onError: (error: Error) =>
      toast({ title: 'Could not restore', description: error.message, variant: 'destructive' }),
  });

  const run = useMutation({
    mutationFn: () => apiClient.runCloudSync(),
    onSuccess: (report) => {
      refreshSync();
      const pushed = report.pushed + report.pushed_deletes;
      const pulled = report.pulled + report.pulled_deletes;
      toast({
        title: 'Sync complete',
        description:
          pushed === 0 && pulled === 0
            ? 'Everything is already up to date.'
            : `Backed up ${pushed} ${pushed === 1 ? 'item' : 'items'}, received ${pulled}.`,
      });
    },
    onError: (error: Error) =>
      toast({ title: 'Sync failed', description: error.message, variant: 'destructive' }),
  });

  const state = sync?.status ?? 'unregistered';

  return (
    <>
      {state === 'unregistered' && (
        <SettingRow
          title="Encrypted backup"
          description="Back up captures, generations, and voice profiles — encrypted on this device before anything is uploaded."
          action={
            <Button disabled={setup.isPending} onClick={() => setup.mutate()} size="sm">
              {setup.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <>
                  <ShieldCheck className="h-3.5 w-3.5 mr-1.5" />
                  Enable backup
                </>
              )}
            </Button>
          }
        />
      )}

      {state === 'awaiting_provision' && (
        <>
          <SettingRow
            title="Waiting for encryption key"
            description="This account already has a backup. Approve this device from another synced device, or restore with your recovery phrase below."
            action={
              <Button disabled={adopt.isPending} onClick={() => adopt.mutate()} size="sm" variant="outline">
                {adopt.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <>
                    <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
                    Check again
                  </>
                )}
              </Button>
            }
          />
          <SettingRow
            title="Restore with recovery phrase"
            description="The 12 words you wrote down when you first enabled backup."
            action={
              <div className="flex items-center gap-2">
                <Input
                  className="w-72"
                  onChange={(e) => setRestoreInput(e.target.value)}
                  placeholder="correct horse battery staple …"
                  value={restoreInput}
                />
                <Button
                  disabled={restore.isPending || restoreInput.trim().split(/\s+/).length < 12}
                  onClick={() => restore.mutate(restoreInput)}
                  size="sm"
                >
                  {restore.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : 'Restore'}
                </Button>
              </div>
            }
          />
        </>
      )}

      {state === 'ready' && (
        <SettingRow
          title="Encrypted backup"
          description="On — content is encrypted on this device before upload. The server can never read it."
          action={
            <Button disabled={run.isPending} onClick={() => run.mutate()} size="sm">
              {run.isPending ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  Syncing…
                </>
              ) : (
                <>
                  <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
                  Sync now
                </>
              )}
            </Button>
          }
        />
      )}

      <RecoveryPhraseDialog
        copied={phraseCopied}
        onConfirm={() => setPhrase(null)}
        onCopy={() => {
          if (phrase) {
            navigator.clipboard?.writeText(phrase);
            setPhraseCopied(true);
          }
        }}
        phrase={phrase}
      />
    </>
  );
}

// The one moment the recovery phrase exists outside the backend. Deliberately
// hard to dismiss: no overlay/escape close, only the explicit confirmation.
function RecoveryPhraseDialog({
  phrase,
  copied,
  onCopy,
  onConfirm,
}: {
  phrase: string | null;
  copied: boolean;
  onCopy: () => void;
  onConfirm: () => void;
}) {
  return (
    <Dialog onOpenChange={() => {}} open={phrase !== null}>
      <DialogContent className="max-w-lg [&>button]:hidden">
        <DialogHeader>
          <DialogTitle>Write down your recovery phrase</DialogTitle>
          <DialogDescription>
            These 12 words are the only way to restore your backup if you lose all your devices.
            They are never sent to the server and will not be shown again.
          </DialogDescription>
        </DialogHeader>
        <div className="grid grid-cols-3 gap-2 py-2">
          {(phrase ?? '').split(' ').map((word, index) => (
            <div
              className="rounded-md border border-border bg-muted/40 px-2.5 py-1.5 text-sm"
              // Position is the identity here — BIP39 phrases can repeat words.
              // biome-ignore lint/suspicious/noArrayIndexKey: static list, never reordered
              key={index}
            >
              <span className="mr-1.5 text-muted-foreground tabular-nums">{index + 1}.</span>
              {word}
            </div>
          ))}
        </div>
        <DialogFooter className="gap-2 sm:justify-between">
          <Button onClick={onCopy} size="sm" type="button" variant="outline">
            <Copy className="h-3.5 w-3.5 mr-1.5" />
            {copied ? 'Copied' : 'Copy'}
          </Button>
          <Button onClick={onConfirm} size="sm" type="button">
            I've written it down
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
