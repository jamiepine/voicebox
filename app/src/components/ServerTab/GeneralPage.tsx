import { zodResolver } from '@hookform/resolvers/zod';
import { AlertCircle, ArrowUpRight, Book, Download, Loader2, RefreshCw } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { Button } from '@/components/ui/button';
import { Form, FormControl, FormField, FormItem, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Toggle } from '@/components/ui/toggle';
import { useToast } from '@/components/ui/use-toast';
import { useAutoUpdater } from '@/hooks/useAutoUpdater';
import { useServerHealth } from '@/lib/hooks/useServer';
import { usePlatform } from '@/platform/PlatformContext';
import { useServerStore } from '@/stores/serverStore';
import { SettingRow, SettingSection } from './SettingRow';

const connectionSchema = z.object({
  serverUrl: z.string().url('Please enter a valid URL'),
});

type ConnectionFormValues = z.infer<typeof connectionSchema>;

export function GeneralPage() {
  const platform = usePlatform();
  const serverUrl = useServerStore((state) => state.serverUrl);
  const setServerUrl = useServerStore((state) => state.setServerUrl);
  const keepServerRunningOnClose = useServerStore((state) => state.keepServerRunningOnClose);
  const setKeepServerRunningOnClose = useServerStore((state) => state.setKeepServerRunningOnClose);
  const mode = useServerStore((state) => state.mode);
  const setMode = useServerStore((state) => state.setMode);
  const { toast } = useToast();
  const { data: health, isLoading, error: healthError } = useServerHealth();

  const form = useForm<ConnectionFormValues>({
    resolver: zodResolver(connectionSchema),
    defaultValues: { serverUrl },
  });

  useEffect(() => {
    form.reset({ serverUrl });
  }, [serverUrl, form]);

  const { isDirty } = form.formState;

  function onSubmit(data: ConnectionFormValues) {
    setServerUrl(data.serverUrl);
    form.reset(data);
    toast({
      title: 'Server URL updated',
      description: `Connected to ${data.serverUrl}`,
    });
  }

  return (
    <div className="space-y-8 max-w-2xl">
      <div className="grid grid-cols-2 gap-3">
        <a
          href="https://docs.voicebox.sh"
          target="_blank"
          rel="noopener noreferrer"
          className="group flex items-center gap-3 rounded-lg border border-border/60 p-4 transition-colors hover:bg-muted/50"
        >
          <Book className="h-5 w-5 shrink-0 text-accent" strokeWidth={2.5} />
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium">Read the Docs</div>
            <div className="text-xs text-muted-foreground">docs.voicebox.sh</div>
          </div>
          <ArrowUpRight className="h-4 w-4 text-muted-foreground/40 group-hover:text-muted-foreground transition-colors" />
        </a>
        <a
          href="https://discord.gg/StkzQasqPS"
          target="_blank"
          rel="noopener noreferrer"
          className="group flex items-center gap-3 rounded-lg border border-border/60 p-4 transition-colors hover:bg-muted/50"
        >
          <svg
            className="h-5 w-5 shrink-0 text-accent"
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.095 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.095 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z" />
          </svg>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium">Join the Discord</div>
            <div className="text-xs text-muted-foreground">Get help & share voices</div>
          </div>
          <ArrowUpRight className="h-4 w-4 text-muted-foreground/40 group-hover:text-muted-foreground transition-colors" />
        </a>
      </div>

      <SettingSection>
        <SettingRow
          title="Server URL"
          description="The address of your voicebox backend server."
          action={
            <ConnectionStatus health={health} isLoading={isLoading} healthError={healthError} />
          }
        >
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="flex gap-2">
              <FormField
                control={form.control}
                name="serverUrl"
                render={({ field }) => (
                  <FormItem className="flex-1">
                    <FormControl>
                      <Input placeholder="http://127.0.0.1:17493" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              {isDirty && (
                <Button type="submit" size="sm">
                  Save
                </Button>
              )}
            </form>
          </Form>
        </SettingRow>

        <SettingRow
          title="Keep server running when app closes"
          description="The server will continue running in the background after closing the app."
          htmlFor="keepServerRunning"
          action={
            <Toggle
              id="keepServerRunning"
              checked={keepServerRunningOnClose}
              onCheckedChange={(checked: boolean) => {
                setKeepServerRunningOnClose(checked);
                platform.lifecycle.setKeepServerRunning(checked).catch((error) => {
                  console.error('Failed to sync setting to Rust:', error);
                  setKeepServerRunningOnClose(!checked);
                  toast({
                    title: 'Failed to update setting',
                    description: 'Could not sync setting to backend.',
                    variant: 'destructive',
                  });
                  return;
                });
                toast({
                  title: 'Setting updated',
                  description: checked
                    ? 'Server will continue running when app closes'
                    : 'Server will stop when app closes',
                });
              }}
            />
          }
        />

        {platform.metadata.isTauri && (
          <SettingRow
            title="Allow network access"
            description="Makes the server accessible from other devices on your network. Restart the app after changing."
            htmlFor="allowNetworkAccess"
            action={
              <Toggle
                id="allowNetworkAccess"
                checked={mode === 'remote'}
                onCheckedChange={(checked: boolean) => {
                  setMode(checked ? 'remote' : 'local');
                  toast({
                    title: 'Setting updated',
                    description: checked
                      ? 'Network access enabled. Restart the app to apply.'
                      : 'Network access disabled. Restart the app to apply.',
                  });
                }}
              />
            }
          />
        )}
      </SettingSection>

      <ApiReferenceCard serverUrl={serverUrl} />

      {platform.metadata.isTauri && <UpdatesSection />}
    </div>
  );
}

function ConnectionStatus({
  health,
  isLoading,
  healthError,
}: {
  health: ReturnType<typeof useServerHealth>['data'];
  isLoading: boolean;
  healthError: ReturnType<typeof useServerHealth>['error'];
}) {
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 rounded-full border border-border/60 px-3 py-1">
        <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
        <span className="text-xs text-muted-foreground">Connecting</span>
      </div>
    );
  }
  if (healthError) {
    return (
      <div className="flex items-center gap-2 rounded-full border border-destructive/30 px-3 py-1">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full rounded-full bg-destructive/40" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-destructive" />
        </span>
        <span className="text-xs text-destructive">Offline</span>
      </div>
    );
  }
  if (health) {
    return (
      <div className="flex items-center gap-2 rounded-full border border-accent/30 px-3 py-1">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent/60" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-accent shadow-[0_0_6px_1px_hsl(var(--accent)/0.5)]" />
        </span>
        <span className="text-xs text-muted-foreground">Online</span>
      </div>
    );
  }
  return null;
}

function UpdatesSection() {
  const platform = usePlatform();
  const { status, checkForUpdates, downloadAndInstall, restartAndInstall } = useAutoUpdater(false);
  const [currentVersion, setCurrentVersion] = useState<string>('');
  const isDev = !import.meta.env?.PROD;

  useEffect(() => {
    platform.metadata
      .getVersion()
      .then(setCurrentVersion)
      .catch(() => setCurrentVersion('Unknown'));
  }, [platform]);

  return (
    <SettingSection title="App Updates" description={`v${currentVersion}${isDev ? ' (dev)' : ''}`}>
      {isDev ? (
        <SettingRow
          title="Development mode"
          description="Auto-updates are disabled in development mode."
        />
      ) : (
        <>
          <SettingRow
            title="Check for updates"
            description={
              status.available
                ? `Version ${status.version} available`
                : status.checking
                  ? 'Checking...'
                  : "You're up to date"
            }
            action={
              <Button
                onClick={checkForUpdates}
                disabled={status.checking || status.downloading || status.readyToInstall}
                variant="outline"
                size="sm"
              >
                <RefreshCw
                  className={`h-3.5 w-3.5 mr-1.5 ${status.checking ? 'animate-spin' : ''}`}
                />
                Check
              </Button>
            }
          />

          {status.error && (
            <SettingRow title="Update error">
              <div className="flex items-center gap-2 text-sm text-destructive">
                <AlertCircle className="h-4 w-4" />
                {status.error}
              </div>
            </SettingRow>
          )}

          {status.available && !status.downloading && !status.readyToInstall && (
            <SettingRow
              title={`Update to ${status.version}`}
              description="Download and install the latest version."
              action={
                <Button onClick={downloadAndInstall} size="sm">
                  <Download className="h-3.5 w-3.5 mr-1.5" />
                  Download
                </Button>
              }
            />
          )}

          {status.downloading && (
            <SettingRow title="Downloading update...">
              <div className="space-y-1.5">
                <Progress value={status.downloadProgress} />
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  {status.downloadedBytes !== undefined &&
                  status.totalBytes !== undefined &&
                  status.totalBytes > 0 ? (
                    <span>
                      {(status.downloadedBytes / 1024 / 1024).toFixed(1)} MB /{' '}
                      {(status.totalBytes / 1024 / 1024).toFixed(1)} MB
                    </span>
                  ) : (
                    <span />
                  )}
                  {status.downloadProgress !== undefined && <span>{status.downloadProgress}%</span>}
                </div>
              </div>
            </SettingRow>
          )}

          {status.readyToInstall && (
            <SettingRow
              title="Update ready to install"
              description={`Version ${status.version} has been downloaded. Restart to complete.`}
              action={
                <Button onClick={restartAndInstall} size="sm">
                  <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
                  Restart Now
                </Button>
              }
            />
          )}
        </>
      )}
    </SettingSection>
  );
}

const API_ENDPOINTS = [
  { method: 'POST', path: '/generate', label: 'Generate speech' },
  { method: 'GET', path: '/health', label: 'Server status' },
  { method: 'GET', path: '/profiles', label: 'List voices' },
  { method: 'GET', path: '/history', label: 'Past generations' },
];

function ApiReferenceCard({ serverUrl }: { serverUrl: string }) {
  return (
    <div className="rounded-lg border border-border/60 p-4 space-y-3">
      <div>
        <h3 className="text-sm font-medium">API Access</h3>
        <p className="text-sm text-muted-foreground">
          Integrate Voicebox into your workflow via the REST API at{' '}
          <code className="text-xs bg-muted px-1 py-0.5 rounded font-mono">{serverUrl}</code>
        </p>
      </div>
      <div className="space-y-1">
        {API_ENDPOINTS.map((ep) => (
          <div key={ep.path} className="flex items-center gap-2.5 py-1">
            <span
              className={`text-[10px] font-mono font-semibold w-9 text-center rounded px-1 py-px ${
                ep.method === 'POST' ? 'bg-accent/10 text-accent' : 'bg-muted text-muted-foreground'
              }`}
            >
              {ep.method}
            </span>
            <code className="text-xs font-mono text-muted-foreground">{ep.path}</code>
            <span className="text-xs text-muted-foreground/50 ml-auto">{ep.label}</span>
          </div>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        <a
          href={`${serverUrl}/docs`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-accent hover:underline"
        >
          View the full API reference
        </a>
      </p>
    </div>
  );
}
