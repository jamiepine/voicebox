import { AlertCircle, Download, RefreshCw } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { useAutoUpdater } from '@/hooks/useAutoUpdater';
import { usePlatform } from '@/platform/PlatformContext';

export function UpdateStatus() {
  const platform = usePlatform();
  const { status, checkForUpdates, downloadAndInstall, restartAndInstall } = useAutoUpdater(false);
  const [currentVersion, setCurrentVersion] = useState<string>('');

  useEffect(() => {
    platform.metadata
      .getVersion()
      .then(setCurrentVersion)
      .catch(() => setCurrentVersion('Unknown'));
  }, [platform]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>App Updates</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="text-sm font-medium">Current Version</div>
            <div className="text-sm text-muted-foreground">v{currentVersion}</div>
          </div>
          <Button
            onClick={checkForUpdates}
            disabled={status.checking || status.downloading || status.readyToInstall}
            variant="outline"
            size="sm"
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${status.checking ? 'animate-spin' : ''}`} />
            Check for Updates
          </Button>
        </div>

        {status.checking && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <RefreshCw className="h-4 w-4 animate-spin" />
            Checking for updates...
          </div>
        )}

        {status.error && (
          <div className="flex items-center gap-2 text-sm text-destructive">
            <AlertCircle className="h-4 w-4" />
            {status.error}
          </div>
        )}

        {status.available && !status.downloading && !status.readyToInstall && (
          <div className="space-y-3 p-4 border rounded-lg bg-primary/5">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-semibold">Update Available</div>
                <div className="text-sm text-muted-foreground">Version {status.version}</div>
              </div>
              <Badge>New</Badge>
            </div>
            <Button onClick={downloadAndInstall} className="w-full" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Download Update
            </Button>
          </div>
        )}

        {status.downloading && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2">
                <Download className="h-4 w-4" />
                Downloading update...
              </div>
              {status.downloadProgress !== undefined && (
                <span className="text-muted-foreground">{status.downloadProgress}%</span>
              )}
            </div>
            <Progress value={status.downloadProgress} />
            {status.downloadedBytes !== undefined &&
              status.totalBytes !== undefined &&
              status.totalBytes > 0 && (
                <div className="text-xs text-muted-foreground">
                  {(status.downloadedBytes / 1024 / 1024).toFixed(1)} MB /{' '}
                  {(status.totalBytes / 1024 / 1024).toFixed(1)} MB
                </div>
              )}
          </div>
        )}

        {status.readyToInstall && (
          <div className="space-y-3 p-4 border rounded-lg bg-accent/30 border-accent/50">
            <div className="flex items-center gap-2">
              <div>
                <div className="font-semibold">Update Ready to Install</div>
                <div className="text-sm text-muted-foreground">
                  Version {status.version} has been downloaded
                </div>
              </div>
            </div>
            <div className="text-sm text-muted-foreground">
              The app needs to restart to complete the installation. You can do this now or later at
              your convenience.
            </div>
            <Button onClick={restartAndInstall} className="w-full" size="sm">
              <RefreshCw className="h-4 w-4 mr-2" />
              Restart Now
            </Button>
          </div>
        )}

        {!status.available && !status.checking && !status.error && status.checking === false && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            You're up to date
          </div>
        )}
      </CardContent>
    </Card>
  );
}
