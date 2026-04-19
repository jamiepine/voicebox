import { Loader2, XCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useServerHealth } from '@/lib/hooks/useServer';
import { useServerStore } from '@/stores/serverStore';

export function ServerStatus() {
  const { data: health, isLoading, error } = useServerHealth();
  const serverUrl = useServerStore((state) => state.serverUrl);

  return (
    <Card role="region" aria-label="Server Status" tabIndex={0}>
      <CardHeader>
        <CardTitle>Server Status</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="text-sm text-muted-foreground mb-1">Server URL</div>
          <div className="font-mono text-sm">{serverUrl}</div>
        </div>

        {isLoading ? (
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm">Checking connection...</span>
          </div>
        ) : error ? (
          <div className="flex items-center gap-2">
            <XCircle className="h-4 w-4 text-destructive" />
            <span className="text-sm text-destructive">Connection failed: {error.message}</span>
          </div>
        ) : health ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-sm">Connected</span>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge
                variant={health.model_loaded || health.model_downloaded ? 'default' : 'secondary'}
              >
                {health.model_loaded || health.model_downloaded ? 'Model Ready' : 'No Model'}
              </Badge>
              <Badge variant={health.gpu_available ? 'default' : 'secondary'}>
                GPU: {health.gpu_available ? 'Available' : 'Not Available'}
              </Badge>
              {health.vram_used_mb && (
                <Badge variant="outline">VRAM: {health.vram_used_mb.toFixed(0)} MB</Badge>
              )}
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
