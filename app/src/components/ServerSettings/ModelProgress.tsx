import { Loader2, XCircle } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import type { ModelProgress as ModelProgressType } from '@/lib/api/types';
import { useServerStore } from '@/stores/serverStore';

interface ModelProgressProps {
  modelName: string;
  displayName: string;
  /** Only connect to SSE when actively downloading - prevents connection exhaustion */
  isDownloading?: boolean;
}

export function ModelProgress({ modelName, displayName, isDownloading = false }: ModelProgressProps) {
  const [progress, setProgress] = useState<ModelProgressType | null>(null);
  const serverUrl = useServerStore((state) => state.serverUrl);

  useEffect(() => {
    // IMPORTANT: Only connect to SSE when this specific model is downloading
    // Opening SSE connections for all models exhausts HTTP/1.1 connection limits (6 per origin)
    // which causes other fetches (like the download trigger) to be queued/blocked
    if (!serverUrl || !isDownloading) {
      return;
    }

    console.log(`[ModelProgress] Connecting SSE for ${modelName}`);

    // Subscribe to progress updates via Server-Sent Events
    const eventSource = new EventSource(`${serverUrl}/models/progress/${modelName}`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as ModelProgressType;
        setProgress(data);

        // Close connection if complete or error
        if (data.status === 'complete' || data.status === 'error') {
          console.log(`[ModelProgress] Download ${data.status} for ${modelName}, closing SSE`);
          eventSource.close();
        }
      } catch (error) {
        console.error('Error parsing progress event:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error(`[ModelProgress] SSE error for ${modelName}:`, error);
      eventSource.close();
    };

    return () => {
      console.log(`[ModelProgress] Cleanup - closing SSE for ${modelName}`);
      eventSource.close();
    };
  }, [serverUrl, modelName, isDownloading]);

  // Don't render if no progress or if complete/error and some time has passed
  if (
    !progress ||
    (progress.status === 'complete' && Date.now() - new Date(progress.timestamp).getTime() > 5000)
  ) {
    return null;
  }

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / k ** i).toFixed(1)} ${sizes[i]}`;
  };

  const getStatusIcon = () => {
    switch (progress.status) {
      case 'error':
        return <XCircle className="h-4 w-4 text-destructive" />;
      case 'downloading':
      case 'extracting':
        return <Loader2 className="h-4 w-4 animate-spin" />;
      default:
        return null;
    }
  };

  const getStatusText = () => {
    switch (progress.status) {
      case 'complete':
        return 'Download complete';
      case 'error':
        return `Error: ${progress.error || 'Unknown error'}`;
      case 'downloading':
        return progress.filename ? `Downloading ${progress.filename}...` : 'Downloading...';
      case 'extracting':
        return 'Extracting...';
      default:
        return 'Processing...';
    }
  };

  return (
    <Card className="mb-4">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          {getStatusIcon()}
          {displayName}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{getStatusText()}</span>
            {progress.total > 0 && (
              <span>
                {formatBytes(progress.current)} / {formatBytes(progress.total)} (
                {progress.progress.toFixed(1)}%)
              </span>
            )}
          </div>
          {progress.total > 0 && <Progress value={progress.progress} className="h-2" />}
        </div>
      </CardContent>
    </Card>
  );
}
