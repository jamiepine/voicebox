import { Loader2, XCircle } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import type { ModelProgress as ModelProgressType } from '@/lib/api/types';
import { useServerStore } from '@/stores/serverStore';

interface ModelProgressProps {
  modelName: string;
  displayName: string;
}

export function ModelProgress({ modelName, displayName }: ModelProgressProps) {
  const [progress, setProgress] = useState<ModelProgressType | null>(null);
  const serverUrl = useServerStore((state) => state.serverUrl);

  useEffect(() => {
    if (!serverUrl) return;

    // Subscribe to progress updates via Server-Sent Events
    const eventSource = new EventSource(`${serverUrl}/models/progress/${modelName}`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as ModelProgressType;
        setProgress(data);

        // Close connection if complete or error
        if (data.status === 'complete' || data.status === 'error') {
          eventSource.close();
        }
      } catch (error) {
        console.error('Error parsing progress event:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE error:', error);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [serverUrl, modelName]);

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
