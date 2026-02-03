import { Folder01Icon, FolderOpenIcon } from '@hugeicons/core-free-icons';
import { HugeiconsIcon } from '@hugeicons/react';
import { Icon } from '@iconify/react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useSystemFolders } from '@/lib/hooks/useSystemFolders';
import { usePlatform } from '@/platform/PlatformContext';

interface FolderRowProps {
  label: string;
  description: string;
  path: string | undefined;
  isLoading: boolean;
  canOpen: boolean;
  onOpen: () => void;
}

function FolderRow({ label, description, path, isLoading, canOpen, onOpen }: FolderRowProps) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm font-medium">{label}</div>
          <div className="text-xs text-muted-foreground">{description}</div>
        </div>
        {canOpen && path && (
          <Button
            variant="outline"
            size="sm"
            onClick={onOpen}
            disabled={isLoading || !path}
            className="shrink-0"
          >
            <HugeiconsIcon icon={FolderOpenIcon} size={16} className="h-4 w-4 mr-2" />
            Open
          </Button>
        )}
      </div>
      <Input
        value={isLoading ? 'Loading...' : path || 'Not available'}
        readOnly
        className="font-mono text-xs text-muted-foreground select-all cursor-text"
      />
    </div>
  );
}

export function DataFolders() {
  const { data: folders, isLoading, error } = useSystemFolders();
  const platform = usePlatform();
  const isTauri = platform.metadata.isTauri;

  const handleOpenFolder = async (path: string | undefined) => {
    if (!path) return;
    const success = await platform.filesystem.openFolder(path);
    if (!success && isTauri) {
      console.error('Failed to open folder:', path);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <HugeiconsIcon icon={Folder01Icon} size={20} className="h-5 w-5" />
          Data Folders
        </CardTitle>
        <CardDescription>
          {isTauri
            ? 'Click "Open" to view folders in your file explorer, or copy the paths below.'
            : 'These are the server-side folder paths where your data is stored.'}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error ? (
          <div className="flex items-center gap-2 text-sm text-destructive">
            <Icon icon="lucide:alert-circle" className="h-4 w-4" />
            <span>Failed to load folder paths: {error.message}</span>
          </div>
        ) : (
          <>
            <FolderRow
              label="App Data"
              description="Voices, generations, and app database"
              path={folders?.data_dir}
              isLoading={isLoading}
              canOpen={isTauri}
              onOpen={() => handleOpenFolder(folders?.data_dir)}
            />
            <FolderRow
              label="Models"
              description="Downloaded AI models from HuggingFace Hub"
              path={folders?.models_dir}
              isLoading={isLoading}
              canOpen={isTauri}
              onOpen={() => handleOpenFolder(folders?.models_dir)}
            />
            <FolderRow
              label="Providers"
              description="External TTS provider binaries (PyTorch CPU/CUDA)"
              path={folders?.providers_dir}
              isLoading={isLoading}
              canOpen={isTauri}
              onOpen={() => handleOpenFolder(folders?.providers_dir)}
            />
          </>
        )}
      </CardContent>
    </Card>
  );
}
