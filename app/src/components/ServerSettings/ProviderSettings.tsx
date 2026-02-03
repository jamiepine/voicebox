import { Delete01Icon, Download01Icon } from '@hugeicons/core-free-icons';
import { HugeiconsIcon } from '@hugeicons/react';
import { Icon } from '@iconify/react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useState } from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';

const isMacOS = () => navigator.platform.toLowerCase().includes('mac');

type ProviderType =
  | 'auto'
  | 'apple-mlx'
  | 'bundled-pytorch'
  | 'pytorch-cpu'
  | 'pytorch-cuda'
  | 'remote'
  | 'openai';

export function ProviderSettings() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [downloadingProvider, setDownloadingProvider] = useState<string | null>(null);

  const { data: providersData, isLoading } = useQuery({
    queryKey: ['providers'],
    queryFn: async () => {
      return await apiClient.listProviders();
    },
    refetchInterval: 5000,
  });

  const { data: activeProvider } = useQuery({
    queryKey: ['activeProvider'],
    queryFn: async () => {
      return await apiClient.getActiveProvider();
    },
    refetchInterval: 5000,
  });

  // Callbacks for download completion
  const handleDownloadComplete = useCallback(() => {
    setDownloadingProvider(null);
    queryClient.invalidateQueries({ queryKey: ['providers'] });
  }, [queryClient]);

  const handleDownloadError = useCallback(() => {
    setDownloadingProvider(null);
  }, []);

  // Use progress toast hook for the downloading provider
  useModelDownloadToast({
    modelName: downloadingProvider || '',
    displayName: downloadingProvider || '',
    enabled: !!downloadingProvider,
    onComplete: handleDownloadComplete,
    onError: handleDownloadError,
  });

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [providerToDelete, setProviderToDelete] = useState<string | null>(null);

  const downloadMutation = useMutation({
    mutationFn: async (providerType: string) => {
      return await apiClient.downloadProvider(providerType);
    },
    onSuccess: (_, providerType) => {
      setDownloadingProvider(providerType);
      queryClient.invalidateQueries({ queryKey: ['providers'] });
    },
    onError: (error: Error) => {
      toast({
        title: 'Download failed',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const startMutation = useMutation({
    mutationFn: async (providerType: string) => {
      return await apiClient.startProvider(providerType);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['activeProvider'] });
      toast({
        title: 'Provider started',
        description: 'The provider has been started successfully',
      });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to start provider',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (providerType: string) => {
      return await apiClient.deleteProvider(providerType);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['providers'] });
      toast({
        title: 'Provider deleted',
        description: 'The provider has been deleted successfully',
      });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to delete provider',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const handleDownload = async (providerType: string) => {
    downloadMutation.mutate(providerType);
  };

  const handleStart = async (providerType: string) => {
    startMutation.mutate(providerType);
  };

  const handleDelete = (providerType: string) => {
    setProviderToDelete(providerType);
    setDeleteDialogOpen(true);
  };

  const confirmDelete = () => {
    if (providerToDelete) {
      deleteMutation.mutate(providerToDelete);
      setDeleteDialogOpen(false);
      setProviderToDelete(null);
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>TTS Provider</CardTitle>
          <CardDescription>Choose how Voicebox generates speech</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <Icon icon="svg-spinners:ring-resize" className="h-6 w-6 animate-spin" />
          </div>
        </CardContent>
      </Card>
    );
  }

  const installedProviders = providersData?.installed || [];

  // Determine current active provider
  const currentProvider = activeProvider?.provider;
  console.log('currentProvider', currentProvider);
  const selectedProvider = currentProvider as ProviderType;

  const isStarting = startMutation.isPending;

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>TTS Provider</CardTitle>
          <CardDescription>Choose how Voicebox generates speech.</CardDescription>
        </CardHeader>
        <CardContent className="relative">
          {isStarting && (
            <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-10 rounded-lg">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Icon icon="svg-spinners:ring-resize" className="h-5 w-5" />
                <span>Starting provider...</span>
              </div>
            </div>
          )}
          <RadioGroup
            value={selectedProvider}
            onValueChange={(value) => handleStart(value)}
            disabled={isStarting}
          >
            {/* PyTorch CUDA */}
            <div
              className={`flex items-center justify-between py-2 ${isMacOS() ? 'opacity-50' : ''}`}
            >
              <div className="flex items-center space-x-3 flex-1">
                <RadioGroupItem value="pytorch-cuda" id="cuda" disabled={isMacOS() || isStarting} />
                <Label
                  htmlFor="cuda"
                  className={`flex-1 ${isMacOS() || isStarting ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                >
                  <div className="font-medium">PyTorch CUDA</div>
                  <div className="text-sm text-muted-foreground">
                    NVIDIA GPU-accelerated provider
                  </div>
                </Label>
              </div>
              <div className="flex items-center gap-2">
                {!installedProviders.includes('pytorch-cuda') && (
                  <Button
                    onClick={() => handleDownload('pytorch-cuda')}
                    size="sm"
                    disabled={downloadingProvider === 'pytorch-cuda' || isStarting}
                  >
                    {downloadingProvider === 'pytorch-cuda' ? (
                      <Icon icon="svg-spinners:ring-resize" className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <HugeiconsIcon icon={Download01Icon} size={16} className="h-4 w-4 mr-1" />
                        Download (2.4GB)
                      </>
                    )}
                  </Button>
                )}
                {installedProviders.includes('pytorch-cuda') &&
                  selectedProvider !== 'pytorch-cuda' && (
                    <Button
                      onClick={() => handleStart('pytorch-cuda')}
                      size="sm"
                      variant="outline"
                      disabled={isStarting}
                    >
                      Start
                    </Button>
                  )}
                {installedProviders.includes('pytorch-cuda') && (
                  <Button
                    onClick={() => handleDelete('pytorch-cuda')}
                    size="sm"
                    variant="ghost"
                    disabled={isStarting}
                  >
                    <HugeiconsIcon icon={Delete01Icon} size={16} className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>

            {/* PyTorch CPU */}
            <div className="flex items-center justify-between py-2">
              <div className="flex items-center space-x-3 flex-1">
                <RadioGroupItem value="pytorch-cpu" id="cpu" disabled={isStarting} />
                <Label
                  htmlFor="cpu"
                  className={`flex-1 ${isStarting ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                >
                  <div className="font-medium">PyTorch CPU</div>
                  <div className="text-sm text-muted-foreground">
                    Works on any system, slower inference
                  </div>
                </Label>
              </div>
              <div className="flex items-center gap-2">
                {!installedProviders.includes('pytorch-cpu') && (
                  <Button
                    onClick={() => handleDownload('pytorch-cpu')}
                    size="sm"
                    disabled={downloadingProvider === 'pytorch-cpu' || isStarting}
                  >
                    {downloadingProvider === 'pytorch-cpu' ? (
                      <Icon icon="svg-spinners:ring-resize" className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <HugeiconsIcon icon={Download01Icon} size={16} className="h-4 w-4 mr-1" />
                        Download (300MB)
                      </>
                    )}
                  </Button>
                )}
                {installedProviders.includes('pytorch-cpu') &&
                  selectedProvider !== 'pytorch-cpu' && (
                    <Button
                      onClick={() => handleStart('pytorch-cpu')}
                      size="sm"
                      variant="outline"
                      disabled={isStarting}
                    >
                      Start
                    </Button>
                  )}
                {installedProviders.includes('pytorch-cpu') && (
                  <Button
                    onClick={() => handleDelete('pytorch-cpu')}
                    size="sm"
                    variant="ghost"
                    disabled={isStarting}
                  >
                    <HugeiconsIcon icon={Delete01Icon} size={16} className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>

            {/* MLX bundled (macOS Apple Silicon only) */}
            <div className="flex items-center space-x-3 py-2">
              <RadioGroupItem value="apple-mlx" id="mlx" disabled={isStarting || !isMacOS()} />
              <Label
                htmlFor="mlx"
                className={`flex-1 ${isStarting ? 'cursor-not-allowed' : 'cursor-pointer'}`}
              >
                <div className="font-medium">Apple MLX</div>
                <div className="text-sm text-muted-foreground">
                  {isMacOS()
                    ? 'Bundled with this version, optimized for Apple Silicon'
                    : 'Only available on Apple Silicon'}
                </div>
              </Label>
            </div>

            {/* Remote */}
            <div className="space-y-2 py-2 opacity-50">
              <div className="flex items-center space-x-3">
                <RadioGroupItem value="remote" id="remote" disabled />
                <Label htmlFor="remote" className="flex-1 cursor-not-allowed">
                  <div className="font-medium">Remote Server</div>
                  <div className="text-sm text-muted-foreground">
                    Connect to your own TTS server
                  </div>
                </Label>
              </div>
            </div>

            {/* OpenAI */}
            <div className="space-y-2 py-2 opacity-50">
              <div className="flex items-center space-x-3">
                <RadioGroupItem value="openai" id="openai" disabled />
                <Label htmlFor="openai" className="flex-1 cursor-not-allowed">
                  <div className="font-medium">OpenAI API</div>
                  <div className="text-sm text-muted-foreground">Use OpenAI's TTS API</div>
                </Label>
              </div>
            </div>
          </RadioGroup>
          <p className="text-xs text-muted-foreground mt-5">
            Note: PyTorch and MLX use different versions of the same model. When switching between
            them, you will need to redownload the model.
          </p>
        </CardContent>
      </Card>

      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Provider</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete {providerToDelete}? This will remove the provider
              binary from your system. You can download it again later if needed.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmDelete}
              className="bg-destructive text-destructive-foreground"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
