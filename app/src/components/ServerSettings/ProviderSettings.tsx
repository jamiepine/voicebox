import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Download, Loader2, Trash2 } from 'lucide-react';
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
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import { useModelDownloadToast } from '@/lib/hooks/useModelDownloadToast';

const isMacOS = () => navigator.platform.toLowerCase().includes('mac');

type ProviderType = 'auto' | 'bundled-mlx' | 'bundled-pytorch' | 'pytorch-cpu' | 'pytorch-cuda' | 'remote' | 'openai';

export function ProviderSettings() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [selectedProvider, setSelectedProvider] = useState<ProviderType>('auto');
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
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        </CardContent>
      </Card>
    );
  }

  const installedProviders = providersData?.installed || [];

  // Determine current active provider
  const currentProvider = activeProvider?.provider || 'auto';

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>TTS Provider</CardTitle>
          <CardDescription>Choose how Voicebox generates speech</CardDescription>
        </CardHeader>
        <CardContent>
          <RadioGroup
            value={selectedProvider}
            onValueChange={(value) => setSelectedProvider(value as ProviderType)}
          >
            {/* Auto-detect */}
            <div className="flex items-center space-x-2 py-2">
              <RadioGroupItem value="auto" id="auto" />
              <Label htmlFor="auto" className="flex-1 cursor-pointer">
                <div className="font-medium">Auto-detect (Recommended)</div>
                <div className="text-sm text-muted-foreground">
                  Automatically choose the best available provider
                </div>
              </Label>
              {currentProvider === 'auto' && (
                <Badge variant="outline" className="ml-2">
                  Active
                </Badge>
              )}
            </div>

            {/* PyTorch CUDA */}
            <div className="flex items-center justify-between py-2">
              <div className="flex items-center space-x-2 flex-1">
                <RadioGroupItem value="pytorch-cuda" id="cuda" />
                <Label htmlFor="cuda" className="flex-1 cursor-pointer">
                  <div className="font-medium">PyTorch CUDA (NVIDIA GPU)</div>
                  <div className="text-sm text-muted-foreground">
                    4-5x faster inference on NVIDIA GPUs
                  </div>
                </Label>
              </div>
              <div className="flex items-center gap-2">
                {currentProvider === 'pytorch-cuda' && (
                  <Badge variant="outline">Active</Badge>
                )}
                {!installedProviders.includes('pytorch-cuda') && (
                  <Button
                    onClick={() => handleDownload('pytorch-cuda')}
                    size="sm"
                    disabled={downloadingProvider === 'pytorch-cuda'}
                  >
                    {downloadingProvider === 'pytorch-cuda' ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <Download className="h-4 w-4 mr-1" />
                        Download (2.4GB)
                      </>
                    )}
                  </Button>
                )}
                {installedProviders.includes('pytorch-cuda') && selectedProvider !== 'pytorch-cuda' && (
                  <Button
                    onClick={() => handleStart('pytorch-cuda')}
                    size="sm"
                    variant="outline"
                  >
                    Start
                  </Button>
                )}
                {installedProviders.includes('pytorch-cuda') && (
                  <Button
                    onClick={() => handleDelete('pytorch-cuda')}
                    size="sm"
                    variant="ghost"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>

            {/* PyTorch CPU (Windows/Linux only) */}
            {!isMacOS() && (
              <div className="flex items-center justify-between py-2">
                <div className="flex items-center space-x-2 flex-1">
                  <RadioGroupItem value="pytorch-cpu" id="cpu" />
                  <Label htmlFor="cpu" className="flex-1 cursor-pointer">
                    <div className="font-medium">PyTorch CPU</div>
                    <div className="text-sm text-muted-foreground">
                      Works on any system, slower inference
                    </div>
                  </Label>
                </div>
                <div className="flex items-center gap-2">
                  {currentProvider === 'pytorch-cpu' && (
                    <Badge variant="outline">Active</Badge>
                  )}
                  {!installedProviders.includes('pytorch-cpu') && (
                    <Button
                      onClick={() => handleDownload('pytorch-cpu')}
                      size="sm"
                      disabled={downloadingProvider === 'pytorch-cpu'}
                    >
                      {downloadingProvider === 'pytorch-cpu' ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <>
                          <Download className="h-4 w-4 mr-1" />
                          Download (300MB)
                        </>
                      )}
                    </Button>
                  )}
                  {installedProviders.includes('pytorch-cpu') && selectedProvider !== 'pytorch-cpu' && (
                    <Button
                      onClick={() => handleStart('pytorch-cpu')}
                      size="sm"
                      variant="outline"
                    >
                      Start
                    </Button>
                  )}
                  {installedProviders.includes('pytorch-cpu') && (
                    <Button
                      onClick={() => handleDelete('pytorch-cpu')}
                      size="sm"
                      variant="ghost"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
            )}

            {/* MLX bundled (macOS only) */}
            {isMacOS() && (
              <div className="p-3 bg-muted rounded-md">
                <div className="text-sm">
                  <div className="font-medium flex items-center gap-2">
                    MLX (Apple Silicon)
                    {currentProvider === 'bundled-mlx' && (
                      <Badge variant="outline">Active</Badge>
                    )}
                  </div>
                  <div className="text-muted-foreground mt-1">
                    Bundled with the app - optimized for M1/M2/M3 chips
                  </div>
                </div>
              </div>
            )}

            {/* Remote */}
            <div className="space-y-2 py-2">
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="remote" id="remote" />
                <Label htmlFor="remote" className="flex-1 cursor-pointer">
                  <div className="font-medium">Remote Server</div>
                  <div className="text-sm text-muted-foreground">
                    Connect to your own TTS server
                  </div>
                </Label>
              </div>
              {selectedProvider === 'remote' && (
                <div className="ml-6">
                  <input
                    type="text"
                    placeholder="http://your-server:8000"
                    className="w-full px-3 py-2 border rounded-md"
                    disabled
                  />
                  <div className="text-xs text-muted-foreground mt-1">
                    Remote provider support coming soon
                  </div>
                </div>
              )}
            </div>

            {/* OpenAI */}
            <div className="space-y-2 py-2">
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="openai" id="openai" />
                <Label htmlFor="openai" className="flex-1 cursor-pointer">
                  <div className="font-medium">OpenAI API</div>
                  <div className="text-sm text-muted-foreground">
                    Use OpenAI's TTS API (requires API key)
                  </div>
                </Label>
              </div>
              {selectedProvider === 'openai' && (
                <div className="ml-6">
                  <input
                    type="password"
                    placeholder="sk-..."
                    className="w-full px-3 py-2 border rounded-md"
                    disabled
                  />
                  <div className="text-xs text-muted-foreground mt-1">
                    OpenAI provider support coming soon
                  </div>
                </div>
              )}
            </div>
          </RadioGroup>
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
            <AlertDialogAction onClick={confirmDelete} className="bg-destructive text-destructive-foreground">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
