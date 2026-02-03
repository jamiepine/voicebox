import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/components/ui/use-toast';
import { Loader2 } from 'lucide-react';
import { apiClient } from '@/lib/api/client';
import type { UpdateBackendResponse, BackendOption } from '@/lib/api/types';

export function BackendSelector() {
    const { toast } = useToast();
    const queryClient = useQueryClient();

    // Fetch available backend options
    const { data: optionsData, isLoading: optionsLoading } = useQuery({
        queryKey: ['backend-options'],
        queryFn: () => apiClient.getBackendOptions(),
    });

    // Fetch current backend
    const { data: currentData, isLoading: currentLoading } = useQuery({
        queryKey: ['backend-current'],
        queryFn: () => apiClient.getSetting('tts_backend'),
    });

    // Update backend mutation
    const updateBackend = useMutation({
        mutationFn: (value: string) => apiClient.updateSetting('tts_backend', value),
        onSuccess: (data: UpdateBackendResponse) => {
            // Invalidate queries to refresh
            queryClient.invalidateQueries({ queryKey: ['backend-current'] });
            queryClient.invalidateQueries({ queryKey: ['health'] });

            toast({
                title: 'Backend updated',
                description: `Switched to ${data.value}${data.reload_required ? ' (backend reloaded)' : ''}`,
            });
        },
        onError: (error: Error) => {
            toast({
                title: 'Failed to update backend',
                description: error.message,
                variant: 'destructive',
            });
        },
    });

    const handleBackendChange = (value: string) => {
        if (value !== currentData?.value) {
            updateBackend.mutate(value);
        }
    };

    const isLoading = optionsLoading || currentLoading;

    return (
        <Card>
            <CardHeader>
                <CardTitle>TTS Backend</CardTitle>
                <CardDescription>
                    Choose which text-to-speech model to use
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                {isLoading ? (
                    <div className="flex items-center justify-center py-4">
                        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                    </div>
                ) : (
                    <div className="space-y-2">
                        <Label htmlFor="backend-select">Active Model</Label>
                        <Select
                            value={currentData?.value}
                            onValueChange={handleBackendChange}
                            disabled={updateBackend.isPending}
                        >
                            <SelectTrigger id="backend-select" className="w-full">
                                <SelectValue placeholder="Select a backend" />
                            </SelectTrigger>
                            <SelectContent>
                                {optionsData?.options.map((option: BackendOption) => (
                                    <SelectItem key={option.value} value={option.value}>
                                        <div className="flex flex-col">
                                            <span className="font-medium">{option.label}</span>
                                            <span className="text-xs text-muted-foreground">
                                                {option.description}
                                            </span>
                                        </div>
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                )}

                {updateBackend.isPending && (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>Switching backend...</span>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
