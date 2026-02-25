import { useState } from 'react';
import { Check, Pencil, Trash2, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useToast } from '@/components/ui/use-toast';
import type { AdapterInfo } from '@/lib/api/types';
import { useAdapters, useSetActiveAdapter, useUpdateAdapterLabel, useDeleteAdapter } from '@/lib/hooks/useFinetune';
import { adapterDisplayName } from '@/lib/utils/adapters';

interface AdapterSelectorProps {
  profileId: string;
}

function formatDate(dateStr?: string) {
  if (!dateStr) return '';
  const d = new Date(dateStr);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function AdapterSelector({ profileId }: AdapterSelectorProps) {
  const { toast } = useToast();
  const { data: adapters, isLoading } = useAdapters(profileId);
  const setActiveAdapter = useSetActiveAdapter();
  const updateLabel = useUpdateAdapterLabel();
  const deleteAdapter = useDeleteAdapter();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');

  if (isLoading || !adapters || adapters.length === 0) return null;

  const activeAdapter = adapters.find((a) => a.is_active);
  const currentValue = activeAdapter?.job_id ?? 'none';

  const handleChange = async (value: string) => {
    const jobId = value === 'none' ? null : value;
    try {
      await setActiveAdapter.mutateAsync({ profileId, jobId });
      toast({
        title: jobId ? 'Adapter activated' : 'Adapter deactivated',
        description: jobId ? 'Generation will use the selected adapter.' : 'Generation will use the base model.',
      });
    } catch (error) {
      toast({
        title: 'Failed to switch adapter',
        description: error instanceof Error ? error.message : 'Error',
        variant: 'destructive',
      });
    }
  };

  const handleStartEdit = (adapter: AdapterInfo) => {
    setEditingId(adapter.job_id);
    setEditValue(adapter.label || '');
  };

  const handleSaveLabel = async (jobId: string) => {
    if (!editValue.trim()) {
      setEditingId(null);
      return;
    }
    try {
      await updateLabel.mutateAsync({ profileId, jobId, label: editValue.trim() });
      setEditingId(null);
    } catch {
      toast({ title: 'Failed to rename adapter', variant: 'destructive' });
    }
  };

  const handleDelete = async (adapter: AdapterInfo) => {
    const confirmed = window.confirm(
      `Delete adapter "${adapterDisplayName(adapter)}"? This permanently removes the trained model files and cannot be undone.`,
    );
    if (!confirmed) return;

    try {
      await deleteAdapter.mutateAsync({ profileId, jobId: adapter.job_id });
      toast({ title: 'Adapter deleted' });
    } catch {
      toast({ title: 'Failed to delete adapter', variant: 'destructive' });
    }
  };

  return (
    <div className="flex flex-col gap-3 border rounded-lg p-4">
      <div className="flex items-center gap-2">
        <Zap className="h-4 w-4 text-accent" />
        <h3 className="text-sm font-semibold">Active Adapter</h3>
      </div>

      <Select value={currentValue} onValueChange={handleChange} disabled={setActiveAdapter.isPending}>
        <SelectTrigger className="h-9">
          <SelectValue placeholder="Select adapter..." />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="none">None (base model)</SelectItem>
          {adapters.map((adapter) => (
            <SelectItem key={adapter.job_id} value={adapter.job_id}>
              {adapterDisplayName(adapter)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Adapter list with management actions */}
      <div className="space-y-2">
        {adapters.map((adapter) => (
          <div
            key={adapter.job_id}
            className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm ${
              adapter.is_active ? 'bg-accent/10 border border-accent/20' : 'bg-muted/50'
            }`}
          >
            {editingId === adapter.job_id ? (
              <div className="flex items-center gap-1 flex-1">
                <Input
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleSaveLabel(adapter.job_id);
                    if (e.key === 'Escape') setEditingId(null);
                  }}
                  className="h-7 text-xs"
                  autoFocus
                />
                <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => handleSaveLabel(adapter.job_id)}>
                  <Check className="h-3 w-3" />
                </Button>
              </div>
            ) : (
              <>
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{adapterDisplayName(adapter)}</p>
                  <p className="text-xs text-muted-foreground">
                    {formatDate(adapter.completed_at)}
                    {adapter.is_active && ' \u00B7 Active'}
                  </p>
                </div>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-7 w-7 shrink-0"
                  onClick={() => handleStartEdit(adapter)}
                  title="Rename"
                >
                  <Pencil className="h-3 w-3" />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-7 w-7 shrink-0 text-destructive hover:text-destructive"
                  onClick={() => handleDelete(adapter)}
                  title="Delete adapter"
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
