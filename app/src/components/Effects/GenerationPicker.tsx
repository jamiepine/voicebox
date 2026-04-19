import { ChevronDown, Search } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import type { HistoryResponse } from '@/lib/api/types';
import { useHistory } from '@/lib/hooks/useHistory';
import { cn } from '@/lib/utils/cn';

interface GenerationPickerProps {
  selectedId: string | null;
  onSelect: (generation: HistoryResponse) => void;
  className?: string;
}

export function GenerationPicker({ selectedId, onSelect, className }: GenerationPickerProps) {
  const [open, setOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const { data: historyData } = useHistory({ limit: 50 });

  const completedGenerations = useMemo(() => {
    if (!historyData?.items) return [];
    return historyData.items.filter((gen) => gen.status === 'completed');
  }, [historyData]);

  const filtered = useMemo(() => {
    if (!searchQuery) return completedGenerations;
    const q = searchQuery.toLowerCase();
    return completedGenerations.filter(
      (gen) => gen.text.toLowerCase().includes(q) || gen.profile_name.toLowerCase().includes(q),
    );
  }, [completedGenerations, searchQuery]);

  const selectedGeneration = completedGenerations.find((g) => g.id === selectedId);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className={cn('h-8 justify-between gap-2 text-xs font-normal', className)}
        >
          {selectedGeneration ? (
            <span className="truncate">
              <span className="font-medium">{selectedGeneration.profile_name}</span>
              <span className="text-muted-foreground ml-1.5">
                {selectedGeneration.text.length > 30
                  ? `${selectedGeneration.text.substring(0, 30)}...`
                  : selectedGeneration.text}
              </span>
            </span>
          ) : (
            <span className="text-muted-foreground">Select a generation...</span>
          )}
          <ChevronDown className="h-3.5 w-3.5 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 p-0" align="start">
        <div className="p-2 border-b">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              placeholder="Search by voice or text..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-8 pl-7 text-xs"
            />
          </div>
        </div>
        <div className="max-h-60 overflow-y-auto">
          {filtered.length === 0 ? (
            <div className="p-4 text-center text-xs text-muted-foreground">
              No generations found
            </div>
          ) : (
            filtered.map((gen) => (
              <button
                key={gen.id}
                type="button"
                className={cn(
                  'w-full text-left px-3 py-2 hover:bg-muted/50 transition-colors border-b border-border/30 last:border-0',
                  gen.id === selectedId && 'bg-accent/10',
                )}
                onClick={() => {
                  onSelect(gen);
                  setOpen(false);
                  setSearchQuery('');
                }}
              >
                <div className="font-medium text-sm">{gen.profile_name}</div>
                <div className="text-xs text-muted-foreground truncate">
                  {gen.text.length > 60 ? `${gen.text.substring(0, 60)}...` : gen.text}
                </div>
              </button>
            ))
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
