import { Download, MoreHorizontal, Play, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { apiClient } from '@/lib/api/client';
import { useDeleteGeneration, useHistory } from '@/lib/hooks/useHistory';
import { cn } from '@/lib/utils/cn';
import { formatDate, formatDuration } from '@/lib/utils/format';
import { usePlayerStore } from '@/stores/playerStore';

export function HistoryTable() {
  const [page, setPage] = useState(0);
  const limit = 20;

  const { data: historyData, isLoading } = useHistory({
    limit,
    offset: page * limit,
  });

  const deleteGeneration = useDeleteGeneration();
  const setAudio = usePlayerStore((state) => state.setAudio);
  const currentAudioId = usePlayerStore((state) => state.audioId);
  const isPlaying = usePlayerStore((state) => state.isPlaying);
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;

  const handlePlay = (audioId: string, text: string) => {
    const audioUrl = apiClient.getAudioUrl(audioId);
    // If clicking the same audio that's playing, it will be handled by the player
    setAudio(audioUrl, audioId, text.substring(0, 50));
  };

  const handleDownload = (audioId: string, text: string) => {
    const audioUrl = apiClient.getAudioUrl(audioId);
    const filename = `${text.substring(0, 30).replace(/[^a-z0-9]/gi, '_')}.wav`;
    const link = document.createElement('a');
    link.href = audioUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-muted-foreground">Loading history...</div>
      </div>
    );
  }

  const history = historyData?.items || [];
  const total = historyData?.total || 0;
  const hasMore = history.length === limit && (page + 1) * limit < total;

  return (
    <div className="flex flex-col h-full min-h-0">
      {history.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground flex-1 flex items-center justify-center">
          No generation history yet. Generate your first audio to see it here.
        </div>
      ) : (
        <>
          <div
            className={cn(
              'flex-1 min-h-0 overflow-y-auto border rounded-md overflow-x-hidden',
              isPlayerVisible && 'max-h-[calc(100vh-220px)]',
            )}
          >
            <Table className="w-full table-fixed">
              <TableHeader className="sticky top-0 bg-background z-10">
                <TableRow>
                  <TableHead className="w-[38%]">Input</TableHead>
                  <TableHead className="w-[13%]">Voice</TableHead>
                  <TableHead className="w-[9%]">Lang</TableHead>
                  <TableHead className="w-[9%]">Length</TableHead>
                  <TableHead className="w-[13%]">Date</TableHead>
                  <TableHead className="w-[8%] text-right"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {history.map((gen) => {
                  const isCurrentlyPlaying = currentAudioId === gen.id && isPlaying;
                  return (
                    <TableRow
                      key={gen.id}
                      className={cn(isCurrentlyPlaying && 'bg-muted/50', 'cursor-pointer')}
                      onClick={() => handlePlay(gen.id, gen.text)}
                    >
                      <TableCell className="truncate">{gen.text}</TableCell>
                      <TableCell className="truncate">{gen.profile_name}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className="text-xs text-muted-foreground">
                          {gen.language}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-sm">{formatDuration(gen.duration)}</TableCell>
                      <TableCell className="text-xs text-muted-foreground/60">
                        {formatDate(gen.created_at)}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end" onClick={(e) => e.stopPropagation()}>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-7 w-7 rounded-full"
                                aria-label="Actions"
                              >
                                <MoreHorizontal className="h-3.5 w-3.5" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem onClick={() => handlePlay(gen.id, gen.text)}>
                                <Play className="mr-2 h-4 w-4" />
                                Play
                              </DropdownMenuItem>
                              <DropdownMenuItem onClick={() => handleDownload(gen.id, gen.text)}>
                                <Download className="mr-2 h-4 w-4" />
                                Download
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                onClick={() => deleteGeneration.mutate(gen.id)}
                                disabled={deleteGeneration.isPending}
                                className="text-destructive focus:text-destructive"
                              >
                                <Trash2 className="mr-2 h-4 w-4" />
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>

          <div className="flex justify-between items-center mt-4 shrink-0">
            <Button
              variant="outline"
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
            >
              Previous
            </Button>
            <div className="text-sm text-muted-foreground">
              Page {page + 1} • {total} total
            </div>
            <Button variant="outline" onClick={() => setPage((p) => p + 1)} disabled={!hasMore}>
              Next
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
