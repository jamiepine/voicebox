import { AudioWaveform, Download, FileArchive, MoreHorizontal, Play, Trash2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Textarea } from '@/components/ui/textarea';
import { apiClient } from '@/lib/api/client';
import {
  useDeleteGeneration,
  useExportGeneration,
  useExportGenerationAudio,
  useHistory,
  useImportGeneration,
} from '@/lib/hooks/useHistory';
import { cn } from '@/lib/utils/cn';
import { formatDate, formatDuration } from '@/lib/utils/format';
import { usePlayerStore } from '@/stores/playerStore';

// OLD TABLE-BASED COMPONENT - REMOVED (can be found in git history)
// This is the new alternate history view with fixed height rows

// NEW ALTERNATE HISTORY VIEW - FIXED HEIGHT ROWS
export function HistoryTable() {
  const [page, setPage] = useState(0);
  const [isScrolled, setIsScrolled] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const limit = 20;

  const { data: historyData, isLoading } = useHistory({
    limit,
    offset: page * limit,
  });

  const deleteGeneration = useDeleteGeneration();
  const exportGeneration = useExportGeneration();
  const exportGenerationAudio = useExportGenerationAudio();
  const importGeneration = useImportGeneration();
  const setAudio = usePlayerStore((state) => state.setAudio);
  const restartCurrentAudio = usePlayerStore((state) => state.restartCurrentAudio);
  const currentAudioId = usePlayerStore((state) => state.audioId);
  const isPlaying = usePlayerStore((state) => state.isPlaying);
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;

  useEffect(() => {
    const scrollEl = scrollRef.current;
    if (!scrollEl) return;

    const handleScroll = () => {
      setIsScrolled(scrollEl.scrollTop > 0);
    };

    scrollEl.addEventListener('scroll', handleScroll);
    return () => scrollEl.removeEventListener('scroll', handleScroll);
  }, []);

  const handlePlay = (audioId: string, text: string) => {
    // If clicking the same audio, restart it from the beginning
    if (currentAudioId === audioId) {
      restartCurrentAudio();
    } else {
      // Otherwise, load the new audio
      const audioUrl = apiClient.getAudioUrl(audioId);
      setAudio(audioUrl, audioId, text.substring(0, 50));
    }
  };

  const handleDownloadAudio = (generationId: string, text: string) => {
    exportGenerationAudio.mutate(
      { generationId, text },
      {
        onError: (error) => {
          alert(`Failed to download audio: ${error.message}`);
        },
      },
    );
  };

  const handleExportPackage = (generationId: string, text: string) => {
    exportGeneration.mutate(
      { generationId, text },
      {
        onError: (error) => {
          alert(`Failed to export generation: ${error.message}`);
        },
      },
    );
  };

  const _handleImportClick = () => {
    file_handleImportClickk.click();
  };

  const _handleFileChange = (_e: React.ChangeEvent<HTMLInputElement>) => {
    cons_handleFileChangeet.files?.[0];
    if (file) {
      // Validate file extension
      if (!file.name.endsWith('.voicebox.zip')) {
        alert('Please select a valid .voicebox.zip file');
        return;
      }
      setSelectedFile(file);
      setImportDialogOpen(true);
    }
  };

  const handleImportConfirm = () => {
    if (selectedFile) {
      importGeneration.mutate(selectedFile, {
        onSuccess: (data) => {
          setImportDialogOpen(false);
          setSelectedFile(null);
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
          alert(data.message || 'Generation imported successfully');
        },
        onError: (error) => {
          alert(`Failed to import generation: ${error.message}`);
        },
      });
    }
  };

  if (isLoading) {
    return null;
  }

  const history = historyData?.items || [];
  const total = historyData?.total || 0;
  const hasMore = history.length === limit && (page + 1) * limit < total;

  return (
    <div className="flex flex-col h-full min-h-0 relative">
      {/* <div className="flex justify-between items-center mb-4 shrink-0">
        <h2 className="text-2xl font-bold">History</h2>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleImportClick}>
            <Upload className="mr-2 h-4 w-4" />
            Import Generation
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".voicebox.zip"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>
      </div> */}

      {history.length === 0 ? (
        <div className="text-center py-12 px-5 border-2 border-dashed border-gray-200 rounded-md text-muted-foreground flex-1 flex items-center justify-center">
          No generation history yet. Generate your first audio to see it here.
        </div>
      ) : (
        <>
          {isScrolled && (
            <div className="absolute top-0 left-0 right-0 h-16 bg-gradient-to-b from-background to-transparent z-10 pointer-events-none" />
          )}
          <div
            ref={scrollRef}
            className={cn(
              'flex-1 min-h-0 overflow-y-auto space-y-2',
              isPlayerVisible && 'max-h-[calc(100vh-117px)]',
            )}
          >
            {history.map((gen) => {
              const isCurrentlyPlaying = currentAudioId === gen.id && isPlaying;
              return (
                <div
                  key={gen.id}
                  className={cn(
                    'flex items-stretch gap-4 h-26 border rounded-md p-3 bg-card hover:bg-muted/70 transition-colors text-left w-full',
                    isCurrentlyPlaying && 'bg-muted/70',
                  )}
                  onMouseDown={(e) => {
                    // Don't trigger play if clicking on textarea or if text is selected
                    const target = e.target as HTMLElement;
                    if (target.closest('textarea') || window.getSelection()?.toString()) {
                      return;
                    }
                    handlePlay(gen.id, gen.text);
                  }}
                >
                  {/* Waveform icon */}
                  <div className="flex items-center shrink-0">
                    <AudioWaveform className="h-5 w-5 text-muted-foreground" />
                  </div>

                  {/* Left side - Meta information */}
                  <div className="flex flex-col gap-1.5 w-48 shrink-0 justify-center">
                    <div className="font-medium text-sm truncate" title={gen.profile_name}>
                      {gen.profile_name}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">{gen.language}</span>
                      <span className="text-xs text-muted-foreground">
                        {formatDuration(gen.duration)}
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {formatDate(gen.created_at)}
                    </div>
                  </div>

                  {/* Right side - Transcript textarea */}
                  <div className="flex-1 min-w-0 flex">
                    <Textarea
                      value={gen.text}
                      className="flex-1 resize-none text-sm text-muted-foreground select-text"
                    />
                  </div>

                  {/* Far right - Ellipsis actions */}
                  <div className="w-10 shrink-0 flex justify-end">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          aria-label="Actions"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={() => handlePlay(gen.id, gen.text)}>
                          <Play className="mr-2 h-4 w-4" />
                          Play
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() => handleDownloadAudio(gen.id, gen.text)}
                          disabled={exportGenerationAudio.isPending}
                        >
                          <Download className="mr-2 h-4 w-4" />
                          Export Audio
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() => handleExportPackage(gen.id, gen.text)}
                          disabled={exportGeneration.isPending}
                        >
                          <FileArchive className="mr-2 h-4 w-4" />
                          Export Package
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
                </div>
              );
            })}
          </div>

          {(total > limit || page > 0) && (
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
          )}
        </>
      )}

      <Dialog open={importDialogOpen} onOpenChange={setImportDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Import Generation</DialogTitle>
            <DialogDescription>
              Import the generation from "{selectedFile?.name}". This will add it to your history.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setImportDialogOpen(false);
                setSelectedFile(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={handleImportConfirm}
              disabled={importGeneration.isPending || !selectedFile}
            >
              {importGeneration.isPending ? 'Importing...' : 'Import'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
