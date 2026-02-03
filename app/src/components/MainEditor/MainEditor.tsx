import { HugeiconsIcon } from '@hugeicons/react';
import { SparklesIcon, Upload01Icon } from '@hugeicons/core-free-icons';
import { useRef, useState } from 'react';
import { FloatingGenerateBox } from '@/components/Generation/FloatingGenerateBox';
import { HistoryTable } from '@/components/History/HistoryTable';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useToast } from '@/components/ui/use-toast';
import { ProfileList } from '@/components/VoiceProfiles/ProfileList';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { useImportProfile } from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';
import { usePlayerStore } from '@/stores/playerStore';
import { useUIStore } from '@/stores/uiStore';

export function MainEditor() {
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;
  const scrollRef = useRef<HTMLDivElement>(null);
  const setDialogOpen = useUIStore((state) => state.setProfileDialogOpen);
  const importProfile = useImportProfile();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const { toast } = useToast();

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.voicebox.zip')) {
        toast({
          title: 'Invalid file type',
          description: 'Please select a valid .voicebox.zip file',
          variant: 'destructive',
        });
        return;
      }
      setSelectedFile(file);
      setImportDialogOpen(true);
    }
  };

  const handleImportConfirm = () => {
    if (selectedFile) {
      importProfile.mutate(selectedFile, {
        onSuccess: () => {
          setImportDialogOpen(false);
          setSelectedFile(null);
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
          toast({
            title: 'Profile imported',
            description: 'Voice profile imported successfully',
          });
        },
        onError: (error) => {
          toast({
            title: 'Failed to import profile',
            description: error.message,
            variant: 'destructive',
          });
        },
      });
    }
  };

  return (
    // Main view: Profiles top left, Generator bottom left, History right
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full min-h-0 overflow-hidden relative">
      {/* Left Column */}
      <div className="flex flex-col min-h-0 overflow-hidden relative">
        {/* Scroll Mask - Always visible, behind content */}
        <div className="absolute top-0 left-0 right-0 h-16 bg-gradient-to-b from-background to-transparent z-0 pointer-events-none" />

        {/* Fixed Header */}
        <div className="absolute top-0 left-0 right-0 z-10">
          <div className="flex items-center justify-between mb-4 px-1">
            <h2 className="text-2xl font-bold">Voicebox</h2>
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleImportClick}>
                <HugeiconsIcon icon={Upload01Icon} size={16} className="mr-2 h-4 w-4" />
                Import Voice
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".voicebox.zip"
                onChange={handleFileChange}
                className="hidden"
              />
              <Button onClick={() => setDialogOpen(true)}>
                <HugeiconsIcon icon={SparklesIcon} size={16} className="mr-2 h-4 w-4" />
                Create Voice
              </Button>
            </div>
          </div>
        </div>

        {/* Scrollable Content */}
        <div
          ref={scrollRef}
          className={cn(
            'flex-1 min-h-0 overflow-y-auto pt-14',
            isPlayerVisible ? BOTTOM_SAFE_AREA_PADDING : 'pb-4',
          )}
        >
          <div className="flex flex-col gap-6">
            <div className="shrink-0 flex flex-col">
              <ProfileList />
            </div>
          </div>
        </div>
      </div>

      {/* Right Column - History */}
      <div className="flex flex-col min-h-0 overflow-hidden">
        <HistoryTable />
      </div>

      {/* Floating Generate Box */}
      <FloatingGenerateBox isPlayerOpen={!!audioUrl} />

      {/* Import Dialog */}
      <Dialog open={importDialogOpen} onOpenChange={setImportDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Import Profile</DialogTitle>
            <DialogDescription>
              Import the profile from "{selectedFile?.name}". This will create a new profile with
              all samples.
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
              disabled={importProfile.isPending || !selectedFile}
            >
              {importProfile.isPending ? 'Importing...' : 'Import'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
