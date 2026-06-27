import { Sparkles, Upload } from 'lucide-react';
import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
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

import { useImportProfile } from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';
import { usePlayerStore } from '@/stores/playerStore';
import { useUIStore } from '@/stores/uiStore';

export function MainEditor() {
  const { t } = useTranslation();
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
          title: t('main.import.invalidTitle'),
          description: t('main.import.invalidDescription'),
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
            title: t('main.import.successTitle'),
            description: t('main.import.successDescription'),
          });
        },
        onError: (error) => {
          toast({
            title: t('main.import.failedTitle'),
            description: error.message,
            variant: 'destructive',
          });
        },
      });
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 lg:gap-6 h-full min-h-0 overflow-hidden relative">
      <div className="flex flex-col min-h-0 overflow-hidden relative lg:overflow-hidden">
        <div className="absolute top-0 start-0 end-0 h-16 bg-gradient-to-b from-background to-transparent z-0 pointer-events-none" />

        <div className="absolute top-0 start-0 end-0 z-10">
          <div className="flex items-center justify-between mb-4 px-1">
            <h2 className="text-2xl font-bold">Voicebox</h2>
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleImportClick}>
                <Upload className="me-2 h-4 w-4" />
                {t('main.importVoice')}
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".voicebox.zip"
                onChange={handleFileChange}
                className="hidden"
              />
              <Button onClick={() => setDialogOpen(true)}>
                <Sparkles className="me-2 h-4 w-4" />
                {t('main.createVoice')}
              </Button>
            </div>
          </div>
        </div>

        <div
          ref={scrollRef}
          className={cn('flex-1 min-h-0 overflow-y-auto pt-14 pb-4', isPlayerVisible && 'lg:pb-32')}
        >
          <div className="flex flex-col gap-6">
            <div className="shrink-0 flex flex-col">
              <ProfileList />
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-col min-h-0 overflow-hidden">
        <HistoryTable />
      </div>

      <FloatingGenerateBox isPlayerOpen={!!audioUrl} />

      <Dialog open={importDialogOpen} onOpenChange={setImportDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('main.import.dialogTitle')}</DialogTitle>
            <DialogDescription>
              {t('main.import.dialogDescription', { name: selectedFile?.name })}
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
              {t('common.cancel')}
            </Button>
            <Button
              onClick={handleImportConfirm}
              disabled={importProfile.isPending || !selectedFile}
            >
              {importProfile.isPending ? t('main.import.importing') : t('main.import.action')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
