import { Mic, Sparkles, Upload } from 'lucide-react';
import { useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useImportProfile, useProfiles } from '@/lib/hooks/useProfiles';
import { useUIStore } from '@/stores/uiStore';
import { ProfileCard } from './ProfileCard';
import { ProfileForm } from './ProfileForm';

export function ProfileList() {
  const { data: profiles, isLoading, error } = useProfiles();
  const setDialogOpen = useUIStore((state) => state.setProfileDialogOpen);
  const importProfile = useImportProfile();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
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
      importProfile.mutate(selectedFile, {
        onSuccess: () => {
          setImportDialogOpen(false);
          setSelectedFile(null);
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
        },
        onError: (error) => {
          alert(`Failed to import profile: ${error.message}`);
        },
      });
    }
  };

  if (isLoading) {
    return null;
  }

  if (error) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-destructive">Error loading profiles: {error.message}</div>
      </div>
    );
  }

  const allProfiles = profiles || [];

  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between mb-4 shrink-0">
        <h2 className="text-2xl font-bold">Voicebox</h2>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleImportClick}>
            <Upload className="mr-2 h-4 w-4" />
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
            <Sparkles className="mr-2 h-4 w-4" />
            Create Voice
          </Button>
        </div>
      </div>

      <div className="shrink-0">
        {allProfiles.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Mic className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground mb-4">
                No voice profiles yet. Create your first profile to get started.
              </p>
              <Button onClick={() => setDialogOpen(true)}>
                <Sparkles className="mr-2 h-4 w-4" />
                Create Voice
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4 grid-cols-3 auto-rows-auto p-1">
            {allProfiles.map((profile) => (
              <ProfileCard key={profile.id} profile={profile} />
            ))}
          </div>
        )}
      </div>

      <ProfileForm />

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
