import { useRef, useState } from 'react';
import { Sparkles } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { useFinetuneSamples, useFinetuneStatus } from '@/lib/hooks/useFinetune';
import { usePlayerStore } from '@/stores/playerStore';
import { cn } from '@/lib/utils/cn';
import { FinetuneSampleManager } from './FinetuneSampleManager';
import { FinetuneTrainingPanel } from './FinetuneTrainingPanel';

export function FinetuneTab() {
  const { data: profiles } = useProfiles();
  const [selectedProfileId, setSelectedProfileId] = useState<string>('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const audioUrl = usePlayerStore((state) => state.audioUrl);
  const isPlayerVisible = !!audioUrl;

  // Auto-select first profile
  const profileId = selectedProfileId || profiles?.[0]?.id || '';
  const selectedProfile = profiles?.find((p) => p.id === profileId);

  const { data: samples } = useFinetuneSamples(profileId);
  const { data: status } = useFinetuneStatus(profileId);

  return (
    <div
      ref={scrollRef}
      className={cn(
        'flex flex-col gap-6 py-6 overflow-y-auto flex-1',
        isPlayerVisible && BOTTOM_SAFE_AREA_PADDING,
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Sparkles className="h-6 w-6 text-accent" />
          <div>
            <h1 className="text-2xl font-bold">Fine-Tune</h1>
            <p className="text-sm text-muted-foreground">
              Train per-profile LoRA adapters for higher quality voice output
            </p>
          </div>
        </div>

        {/* Profile Selector */}
        <Select value={profileId} onValueChange={setSelectedProfileId}>
          <SelectTrigger className="w-64">
            <SelectValue placeholder="Select a profile" />
          </SelectTrigger>
          <SelectContent>
            {profiles?.map((profile) => (
              <SelectItem key={profile.id} value={profile.id}>
                <div className="flex items-center gap-2">
                  <span>{profile.name}</span>
                  {profile.has_finetune && (
                    <Sparkles className="h-3 w-3 text-accent" />
                  )}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {!profileId ? (
        <div className="text-center py-20 text-muted-foreground">
          <Sparkles className="h-12 w-12 mx-auto mb-4 opacity-30" />
          <p className="text-lg">Select a voice profile to get started</p>
          <p className="text-sm mt-1">Create profiles in the Voices tab first</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Sample Manager */}
          <FinetuneSampleManager
            profileId={profileId}
            profileLanguage={selectedProfile?.language}
            samples={samples || []}
          />

          {/* Right: Training Panel */}
          {status && (
            <FinetuneTrainingPanel
              profileId={profileId}
              status={status}
              samples={samples || []}
            />
          )}
        </div>
      )}
    </div>
  );
}
