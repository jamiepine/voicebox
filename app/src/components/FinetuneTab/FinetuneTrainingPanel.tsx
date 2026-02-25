import { useEffect, useMemo, useState } from 'react';
import { AlertCircle, CheckCircle2, ChevronDown, ChevronUp, Loader2, Play, Square, XCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/components/ui/use-toast';
import type { FinetuneStatusResponse, FinetuneSampleResponse } from '@/lib/api/types';
import { useCancelFinetune, useStartFinetune } from '@/lib/hooks/useFinetune';
import { AdapterSelector } from './AdapterSelector';

interface FinetuneTrainingPanelProps {
  profileId: string;
  status: FinetuneStatusResponse;
  samples: FinetuneSampleResponse[];
}

export function FinetuneTrainingPanel({
  profileId,
  status,
  samples,
}: FinetuneTrainingPanelProps) {
  const { toast } = useToast();
  const startFinetune = useStartFinetune();
  const cancelFinetune = useCancelFinetune();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [adapterLabel, setAdapterLabel] = useState('');
  const [epochs, setEpochs] = useState(3);
  const [learningRate, setLearningRate] = useState(0.00002);
  const [loraRank, setLoraRank] = useState(16);

  // Reset label when switching profiles
  useEffect(() => {
    setAdapterLabel('');
  }, [profileId]);

  const totalDuration = useMemo(() => samples.reduce((sum, s) => sum + s.duration_seconds, 0), [samples]);
  const hasRefAudio = samples.some((s) => s.is_ref_audio);
  const canStart = samples.length >= 10 && hasRefAudio;
  const activeJob = status.active_job;
  const isTraining = activeJob && ['pending', 'preparing', 'training'].includes(activeJob.status);

  const handleStart = async () => {
    if (!adapterLabel.trim()) {
      toast({
        title: 'Name required',
        description: 'Please give this adapter a name before starting training.',
        variant: 'destructive',
      });
      return;
    }

    try {
      await startFinetune.mutateAsync({
        profileId,
        config: {
          epochs,
          learning_rate: learningRate,
          lora_rank: loraRank,
          label: adapterLabel.trim(),
        },
      });
      setAdapterLabel('');
      toast({ title: 'Training started', description: 'Fine-tuning has begun. This may take a while.' });
    } catch (error) {
      toast({
        title: 'Failed to start training',
        description: error instanceof Error ? error.message : 'Error',
        variant: 'destructive',
      });
    }
  };

  const handleCancel = async () => {
    try {
      await cancelFinetune.mutateAsync(profileId);
      toast({ title: 'Training cancelled' });
    } catch (error) {
      toast({
        title: 'Failed to cancel',
        description: error instanceof Error ? error.message : 'Error',
        variant: 'destructive',
      });
    }
  };

  const progressPercent =
    activeJob && activeJob.total_steps > 0
      ? Math.round((activeJob.current_step / activeJob.total_steps) * 100)
      : 0;

  return (
    <div className="flex flex-col gap-4 border rounded-lg p-4">
      <h3 className="text-lg font-semibold">Training</h3>

      {/* Dataset Summary */}
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <span className="text-muted-foreground">Samples</span>
          <p className="font-medium">{samples.length}</p>
        </div>
        <div>
          <span className="text-muted-foreground">Total Duration</span>
          <p className="font-medium">{Math.floor(totalDuration / 60)}m {Math.floor(totalDuration % 60)}s</p>
        </div>
        <div>
          <span className="text-muted-foreground">Ref Audio</span>
          <p className="font-medium">{hasRefAudio ? 'Set' : 'Not set'}</p>
        </div>
      </div>

      {/* Training In Progress */}
      {isTraining && activeJob && (
        <div className="bg-muted/50 rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin text-accent" />
            <span className="font-medium">
              {activeJob.status === 'preparing' ? 'Preparing dataset...' : 'Training in progress...'}
            </span>
          </div>

          {activeJob.status === 'training' && (
            <>
              <Progress value={progressPercent} className="h-2" />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>
                  Epoch {activeJob.current_epoch}/{activeJob.epochs} &middot; Step{' '}
                  {activeJob.current_step}/{activeJob.total_steps}
                </span>
                <span>{progressPercent}%</span>
              </div>
              {activeJob.current_loss != null && (
                <p className="text-sm text-muted-foreground">Loss: {activeJob.current_loss.toFixed(4)}</p>
              )}
            </>
          )}

          <Button
            variant="destructive"
            size="sm"
            onClick={handleCancel}
            disabled={cancelFinetune.isPending}
          >
            <Square className="h-4 w-4 mr-1" />
            Cancel Training
          </Button>

          <p className="text-xs text-muted-foreground">
            Generation is blocked while training is active (GPU busy).
          </p>
        </div>
      )}

      {/* Adapter Selector — show when profile has any adapters */}
      {status.has_finetune && !isTraining && (
        <>
          <AdapterSelector profileId={profileId} />
          {status.has_active_adapter && (
            <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-3 space-y-1">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                <span className="text-sm font-medium">Adapter active</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Generation uses Qwen3-TTS + LoRA adapter. Hebrew bypasses Chatterbox.
              </p>
            </div>
          )}
        </>
      )}

      {/* Last job error */}
      {activeJob && activeJob.status === 'failed' && (
        <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 space-y-2">
          <div className="flex items-center gap-2">
            <XCircle className="h-4 w-4 text-destructive" />
            <span className="font-medium">Training failed</span>
          </div>
          {activeJob.error_message && (
            <p className="text-sm text-muted-foreground">{activeJob.error_message}</p>
          )}
        </div>
      )}

      {/* Start Training */}
      {!isTraining && (
        <>
          {!canStart && (
            <div className="flex items-start gap-2 text-sm text-muted-foreground">
              <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
              <div>
                {samples.length < 10 && <p>Need at least 10 samples ({samples.length}/10)</p>}
                {!hasRefAudio && <p>Set a reference audio sample</p>}
              </div>
            </div>
          )}

          {/* Advanced Settings */}
          <button
            className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            Advanced Settings
          </button>

          {showAdvanced && (
            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-1">
                <Label className="text-xs">Epochs</Label>
                <Input
                  type="number"
                  min={1}
                  max={50}
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value) || 3)}
                  className="h-8"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">Learning Rate</Label>
                <Input
                  type="number"
                  step={0.000001}
                  min={0.000001}
                  max={0.01}
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value) || 2e-5)}
                  className="h-8"
                />
              </div>
              <div className="space-y-1">
                <Label className="text-xs">LoRA Rank</Label>
                <Input
                  type="number"
                  min={4}
                  max={128}
                  value={loraRank}
                  onChange={(e) => setLoraRank(parseInt(e.target.value) || 32)}
                  className="h-8"
                />
              </div>
            </div>
          )}

          <div className="space-y-1">
            <Label className="text-xs">Adapter Name</Label>
            <Input
              placeholder="e.g. Hebrew v1, Warm tone, Fast speech..."
              value={adapterLabel}
              onChange={(e) => setAdapterLabel(e.target.value)}
              maxLength={100}
              className="h-8"
            />
            <p className="text-xs text-muted-foreground">
              Give this adapter a name so you can identify it later
            </p>
          </div>

          <Button
            onClick={handleStart}
            disabled={!canStart || !adapterLabel.trim() || startFinetune.isPending}
            className="w-full"
          >
            {startFinetune.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            {status.has_finetune ? 'Retrain' : 'Start Training'}
          </Button>
        </>
      )}
    </div>
  );
}
