import { Loader2 } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import type { ActiveFinetuneTask } from '@/lib/api/types';

interface FinetuneProgressToastProps {
  task: ActiveFinetuneTask;
}

export function FinetuneProgressToast({ task }: FinetuneProgressToastProps) {
  const progressPercent =
    task.total_steps > 0
      ? Math.round((task.current_step / task.total_steps) * 100)
      : 0;

  return (
    <div className="fixed bottom-4 right-4 z-50 bg-card border rounded-lg shadow-lg p-4 w-80 space-y-2">
      <div className="flex items-center gap-2">
        <Loader2 className="h-4 w-4 animate-spin text-accent" />
        <span className="font-medium text-sm">Fine-tuning in progress</span>
      </div>

      {task.status === 'training' && (
        <>
          <Progress value={progressPercent} className="h-1.5" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>
              Epoch {task.current_epoch}/{task.total_epochs}
            </span>
            <span>{progressPercent}%</span>
          </div>
          {task.current_loss != null && (
            <p className="text-xs text-muted-foreground">Loss: {task.current_loss.toFixed(4)}</p>
          )}
        </>
      )}

      {task.status === 'preparing' && (
        <p className="text-xs text-muted-foreground">Preparing dataset...</p>
      )}
    </div>
  );
}
