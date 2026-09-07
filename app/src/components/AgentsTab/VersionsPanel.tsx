import { History, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';
import type { VoiceAgent } from '@/lib/api/types';
import { useAgentVersions, useRestoreAgentVersion } from '@/lib/hooks/useVoiceAgents';
import { cn } from '@/lib/utils/cn';
import { formatDate } from '@/lib/utils/format';

const SHOWN_KEYS = [
  'brief',
  'goal',
  'opening_line',
  'persona',
  'objection_notes',
  'disclosure',
] as const;

export function VersionsPanel({ agent }: { agent: VoiceAgent }) {
  const { toast } = useToast();
  const { data: versions, isLoading } = useAgentVersions(agent.id);
  const restore = useRestoreAgentVersion(agent.id);

  if (isLoading) return <div className="text-sm text-muted-foreground">Loading…</div>;
  const list = versions ?? [];

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        Every save is a version. Restore one to roll back a script change that hurt results — the
        restore itself becomes a new version, so nothing is lost.
      </p>
      {list.length === 0 && (
        <div className="text-sm text-muted-foreground py-8 text-center rounded-lg border border-dashed border-border">
          <History className="h-5 w-5 mx-auto mb-2 opacity-40" /> No versions yet.
        </div>
      )}
      {list.map((v, i) => {
        const prev = list[i + 1];
        const changed = prev
          ? Object.keys(v.snapshot).filter(
              (k) => JSON.stringify(v.snapshot[k]) !== JSON.stringify(prev.snapshot[k]),
            )
          : [];
        const current = v.version === agent.version;
        return (
          <div
            key={v.id}
            className={cn(
              'rounded-lg border p-3',
              current ? 'border-accent/50 bg-accent/5' : 'border-border',
            )}
          >
            <div className="flex items-center gap-2">
              <span className="font-medium">v{v.version}</span>
              {current && (
                <span className="text-[10px] uppercase tracking-wider text-accent">current</span>
              )}
              <span className="text-xs text-muted-foreground">
                {v.note ?? ''} · {formatDate(v.created_at)}
              </span>
              {!current && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="ml-auto"
                  disabled={restore.isPending}
                  onClick={async () => {
                    try {
                      await restore.mutateAsync(v.id);
                      toast({ title: `Restored v${v.version}` });
                    } catch (err) {
                      toast({
                        title: 'Restore failed',
                        description: String(err),
                        variant: 'destructive',
                      });
                    }
                  }}
                >
                  <RotateCcw className="h-4 w-4" /> Restore
                </Button>
              )}
            </div>
            {changed.length > 0 && (
              <div className="text-[11px] text-muted-foreground mt-1">
                changed: {changed.join(', ')}
              </div>
            )}
            <details className="mt-1">
              <summary className="text-xs text-muted-foreground cursor-pointer">
                Script at this version
              </summary>
              <div className="mt-2 space-y-1.5 text-xs">
                {SHOWN_KEYS.map((k) => {
                  const value = v.snapshot[k];
                  if (!value) return null;
                  return (
                    <div key={k}>
                      <span className="font-mono text-muted-foreground">{k}: </span>
                      <span className="whitespace-pre-wrap">{String(value)}</span>
                    </div>
                  );
                })}
              </div>
            </details>
          </div>
        );
      })}
    </div>
  );
}
