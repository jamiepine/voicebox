import { FlaskConical, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import type { VoiceAgent } from '@/lib/api/types';
import { useSimulateCall } from '@/lib/hooks/useVoiceAgents';
import { cn } from '@/lib/utils/cn';

const PERSONAS: Array<{ label: string; text: string }> = [
  {
    label: 'Busy but polite',
    text: 'A busy but polite homeowner who is mildly sceptical and asks one practical question before deciding.',
  },
  {
    label: 'Interested',
    text: 'Someone who is genuinely interested, asks about price and timing, and agrees to the next step if it sounds reasonable.',
  },
  {
    label: 'Firm no',
    text: 'A person who is not interested, says so clearly after the first pitch, and gets irritated if pushed.',
  },
  {
    label: 'Angry customer',
    text: 'A frustrated customer whose service has failed twice this week. Short-tempered, wants it fixed now, threatens to cancel.',
  },
  {
    label: 'Confused caller',
    text: 'An older caller who is unsure what they need, gives vague answers, and needs simple step-by-step help.',
  },
  {
    label: 'Wants a human',
    text: 'A caller who distrusts automated systems and asks to speak to a real person within two turns.',
  },
];

interface SimulateDialogProps {
  agent: VoiceAgent;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDone: (callId: string) => void;
}

export function SimulateDialog({ agent, open, onOpenChange, onDone }: SimulateDialogProps) {
  const { toast } = useToast();
  const simulate = useSimulateCall(agent.id);
  const [persona, setPersona] = useState(PERSONAS[0].text);
  const [maxTurns, setMaxTurns] = useState(12);
  const [variant, setVariant] = useState<string>('auto');
  const variants = agent.variants ?? [];

  const run = async () => {
    try {
      const call = await simulate.mutateAsync({
        persona,
        max_turns: maxTurns,
        variant: variant === 'auto' ? null : variant,
      });
      onOpenChange(false);
      onDone(call.id);
      toast({
        title: `Test call finished — ${call.outcome ?? 'no outcome'}`,
        description:
          call.score != null ? `Score ${call.score}/100. ${call.score_reason ?? ''}` : undefined,
      });
    } catch (err) {
      toast({
        title: 'Simulation failed',
        description: err instanceof Error ? err.message : String(err),
        variant: 'destructive',
      });
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FlaskConical className="h-4 w-4 text-accent" /> Run a test call
          </DialogTitle>
          <DialogDescription>
            The local model plays a customer with this persona against the agent's real script,
            knowledge and tools. No audio, no contacts touched. The transcript gets a score.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <div className="flex flex-wrap gap-1.5">
            {PERSONAS.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => setPersona(p.text)}
                className={cn(
                  'h-7 px-2.5 rounded-full text-xs border',
                  persona === p.text
                    ? 'bg-accent/15 border-accent'
                    : 'border-border text-muted-foreground hover:bg-muted/40',
                )}
              >
                {p.label}
              </button>
            ))}
          </div>
          <Textarea rows={3} value={persona} onChange={(e) => setPersona(e.target.value)} />
          <div className="flex items-center gap-3">
            <label className="text-xs text-muted-foreground" htmlFor="sim-turns">
              Max customer turns
            </label>
            <Input
              id="sim-turns"
              className="h-8 w-20"
              type="number"
              min={2}
              max={40}
              value={maxTurns}
              onChange={(e) => setMaxTurns(Number(e.target.value) || 12)}
            />
            {variants.length > 0 && (
              <Select value={variant} onValueChange={setVariant}>
                <SelectTrigger className="h-8 w-40 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Variant: weighted</SelectItem>
                  {variants.map((v) => (
                    <SelectItem key={v.name} value={v.name}>
                      Variant: {v.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button onClick={run} disabled={simulate.isPending || !persona.trim()}>
              {simulate.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <FlaskConical className="h-4 w-4" />
              )}
              {simulate.isPending ? 'Running…' : 'Run test call'}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
