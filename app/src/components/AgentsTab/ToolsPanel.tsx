import { Plus, Trash2, Wrench } from 'lucide-react';
import { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
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
import type { AgentToolCreate, ToolParam, VoiceAgent } from '@/lib/api/types';
import { useToolMutations, useTools } from '@/lib/hooks/useVoiceAgents';

const EMPTY: AgentToolCreate = {
  name: '',
  description: '',
  method: 'GET',
  url: '',
  headers: {},
  params: [],
  timeout_s: 10,
  enabled: true,
};

export function ToolsPanel({ agent }: { agent: VoiceAgent }) {
  const { toast } = useToast();
  const { data: tools, isLoading } = useTools(agent.id);
  const { create, update, remove } = useToolMutations(agent.id);
  const [draft, setDraft] = useState<AgentToolCreate>(EMPTY);
  const [headerText, setHeaderText] = useState('');

  const builtins = [
    ...(agent.mode === 'outbound_sales' || agent.booking_instructions ? ['book_appointment'] : []),
    'schedule_callback',
    'transfer_to_human',
    ...(agent.provider === 'twilio' ? ['send_sms'] : []),
  ];

  const add = async () => {
    const headers: Record<string, string> = {};
    for (const line of headerText.split('\n')) {
      const idx = line.indexOf(':');
      if (idx > 0) headers[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
    }
    try {
      await create.mutateAsync({ ...draft, headers, params: draft.params ?? [] });
      setDraft(EMPTY);
      setHeaderText('');
    } catch (err) {
      toast({ title: 'Could not add tool', description: String(err), variant: 'destructive' });
    }
  };

  const params = draft.params ?? [];
  const setParam = (i: number, p: ToolParam) =>
    setDraft({ ...draft, params: params.map((x, j) => (j === i ? p : x)) });

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">
        Tools let the agent act mid-call. Built-ins write to Voicebox; HTTP tools call your own
        systems (order lookup, stock check, CRM). The model sees each tool's description and calls
        it with the arguments you declare — never invent results, never hide a failure.
      </p>
      {!agent.tools_enabled && (
        <div className="text-xs rounded-lg border border-amber-500/40 bg-amber-500/10 text-amber-200 px-3 py-2">
          Tools are switched off for this agent (Setup → Conversation engine).
        </div>
      )}

      <div className="flex flex-wrap gap-1.5">
        {builtins.map((b) => (
          <Badge key={b} variant="outline" className="font-mono text-[11px]">
            <Wrench className="h-3 w-3 mr-1" />
            {b}
          </Badge>
        ))}
        <span className="text-[11px] text-muted-foreground self-center">built-in</span>
      </div>

      <div className="rounded-lg border border-border p-3 space-y-2">
        <div className="grid grid-cols-3 gap-2">
          <Input
            className="h-9"
            placeholder="tool_name"
            value={draft.name}
            onChange={(e) =>
              setDraft({ ...draft, name: e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, '_') })
            }
          />
          <Select
            value={draft.method}
            onValueChange={(v) => setDraft({ ...draft, method: v as AgentToolCreate['method'] })}
          >
            <SelectTrigger className="h-9">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {['GET', 'POST', 'PUT', 'PATCH', 'DELETE'].map((m) => (
                <SelectItem key={m} value={m}>
                  {m}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Input
            className="h-9"
            type="number"
            min={1}
            max={60}
            value={draft.timeout_s}
            onChange={(e) => setDraft({ ...draft, timeout_s: Number(e.target.value) || 10 })}
            placeholder="timeout s"
          />
        </div>
        <Input
          className="h-9"
          placeholder="https://api.example.com/orders/{order_id}"
          value={draft.url}
          onChange={(e) => setDraft({ ...draft, url: e.target.value })}
        />
        <Textarea
          rows={2}
          placeholder="When and why the agent should use it — the model reads this."
          value={draft.description}
          onChange={(e) => setDraft({ ...draft, description: e.target.value })}
        />
        <Textarea
          rows={2}
          placeholder={'Headers, one per line:\nAuthorization: Bearer …'}
          value={headerText}
          onChange={(e) => setHeaderText(e.target.value)}
        />
        <div className="space-y-1.5">
          {params.map((p, i) => (
            <div key={`${i}-${p.name}`} className="flex gap-2 items-center">
              <Input
                className="h-8 w-36"
                placeholder="param_name"
                value={p.name}
                onChange={(e) =>
                  setParam(i, {
                    ...p,
                    name: e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, '_'),
                  })
                }
              />
              <Select
                value={p.type}
                onValueChange={(v) => setParam(i, { ...p, type: v as ToolParam['type'] })}
              >
                <SelectTrigger className="h-8 w-28">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="string">string</SelectItem>
                  <SelectItem value="number">number</SelectItem>
                  <SelectItem value="boolean">boolean</SelectItem>
                </SelectContent>
              </Select>
              <Input
                className="h-8 flex-1"
                placeholder="what it means"
                value={p.description ?? ''}
                onChange={(e) => setParam(i, { ...p, description: e.target.value })}
              />
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Checkbox
                  id={`req-${i}`}
                  checked={p.required}
                  onCheckedChange={(v) => setParam(i, { ...p, required: v })}
                />
                <label htmlFor={`req-${i}`}>required</label>
              </div>
              <Button
                size="icon"
                variant="ghost"
                className="h-8 w-8"
                onClick={() => setDraft({ ...draft, params: params.filter((_, j) => j !== i) })}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          ))}
          <Button
            size="sm"
            variant="ghost"
            onClick={() =>
              setDraft({
                ...draft,
                params: [...params, { name: '', type: 'string', required: true }],
              })
            }
          >
            <Plus className="h-4 w-4" /> Argument
          </Button>
        </div>
        <div className="flex justify-end">
          <Button
            size="sm"
            onClick={add}
            disabled={!draft.name || !draft.url || !draft.description || create.isPending}
          >
            <Plus className="h-4 w-4" /> Add tool
          </Button>
        </div>
        <p className="text-[11px] text-muted-foreground">
          Use <code>{'{param}'}</code> in the URL to substitute an argument; other arguments go in
          the query string (GET) or JSON body (POST/PUT/PATCH).
        </p>
      </div>

      {isLoading ? (
        <div className="text-sm text-muted-foreground">Loading…</div>
      ) : (
        <div className="space-y-2">
          {(tools ?? []).map((t) => (
            <div key={t.id} className="rounded-lg border border-border p-3 flex items-start gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm">{t.name}</span>
                  <Badge variant="outline" className="text-[10px]">
                    {t.method}
                  </Badge>
                  {!t.enabled && (
                    <Badge variant="outline" className="text-[10px] text-muted-foreground">
                      disabled
                    </Badge>
                  )}
                </div>
                <div className="text-xs text-muted-foreground truncate">{t.url}</div>
                <div className="text-sm mt-1">{t.description}</div>
                {(t.params ?? []).length > 0 && (
                  <div className="text-[11px] text-muted-foreground mt-1">
                    {(t.params ?? [])
                      .map((p) => `${p.name}: ${p.type}${p.required ? '' : '?'}`)
                      .join(', ')}
                  </div>
                )}
              </div>
              <div className="flex items-center gap-1">
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => update.mutate({ toolId: t.id, data: { enabled: !t.enabled } })}
                >
                  {t.enabled ? 'Disable' : 'Enable'}
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-8 w-8 text-muted-foreground hover:text-destructive"
                  onClick={() => remove.mutate(t.id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
