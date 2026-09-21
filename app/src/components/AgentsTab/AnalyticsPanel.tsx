import { BarChart3, Download } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { apiClient } from '@/lib/api/client';
import type { CallOutcome, VoiceAgent } from '@/lib/api/types';
import { useVoiceAgentAnalytics, useWebhookDeliveries } from '@/lib/hooks/useVoiceAgents';
import { cn } from '@/lib/utils/cn';
import { formatMs, OUTCOME_META } from './shared';

export function AnalyticsPanel({ agent }: { agent: VoiceAgent }) {
  const [days, setDays] = useState(30);
  const { data, isLoading } = useVoiceAgentAnalytics(agent.id, days);
  const { data: deliveries } = useWebhookDeliveries(agent.webhook_url ? agent.id : null);
  const goalLabel = agent.mode === 'outbound_sales' ? 'Interested' : 'Resolved';

  if (isLoading || !data) return <div className="text-sm text-muted-foreground">Loading…</div>;

  const maxCalls = Math.max(1, ...data.series.map((p) => p.calls));
  const funnel = [
    ['Contacts', data.funnel.contacts],
    ['Attempted', data.funnel.attempted],
    ['Connected', data.funnel.connected],
    [goalLabel, data.funnel.goal],
  ] as const;
  const funnelMax = Math.max(1, data.funnel.contacts, data.funnel.attempted);
  const outcomeEntries = Object.entries(data.outcomes).sort((a, b) => b[1] - a[1]);
  const outcomeTotal = outcomeEntries.reduce((n, [, v]) => n + v, 0) || 1;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Select value={String(days)} onValueChange={(v) => setDays(Number(v))}>
          <SelectTrigger className="h-8 w-[140px] text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="7">Last 7 days</SelectItem>
            <SelectItem value="30">Last 30 days</SelectItem>
            <SelectItem value="90">Last 90 days</SelectItem>
          </SelectContent>
        </Select>
        <span className="text-xs text-muted-foreground">
          {data.simulations} simulation{data.simulations === 1 ? '' : 's'} excluded
        </span>
        <div className="ml-auto flex gap-2">
          <Button variant="outline" size="sm" asChild>
            <a href={apiClient.getCallsCsvUrl(agent.id)} target="_blank" rel="noreferrer">
              <Download className="h-4 w-4" /> Calls CSV
            </a>
          </Button>
          <Button variant="outline" size="sm" asChild>
            <a href={apiClient.getContactsCsvUrl(agent.id)} target="_blank" rel="noreferrer">
              <Download className="h-4 w-4" /> Contacts CSV
            </a>
          </Button>
        </div>
      </div>

      {/* Tiles */}
      <div className="grid grid-cols-6 gap-2">
        <Tile label="Avg turns" value={data.avg_turns} />
        <Tile label="Avg duration" value={`${Math.round(data.avg_duration_s)}s`} />
        <Tile label="Avg score" value={data.avg_score ?? '—'} />
        <Tile
          label="Sentiment"
          value={data.avg_sentiment == null ? '—' : data.avg_sentiment.toFixed(2)}
        />
        <Tile label="STT latency" value={formatMs(data.avg_stt_ms) || '—'} />
        <Tile label="LLM latency" value={formatMs(data.avg_llm_ms) || '—'} />
      </div>

      {/* Calls per day */}
      <div className="rounded-xl border border-border p-4">
        <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
          Calls per day
        </div>
        {data.series.length === 0 ? (
          <div className="text-sm text-muted-foreground py-6 text-center">
            <BarChart3 className="h-5 w-5 mx-auto mb-2 opacity-40" /> No calls in this period.
          </div>
        ) : (
          <div className="flex items-end gap-1 h-32">
            {data.series.map((p) => (
              <div
                key={p.date}
                className="flex-1 flex flex-col items-center justify-end gap-1 min-w-[10px]"
                title={`${p.date}: ${p.calls} calls, ${p.goal} ${goalLabel.toLowerCase()}`}
              >
                <div className="w-full flex flex-col justify-end" style={{ height: '100%' }}>
                  <div
                    className="w-full rounded-t bg-muted-foreground/30 relative"
                    style={{ height: `${(p.calls / maxCalls) * 100}%` }}
                  >
                    <div
                      className="absolute bottom-0 left-0 right-0 rounded-t bg-emerald-400/70"
                      style={{ height: `${p.calls ? (p.goal / p.calls) * 100 : 0}%` }}
                    />
                  </div>
                </div>
                <div className="text-[9px] text-muted-foreground">{p.date.slice(5)}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Funnel */}
        <div className="rounded-xl border border-border p-4 space-y-2">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
            Funnel
          </div>
          {funnel.map(([label, value]) => (
            <div key={label} className="flex items-center gap-2 text-sm">
              <span className="w-24 text-muted-foreground">{label}</span>
              <div className="flex-1 h-4 rounded bg-muted/40 overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded',
                    label === goalLabel ? 'bg-emerald-400/70' : 'bg-accent/50',
                  )}
                  style={{ width: `${(value / funnelMax) * 100}%` }}
                />
              </div>
              <span className="w-10 text-right tabular-nums">{value}</span>
            </div>
          ))}
        </div>

        {/* Outcomes */}
        <div className="rounded-xl border border-border p-4 space-y-2">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
            Outcomes
          </div>
          {outcomeEntries.length === 0 && <div className="text-sm text-muted-foreground">—</div>}
          {outcomeEntries.map(([k, v]) => (
            <div key={k} className="flex items-center gap-2 text-sm">
              <span className="w-28 text-muted-foreground truncate">
                {OUTCOME_META[k as CallOutcome]?.label ?? k}
              </span>
              <div className="flex-1 h-3 rounded bg-muted/40 overflow-hidden">
                <div
                  className="h-full rounded bg-accent/50"
                  style={{ width: `${(v / outcomeTotal) * 100}%` }}
                />
              </div>
              <span className="w-10 text-right tabular-nums">{v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Variants + analysis */}
      {(data.variants.length > 0 || Object.keys(data.analysis).length > 0) && (
        <div className="grid grid-cols-2 gap-4">
          {data.variants.length > 0 && (
            <div className="rounded-xl border border-border p-4 space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                A/B variants
              </div>
              {data.variants.map((v) => (
                <div key={v.name} className="flex items-center gap-2 text-sm">
                  <span className="w-28 truncate">{v.name}</span>
                  <div className="flex-1 h-3 rounded bg-muted/40 overflow-hidden">
                    <div
                      className="h-full rounded bg-emerald-400/70"
                      style={{ width: `${v.goal_rate * 100}%` }}
                    />
                  </div>
                  <span className="w-24 text-right tabular-nums text-xs text-muted-foreground">
                    {Math.round(v.goal_rate * 100)}% of {v.calls}
                  </span>
                </div>
              ))}
            </div>
          )}
          {Object.keys(data.analysis).length > 0 && (
            <div className="rounded-xl border border-border p-4 space-y-3">
              <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                Post-call analysis
              </div>
              {Object.entries(data.analysis).map(([key, counts]) => (
                <div key={key}>
                  <div className="text-xs font-mono mb-1">{key}</div>
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(counts)
                      .sort((a, b) => b[1] - a[1])
                      .map(([val, n]) => (
                        <span
                          key={val}
                          className="rounded-full border border-border px-2 py-0.5 text-[11px]"
                        >
                          {val} <span className="text-muted-foreground">×{n}</span>
                        </span>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {agent.webhook_url && (
        <div className="rounded-xl border border-border p-4 space-y-2">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
            Webhook deliveries
          </div>
          {(deliveries ?? []).length === 0 && (
            <div className="text-sm text-muted-foreground">None yet.</div>
          )}
          {(deliveries ?? []).slice(0, 10).map((d) => (
            <div key={d.id} className="flex items-center gap-2 text-xs">
              <span
                className={cn(
                  'inline-block h-2 w-2 rounded-full',
                  d.status === 'delivered'
                    ? 'bg-emerald-400'
                    : d.status === 'failed'
                      ? 'bg-red-400'
                      : 'bg-amber-400',
                )}
              />
              <span className="text-muted-foreground">{d.status}</span>
              <span className="truncate">{d.url}</span>
              <span className="ml-auto text-muted-foreground">
                {d.attempts} attempt{d.attempts === 1 ? '' : 's'}
                {d.response_code ? ` · HTTP ${d.response_code}` : ''}
                {d.last_error ? ` · ${d.last_error}` : ''}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Tile({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl border border-border bg-card/40 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="text-lg font-semibold leading-tight mt-0.5">{value}</div>
    </div>
  );
}
