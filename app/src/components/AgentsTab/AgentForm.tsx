import { Mic, Sparkles } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import type { VoiceAgent, VoiceAgentCreate, VoiceAgentMode } from '@/lib/api/types';
import { LANGUAGE_OPTIONS } from '@/lib/constants/languages';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';
import { MODE_META } from './shared';

const DEFAULT_DISCLOSURE =
  "Just so you know, I'm an automated AI assistant and this call may be recorded.";

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

const GOAL_PLACEHOLDER: Record<VoiceAgentMode, string> = {
  outbound_sales: 'Book a free 20-minute consultation for next week.',
  customer_service: 'Answer the caller’s question fully, or log it for a person to follow up.',
  support: 'Get the customer working again, or open a ticket with a clear description.',
};

const BRIEF_PLACEHOLDER: Record<VoiceAgentMode, string> = {
  outbound_sales:
    'What you are offering, in plain facts the agent may state. Prices, guarantees, what happens next. The agent will not claim anything that is not written here.',
  customer_service:
    'What the business does, opening hours, policies, what the agent can and cannot do on a call.',
  support:
    'The product, common issues, what the agent may ask the customer to try, and what must go to an engineer.',
};

function emptyForm(): VoiceAgentCreate {
  const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
  return {
    name: '',
    mode: 'outbound_sales',
    profile: '',
    engine: null,
    language: 'en',
    llm_model_size: null,
    agent_name: '',
    company_name: '',
    brief: '',
    goal: '',
    objection_notes: '',
    persona: '',
    opening_line: '',
    disclosure: DEFAULT_DISCLOSURE,
    escalation_promise: '',
    timezone: tz,
    calling_window_start: 9,
    calling_window_end: 20,
    calling_days: [0, 1, 2, 3, 4],
    max_attempts: 3,
    daily_call_cap: 200,
    retry_delay_hours: 24,
    callback_delay_hours: 24,
    require_consent: false,
    max_turns: 30,
    handoff_after_negative_turns: 3,
    provider: 'local',
    from_number: '',
  };
}

function fromAgent(agent: VoiceAgent): VoiceAgentCreate {
  const {
    id: _id,
    status: _s,
    profile_id,
    running: _r,
    created_at: _c,
    updated_at: _u,
    ...rest
  } = agent;
  return { ...rest, profile: profile_id };
}

interface AgentFormProps {
  initial?: VoiceAgent | null;
  submitting?: boolean;
  submitLabel: string;
  onSubmit: (data: VoiceAgentCreate) => void;
  onCancel?: () => void;
}

export function AgentForm({
  initial,
  submitting,
  submitLabel,
  onSubmit,
  onCancel,
}: AgentFormProps) {
  const [form, setForm] = useState<VoiceAgentCreate>(() =>
    initial ? fromAgent(initial) : emptyForm(),
  );
  const { data: profiles } = useProfiles();
  const set = <K extends keyof VoiceAgentCreate>(key: K, value: VoiceAgentCreate[K]) =>
    setForm((f) => ({ ...f, [key]: value }));

  const sortedProfiles = useMemo(
    () => [...(profiles ?? [])].sort((a, b) => a.name.localeCompare(b.name)),
    [profiles],
  );
  const isOutbound = form.mode === 'outbound_sales';
  const canSubmit =
    form.name.trim() &&
    form.profile &&
    form.agent_name.trim() &&
    form.company_name.trim() &&
    form.brief.trim() &&
    form.goal.trim();

  const submit = () => {
    if (!canSubmit) return;
    const clean = (v: string | null | undefined) => (v?.trim() ? v.trim() : null);
    onSubmit({
      ...form,
      objection_notes: clean(form.objection_notes),
      persona: clean(form.persona),
      opening_line: clean(form.opening_line),
      escalation_promise: clean(form.escalation_promise),
      from_number: clean(form.from_number),
      engine: form.engine || null,
      llm_model_size: form.llm_model_size || null,
    });
  };

  return (
    <div className="space-y-6">
      {/* Mode */}
      <div className="grid grid-cols-3 gap-3">
        {(Object.keys(MODE_META) as VoiceAgentMode[]).map((mode) => {
          const meta = MODE_META[mode];
          const Icon = meta.icon;
          const active = form.mode === mode;
          return (
            <button
              key={mode}
              type="button"
              onClick={() => set('mode', mode)}
              className={cn(
                'text-left rounded-xl border p-3 transition-colors',
                active
                  ? 'border-accent bg-accent/10'
                  : 'border-border hover:border-muted-foreground/40 hover:bg-muted/30',
              )}
            >
              <div className="flex items-center gap-2 font-medium">
                <Icon className={cn('h-4 w-4', active ? 'text-accent' : 'text-muted-foreground')} />
                {meta.label}
              </div>
              <div className="text-xs text-muted-foreground mt-1 leading-snug">{meta.blurb}</div>
            </button>
          );
        })}
      </div>

      {/* Identity */}
      <Section title="Identity">
        <div className="grid grid-cols-2 gap-3">
          <Field label="Agent name (internal)">
            <Input
              value={form.name}
              onChange={(e) => set('name', e.target.value)}
              placeholder="Q4 solar campaign"
            />
          </Field>
          <Field label="Voice">
            <Select value={form.profile} onValueChange={(v) => set('profile', v)}>
              <SelectTrigger>
                <SelectValue placeholder="Pick a voice profile" />
              </SelectTrigger>
              <SelectContent>
                {sortedProfiles.map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    <span className="inline-flex items-center gap-2">
                      {p.voice_type === 'cloned' ? (
                        <Mic className="h-3.5 w-3.5 text-muted-foreground" />
                      ) : (
                        <Sparkles className="h-3.5 w-3.5 text-muted-foreground" />
                      )}
                      {p.name}
                      <span className="text-xs text-muted-foreground">
                        {p.voice_type === 'cloned' ? 'cloned' : p.voice_type}
                      </span>
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-[11px] text-muted-foreground mt-1">
              Any profile works — including voices you cloned from your own recordings. Create new
              ones in the Voices tab.
            </p>
          </Field>
          <Field label="Introduces itself as">
            <Input
              value={form.agent_name}
              onChange={(e) => set('agent_name', e.target.value)}
              placeholder="Sam"
            />
          </Field>
          <Field label="Company">
            <Input
              value={form.company_name}
              onChange={(e) => set('company_name', e.target.value)}
              placeholder="Acme Solar"
            />
          </Field>
          <Field label="Language">
            <Select value={form.language} onValueChange={(v) => set('language', v)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {LANGUAGE_OPTIONS.map((l) => (
                  <SelectItem key={l.value} value={l.value}>
                    {l.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Field label="Reasoning model">
            <Select
              value={form.llm_model_size ?? 'default'}
              onValueChange={(v) => set('llm_model_size', v === 'default' ? null : v)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="default">App default</SelectItem>
                <SelectItem value="0.6B">Qwen3 0.6B — fastest</SelectItem>
                <SelectItem value="1.7B">Qwen3 1.7B — balanced</SelectItem>
                <SelectItem value="4B">Qwen3 4B — best judgement</SelectItem>
              </SelectContent>
            </Select>
          </Field>
        </div>
      </Section>

      {/* Script */}
      <Section title="What the agent knows and wants">
        <Field label={isOutbound ? 'The offer (facts only)' : 'Service brief'}>
          <Textarea
            rows={5}
            value={form.brief}
            onChange={(e) => set('brief', e.target.value)}
            placeholder={BRIEF_PLACEHOLDER[form.mode]}
          />
        </Field>
        <Field label="Goal of every call">
          <Input
            value={form.goal}
            onChange={(e) => set('goal', e.target.value)}
            placeholder={GOAL_PLACEHOLDER[form.mode]}
          />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label={isOutbound ? 'Objection handling' : 'Policies & edge cases'}>
            <Textarea
              rows={4}
              value={form.objection_notes ?? ''}
              onChange={(e) => set('objection_notes', e.target.value)}
              placeholder={
                isOutbound
                  ? '“Too expensive” → mention 0% finance. “Already have one” → ask when it was installed.'
                  : 'Refunds over £100 go to a person. Never promise same-day engineer visits.'
              }
            />
          </Field>
          <Field label="Persona / tone">
            <Textarea
              rows={4}
              value={form.persona ?? ''}
              onChange={(e) => set('persona', e.target.value)}
              placeholder="Warm, unhurried, a little dry. Never salesy. Uses first names."
            />
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Custom opening hook (optional)">
            <Input
              value={form.opening_line ?? ''}
              onChange={(e) => set('opening_line', e.target.value)}
              placeholder={isOutbound ? 'Do you have a quick minute?' : 'How can I help you today?'}
            />
          </Field>
          <Field label="What happens when it hands off">
            <Input
              value={form.escalation_promise ?? ''}
              onChange={(e) => set('escalation_promise', e.target.value)}
              placeholder="a specialist will call you back within one business day"
            />
          </Field>
        </div>
        <Field label="Disclosure (spoken at the top of every call)">
          <Input value={form.disclosure} onChange={(e) => set('disclosure', e.target.value)} />
        </Field>
      </Section>

      {/* Guard-rails */}
      <Section title={isOutbound ? 'Dialing rules' : 'Call limits'}>
        {isOutbound && (
          <>
            <div className="grid grid-cols-3 gap-3">
              <Field label="Time zone (fallback)">
                <Input
                  value={form.timezone}
                  onChange={(e) => set('timezone', e.target.value)}
                  placeholder="Europe/London"
                />
              </Field>
              <Field label="Call from (hour)">
                <Input
                  type="number"
                  min={0}
                  max={23}
                  value={form.calling_window_start}
                  onChange={(e) => set('calling_window_start', Number(e.target.value))}
                />
              </Field>
              <Field label="Until (hour)">
                <Input
                  type="number"
                  min={1}
                  max={24}
                  value={form.calling_window_end}
                  onChange={(e) => set('calling_window_end', Number(e.target.value))}
                />
              </Field>
            </div>
            <Field label="Days">
              <div className="flex gap-1.5">
                {DAYS.map((d, i) => {
                  const on = form.calling_days.includes(i);
                  return (
                    <button
                      key={d}
                      type="button"
                      onClick={() =>
                        set(
                          'calling_days',
                          on
                            ? form.calling_days.filter((x) => x !== i)
                            : [...form.calling_days, i].sort(),
                        )
                      }
                      className={cn(
                        'h-8 px-3 rounded-full text-xs border transition-colors',
                        on
                          ? 'bg-accent/15 border-accent text-foreground'
                          : 'border-border text-muted-foreground hover:bg-muted/40',
                      )}
                    >
                      {d}
                    </button>
                  );
                })}
              </div>
            </Field>
            <div className="grid grid-cols-4 gap-3">
              <Field label="Max attempts">
                <Input
                  type="number"
                  min={1}
                  max={20}
                  value={form.max_attempts}
                  onChange={(e) => set('max_attempts', Number(e.target.value))}
                />
              </Field>
              <Field label="Daily cap">
                <Input
                  type="number"
                  min={1}
                  value={form.daily_call_cap}
                  onChange={(e) => set('daily_call_cap', Number(e.target.value))}
                />
              </Field>
              <Field label="Retry after (h)">
                <Input
                  type="number"
                  min={1}
                  value={form.retry_delay_hours}
                  onChange={(e) => set('retry_delay_hours', Number(e.target.value))}
                />
              </Field>
              <Field label="Callback after (h)">
                <Input
                  type="number"
                  min={1}
                  value={form.callback_delay_hours}
                  onChange={(e) => set('callback_delay_hours', Number(e.target.value))}
                />
              </Field>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <Checkbox
                id="agent-require-consent"
                checked={form.require_consent}
                onCheckedChange={(v) => set('require_consent', v)}
              />
              <label htmlFor="agent-require-consent" className="cursor-pointer">
                Only dial contacts marked as having given consent
              </label>
            </div>
          </>
        )}
        <div className="grid grid-cols-2 gap-3">
          <Field label="Max turns per call">
            <Input
              type="number"
              min={2}
              max={200}
              value={form.max_turns}
              onChange={(e) => set('max_turns', Number(e.target.value))}
            />
          </Field>
          <Field label="Hand off after N upset turns">
            <Input
              type="number"
              min={1}
              max={20}
              value={form.handoff_after_negative_turns}
              onChange={(e) => set('handoff_after_negative_turns', Number(e.target.value))}
            />
          </Field>
        </div>
      </Section>

      {/* Telephony */}
      <Section title="Phone line">
        <div className="grid grid-cols-2 gap-3">
          <Field label="Provider">
            <Select
              value={form.provider}
              onValueChange={(v) => set('provider', v as 'local' | 'twilio')}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="local">Local — speakers + this console</SelectItem>
                <SelectItem value="twilio">Twilio — real phone calls</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          {form.provider === 'twilio' && (
            <Field label="From number (Twilio-owned)">
              <Input
                value={form.from_number ?? ''}
                onChange={(e) => set('from_number', e.target.value)}
                placeholder="+15550100000"
              />
            </Field>
          )}
        </div>
        {form.provider === 'twilio' && (
          <p className="text-xs text-muted-foreground">
            Set <code>TWILIO_ACCOUNT_SID</code>, <code>TWILIO_AUTH_TOKEN</code> and{' '}
            <code>VOICEBOX_PUBLIC_URL</code> in the server environment. For inbound agents, point
            the number’s voice webhook at{' '}
            <code>
              {'{VOICEBOX_PUBLIC_URL}'}/webhooks/twilio/inbound/{'{agent id}'}
            </code>
            .
          </p>
        )}
      </Section>

      <div className="flex justify-end gap-2 pt-2">
        {onCancel && (
          <Button type="button" variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button type="button" onClick={submit} disabled={!canSubmit || submitting}>
          {submitLabel}
        </Button>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </h3>
      {children}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs text-muted-foreground">{label}</Label>
      {children}
    </div>
  );
}
