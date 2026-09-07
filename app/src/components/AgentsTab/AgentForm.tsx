import { Mic, Plus, Sparkles, Trash2 } from 'lucide-react';
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
import type {
  AnalysisField,
  ScriptVariant,
  VoiceAgent,
  VoiceAgentCreate,
  VoiceAgentMode,
} from '@/lib/api/types';
import { LANGUAGE_OPTIONS } from '@/lib/constants/languages';
import { useProfiles } from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';
import { MODE_META, OUTCOME_META } from './shared';

const DEFAULT_DISCLOSURE =
  "Just so you know, I'm an automated AI assistant and this call may be recorded.";
const DEFAULT_FILLERS = [
  'One moment.',
  'Sure, let me check that for you.',
  'Okay, bear with me a second.',
];

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

const COMPLIANCE_PRESETS: Array<{
  label: string;
  hint: string;
  start: number;
  end: number;
  days: number[];
  attempts: number;
}> = [
  {
    label: 'US (TCPA)',
    hint: '8am–9pm local, any day, 3 attempts',
    start: 8,
    end: 21,
    days: [0, 1, 2, 3, 4, 5, 6],
    attempts: 3,
  },
  {
    label: 'UK (Ofcom)',
    hint: '9am–8pm local, Mon–Sat, 3 attempts',
    start: 9,
    end: 20,
    days: [0, 1, 2, 3, 4, 5],
    attempts: 3,
  },
  {
    label: 'EU conservative',
    hint: '9am–6pm local, weekdays, 2 attempts',
    start: 9,
    end: 18,
    days: [0, 1, 2, 3, 4],
    attempts: 2,
  },
];

const GOAL_PLACEHOLDER: Record<VoiceAgentMode, string> = {
  outbound_sales: 'Book a free 20-minute consultation for next week.',
  customer_service: 'Answer the caller’s question fully, or log it for a person to follow up.',
  support: 'Get the customer working again, or open a ticket with a clear description.',
};

const BRIEF_PLACEHOLDER: Record<VoiceAgentMode, string> = {
  outbound_sales:
    'What you are offering, in plain facts the agent may state. Prices, guarantees, what happens next. The agent will not claim anything that is not written here. You can use {{contact.first_name}}, {{contact.company}} and {{contact.custom.<field>}}.',
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
    voice_style: '',
    empathetic_voice_style: 'calm, warm and apologetic',
    agent_name: '',
    company_name: '',
    brief: '',
    goal: '',
    objection_notes: '',
    persona: '',
    opening_line: '',
    disclosure: DEFAULT_DISCLOSURE,
    escalation_promise: '',
    variants: [],
    filler_phrases: [...DEFAULT_FILLERS],
    fast_first_audio: true,
    tools_enabled: true,
    booking_instructions: '',
    appointment_duration_min: 30,
    analysis_schema: [],
    webhook_url: '',
    webhook_secret: '',
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
    redact_pii: true,
    max_concurrent_calls: 1,
    schedule_start_at: null,
    schedule_end_at: null,
    provider: 'local',
    from_number: '',
    transfer_number: '',
    voicemail_message: '',
    sms_followup_template: '',
    sms_followup_outcomes: ['interested'],
  };
}

function fromAgent(agent: VoiceAgent): VoiceAgentCreate {
  const {
    id: _id,
    status: _s,
    version: _v,
    profile_id,
    filler_audio: _f,
    running: _r,
    created_at: _c,
    updated_at: _u,
    ...rest
  } = agent;
  return {
    ...rest,
    profile: profile_id,
    variants: rest.variants ?? [],
    analysis_schema: rest.analysis_schema ?? [],
  };
}

function toLocalInput(iso: string | null | undefined): string {
  if (!iso) return '';
  const d = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(iso) ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return '';
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function fromLocalInput(value: string): string | null {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d.toISOString();
}

interface AgentFormProps {
  initial?: VoiceAgent | null;
  submitting?: boolean;
  submitLabel: string;
  onSubmit: (data: VoiceAgentCreate) => void;
  onCancel?: () => void;
  onTestWebhook?: () => void;
}

export function AgentForm({
  initial,
  submitting,
  submitLabel,
  onSubmit,
  onCancel,
  onTestWebhook,
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
  const isTwilio = form.provider === 'twilio';
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
      voice_style: clean(form.voice_style),
      empathetic_voice_style: clean(form.empathetic_voice_style),
      objection_notes: clean(form.objection_notes),
      persona: clean(form.persona),
      opening_line: clean(form.opening_line),
      escalation_promise: clean(form.escalation_promise),
      booking_instructions: clean(form.booking_instructions),
      webhook_url: clean(form.webhook_url),
      webhook_secret: clean(form.webhook_secret),
      from_number: clean(form.from_number),
      transfer_number: clean(form.transfer_number),
      voicemail_message: clean(form.voicemail_message),
      sms_followup_template: clean(form.sms_followup_template),
      engine: form.engine || null,
      llm_model_size: form.llm_model_size || null,
      variants: (form.variants ?? []).filter((v) => v.name.trim()),
      analysis_schema: (form.analysis_schema ?? []).filter(
        (f) => f.key.trim() && f.question.trim(),
      ),
      filler_phrases: form.filler_phrases.map((p) => p.trim()).filter(Boolean),
    });
  };

  const variants = form.variants ?? [];
  const schema = form.analysis_schema ?? [];

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
          <Field label="Voice style (Qwen engines)">
            <Input
              value={form.voice_style ?? ''}
              onChange={(e) => set('voice_style', e.target.value)}
              placeholder="friendly, unhurried"
            />
          </Field>
          <Field label="Voice style when the caller is upset">
            <Input
              value={form.empathetic_voice_style ?? ''}
              onChange={(e) => set('empathetic_voice_style', e.target.value)}
              placeholder="calm, warm and apologetic"
            />
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
              placeholder={
                isOutbound
                  ? 'Quick question about {{contact.company}}?'
                  : 'How can I help you today?'
              }
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

      {/* A/B variants */}
      <Section
        title="A/B script variants"
        hint="Each call picks a variant by weight. Empty fields fall back to the base script. Results show per variant in Analytics."
      >
        {variants.map((v, i) => (
          <div key={`${i}-${v.name}`} className="rounded-lg border border-border p-3 space-y-2">
            <div className="flex gap-2 items-center">
              <Input
                className="h-9 w-40"
                placeholder="Variant name"
                value={v.name}
                onChange={(e) =>
                  updateAt(set, 'variants', variants, i, { ...v, name: e.target.value })
                }
              />
              <Label className="text-xs text-muted-foreground">weight</Label>
              <Input
                className="h-9 w-20"
                type="number"
                min={1}
                max={100}
                value={v.weight}
                onChange={(e) =>
                  updateAt(set, 'variants', variants, i, {
                    ...v,
                    weight: Number(e.target.value) || 1,
                  })
                }
              />
              <Button
                size="icon"
                variant="ghost"
                className="ml-auto h-8 w-8 text-muted-foreground hover:text-destructive"
                onClick={() =>
                  set(
                    'variants',
                    variants.filter((_, j) => j !== i),
                  )
                }
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
            <Input
              className="h-9"
              placeholder="Opening hook override"
              value={v.opening_line ?? ''}
              onChange={(e) =>
                updateAt(set, 'variants', variants, i, {
                  ...v,
                  opening_line: e.target.value || null,
                })
              }
            />
            <Textarea
              rows={2}
              placeholder="Brief override (optional)"
              value={v.brief ?? ''}
              onChange={(e) =>
                updateAt(set, 'variants', variants, i, { ...v, brief: e.target.value || null })
              }
            />
          </div>
        ))}
        <Button
          size="sm"
          variant="outline"
          disabled={variants.length >= 8}
          onClick={() =>
            set('variants', [
              ...variants,
              {
                name: `Variant ${String.fromCharCode(65 + variants.length)}`,
                weight: 1,
              } as ScriptVariant,
            ])
          }
        >
          <Plus className="h-4 w-4" /> Add variant
        </Button>
      </Section>

      {/* Conversation engine */}
      <Section title="Conversation engine">
        <div className="grid grid-cols-2 gap-3">
          <Field label="Filler phrases (one per line, spoken while the model thinks)">
            <Textarea
              rows={3}
              value={form.filler_phrases.join('\n')}
              onChange={(e) => set('filler_phrases', e.target.value.split('\n'))}
            />
          </Field>
          <div className="space-y-3 pt-5">
            <Check
              id="fast-first-audio"
              checked={form.fast_first_audio}
              onChange={(v) => set('fast_first_audio', v)}
              label="Fast first audio — voice the first sentence separately so the reply starts sooner"
            />
            <Check
              id="tools-enabled"
              checked={form.tools_enabled}
              onChange={(v) => set('tools_enabled', v)}
              label="Tools — let the agent book appointments, schedule callbacks, transfer, and call your HTTP tools"
            />
            <Check
              id="redact-pii"
              checked={form.redact_pii}
              onChange={(v) => set('redact_pii', v)}
              label="Redact card numbers, national IDs and one-time codes from transcripts"
            />
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <Field label="Booking rules (enables book_appointment)">
            <Input
              value={form.booking_instructions ?? ''}
              onChange={(e) => set('booking_instructions', e.target.value)}
              placeholder="weekday mornings only, 30 minutes"
            />
          </Field>
          <Field label="Appointment length (min)">
            <Input
              type="number"
              min={5}
              max={480}
              value={form.appointment_duration_min}
              onChange={(e) => set('appointment_duration_min', Number(e.target.value))}
            />
          </Field>
          <Field label="Max turns per call">
            <Input
              type="number"
              min={2}
              max={200}
              value={form.max_turns}
              onChange={(e) => set('max_turns', Number(e.target.value))}
            />
          </Field>
        </div>
        <Field label="Hand off after N upset turns">
          <Input
            className="w-32"
            type="number"
            min={1}
            max={20}
            value={form.handoff_after_negative_turns}
            onChange={(e) => set('handoff_after_negative_turns', Number(e.target.value))}
          />
        </Field>
      </Section>

      {/* After the call */}
      <Section
        title="After the call"
        hint="Questions the model answers from the transcript, a signed webhook, and an optional text follow-up."
      >
        {schema.map((f, i) => (
          <div key={`${i}-${f.key}`} className="flex gap-2 items-start">
            <Input
              className="h-9 w-36"
              placeholder="key_name"
              value={f.key}
              onChange={(e) =>
                updateAt(set, 'analysis_schema', schema, i, {
                  ...f,
                  key: e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, '_'),
                })
              }
            />
            <Input
              className="h-9 flex-1"
              placeholder="Question the model should answer"
              value={f.question}
              onChange={(e) =>
                updateAt(set, 'analysis_schema', schema, i, { ...f, question: e.target.value })
              }
            />
            <Select
              value={f.type}
              onValueChange={(v) =>
                updateAt(set, 'analysis_schema', schema, i, {
                  ...f,
                  type: v as AnalysisField['type'],
                })
              }
            >
              <SelectTrigger className="h-9 w-28">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="string">text</SelectItem>
                <SelectItem value="boolean">yes/no</SelectItem>
                <SelectItem value="number">number</SelectItem>
                <SelectItem value="enum">one of</SelectItem>
              </SelectContent>
            </Select>
            {f.type === 'enum' && (
              <Input
                className="h-9 w-44"
                placeholder="low, medium, high"
                value={(f.options ?? []).join(', ')}
                onChange={(e) =>
                  updateAt(set, 'analysis_schema', schema, i, {
                    ...f,
                    options: e.target.value
                      .split(',')
                      .map((s) => s.trim())
                      .filter(Boolean),
                  })
                }
              />
            )}
            <Button
              size="icon"
              variant="ghost"
              className="h-9 w-9 text-muted-foreground hover:text-destructive"
              onClick={() =>
                set(
                  'analysis_schema',
                  schema.filter((_, j) => j !== i),
                )
              }
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        ))}
        <Button
          size="sm"
          variant="outline"
          disabled={schema.length >= 20}
          onClick={() =>
            set('analysis_schema', [...schema, { key: '', question: '', type: 'string' }])
          }
        >
          <Plus className="h-4 w-4" /> Add question
        </Button>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Webhook URL (POST call.ended, HMAC-SHA256 signed)">
            <div className="flex gap-2">
              <Input
                value={form.webhook_url ?? ''}
                onChange={(e) => set('webhook_url', e.target.value)}
                placeholder="https://example.com/voicebox/hook"
              />
              {onTestWebhook && initial?.webhook_url && (
                <Button variant="outline" size="sm" className="h-10" onClick={onTestWebhook}>
                  Test
                </Button>
              )}
            </div>
          </Field>
          <Field label="Webhook secret">
            <Input
              value={form.webhook_secret ?? ''}
              onChange={(e) => set('webhook_secret', e.target.value)}
              placeholder="shared secret"
            />
          </Field>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Field label="SMS follow-up (Twilio agents)">
            <Textarea
              rows={2}
              value={form.sms_followup_template ?? ''}
              onChange={(e) => set('sms_followup_template', e.target.value)}
              placeholder="Hi {{contact.first_name}}, thanks for your time today — here's the link we mentioned: https://…"
            />
          </Field>
          <Field label="Send after these outcomes">
            <div className="flex flex-wrap gap-1.5 pt-1">
              {(Object.keys(OUTCOME_META) as Array<keyof typeof OUTCOME_META>)
                .filter(
                  (o) =>
                    ![
                      'error',
                      'no_answer',
                      'voicemail',
                      'voicemail_left',
                      'max_turns',
                      'opt_out',
                    ].includes(o),
                )
                .map((o) => {
                  const on = form.sms_followup_outcomes.includes(o);
                  return (
                    <button
                      key={o}
                      type="button"
                      onClick={() =>
                        set(
                          'sms_followup_outcomes',
                          on
                            ? form.sms_followup_outcomes.filter((x) => x !== o)
                            : [...form.sms_followup_outcomes, o],
                        )
                      }
                      className={cn(
                        'h-7 px-2.5 rounded-full text-xs border',
                        on ? 'bg-accent/15 border-accent' : 'border-border text-muted-foreground',
                      )}
                    >
                      {OUTCOME_META[o].label}
                    </button>
                  );
                })}
            </div>
          </Field>
        </div>
      </Section>

      {/* Guard-rails */}
      <Section title={isOutbound ? 'Dialing rules' : 'Call limits'}>
        {isOutbound && (
          <>
            <div className="flex flex-wrap gap-2">
              {COMPLIANCE_PRESETS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  title={p.hint}
                  onClick={() =>
                    setForm((f) => ({
                      ...f,
                      calling_window_start: p.start,
                      calling_window_end: p.end,
                      calling_days: p.days,
                      max_attempts: p.attempts,
                    }))
                  }
                  className="h-8 px-3 rounded-full text-xs border border-border hover:bg-muted/40"
                >
                  {p.label}
                </button>
              ))}
              <span className="text-[11px] text-muted-foreground self-center">
                Presets fill the fields below; check your own jurisdiction.
              </span>
            </div>
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
            <div className="grid grid-cols-3 gap-3">
              <Field label="Campaign starts (optional)">
                <Input
                  type="datetime-local"
                  value={toLocalInput(form.schedule_start_at)}
                  onChange={(e) => set('schedule_start_at', fromLocalInput(e.target.value))}
                />
              </Field>
              <Field label="Campaign ends (optional)">
                <Input
                  type="datetime-local"
                  value={toLocalInput(form.schedule_end_at)}
                  onChange={(e) => set('schedule_end_at', fromLocalInput(e.target.value))}
                />
              </Field>
              <Field label="Parallel lines (Twilio)">
                <Input
                  type="number"
                  min={1}
                  max={20}
                  value={form.max_concurrent_calls}
                  disabled={!isTwilio}
                  onChange={(e) => set('max_concurrent_calls', Number(e.target.value))}
                />
              </Field>
            </div>
            <Check
              id="agent-require-consent"
              checked={form.require_consent}
              onChange={(v) => set('require_consent', v)}
              label="Only dial contacts marked as having given consent"
            />
            <Field label="Voicemail drop (leave this message on answering machines; blank = hang up and retry)">
              <Textarea
                rows={2}
                value={form.voicemail_message ?? ''}
                onChange={(e) => set('voicemail_message', e.target.value)}
                placeholder="Hi {{contact.first_name}}, this is {{agent.agent_name}} from {{agent.company_name}}. Sorry I missed you — I'll try again soon."
              />
            </Field>
          </>
        )}
      </Section>

      {/* Telephony */}
      <Section title="Phone line">
        <div className="grid grid-cols-3 gap-3">
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
          {isTwilio && (
            <>
              <Field label="From number (Twilio-owned)">
                <Input
                  value={form.from_number ?? ''}
                  onChange={(e) => set('from_number', e.target.value)}
                  placeholder="+15550100000"
                />
              </Field>
              <Field label="Transfer hand-offs to (warm transfer)">
                <Input
                  value={form.transfer_number ?? ''}
                  onChange={(e) => set('transfer_number', e.target.value)}
                  placeholder="+15550100001"
                />
              </Field>
            </>
          )}
        </div>
        {isTwilio && (
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

function updateAt<K extends 'variants' | 'analysis_schema'>(
  set: <F extends keyof VoiceAgentCreate>(key: F, value: VoiceAgentCreate[F]) => void,
  key: K,
  list: NonNullable<VoiceAgentCreate[K]>,
  index: number,
  value: NonNullable<VoiceAgentCreate[K]>[number],
) {
  const next = [...list] as NonNullable<VoiceAgentCreate[K]>;
  (next as unknown[])[index] = value;
  set(key, next);
}

function Section({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {title}
        </h3>
        {hint && <p className="text-[11px] text-muted-foreground mt-0.5">{hint}</p>}
      </div>
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

function Check({
  id,
  checked,
  onChange,
  label,
}: {
  id: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <div className="flex items-start gap-2 text-sm">
      <Checkbox id={id} checked={checked} onCheckedChange={onChange} className="mt-0.5" />
      <label htmlFor={id} className="cursor-pointer leading-snug">
        {label}
      </label>
    </div>
  );
}
