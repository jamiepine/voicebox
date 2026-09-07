import { Ban, PhoneCall, Plus, Trash2, Upload } from 'lucide-react';
import { useRef, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { Contact, VoiceAgent } from '@/lib/api/types';
import {
  useContactMutations,
  useContacts,
  useDoNotCall,
  useDoNotCallMutations,
} from '@/lib/hooks/useVoiceAgents';
import { formatDate } from '@/lib/utils/format';
import { contactStatusLabel } from './shared';

interface ContactsPanelProps {
  agent: VoiceAgent;
  onCallStarted: (callId: string) => void;
}

export function ContactsPanel({ agent, onCallStarted }: ContactsPanelProps) {
  const { toast } = useToast();
  const { data, isLoading } = useContacts(agent.id);
  const { create, importCsv, remove } = useContactMutations(agent.id);
  const dnc = useDoNotCall();
  const dncMut = useDoNotCallMutations();
  const fileRef = useRef<HTMLInputElement>(null);
  const [draft, setDraft] = useState({ name: '', phone: '', company: '', consent: false });
  const [showDnc, setShowDnc] = useState(false);

  const addContact = async () => {
    if (!draft.phone.trim()) return;
    try {
      await create.mutateAsync({
        name: draft.name.trim() || draft.phone.trim(),
        phone: draft.phone.trim(),
        company: draft.company.trim() || null,
        consent: draft.consent,
      });
      setDraft({ name: '', phone: '', company: '', consent: false });
    } catch (err) {
      toast({ title: 'Could not add contact', description: String(err), variant: 'destructive' });
    }
  };

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    try {
      const r = await importCsv.mutateAsync(file);
      const reasons = Object.entries(r.skipped_reasons)
        .map(([k, v]) => `${v} ${k.replace('_', ' ')}`)
        .join(', ');
      toast({
        title: `Imported ${r.imported} contact${r.imported === 1 ? '' : 's'}`,
        description: r.skipped ? `Skipped ${r.skipped} (${reasons})` : undefined,
      });
    } catch (err) {
      toast({ title: 'Import failed', description: String(err), variant: 'destructive' });
    } finally {
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const callNow = async (c: Contact) => {
    try {
      const r = await apiClient.startNextCall(agent.id, c.id);
      onCallStarted(r.call_id);
    } catch (err) {
      toast({ title: 'Could not start call', description: String(err), variant: 'destructive' });
    }
  };

  const block = async (c: Contact) => {
    try {
      await dncMut.add.mutateAsync({ phone: c.phone, reason: 'Added from contacts' });
    } catch (err) {
      toast({ title: 'Could not block number', description: String(err), variant: 'destructive' });
    }
  };

  const contacts = data?.contacts ?? [];

  return (
    <div className="space-y-4">
      {/* Add row */}
      <div className="flex items-end gap-2 flex-wrap">
        <Input
          className="h-9 w-40"
          placeholder="Name"
          value={draft.name}
          onChange={(e) => setDraft({ ...draft, name: e.target.value })}
        />
        <Input
          className="h-9 w-40"
          placeholder="Phone"
          value={draft.phone}
          onChange={(e) => setDraft({ ...draft, phone: e.target.value })}
          onKeyDown={(e) => e.key === 'Enter' && addContact()}
        />
        <Input
          className="h-9 w-40"
          placeholder="Company"
          value={draft.company}
          onChange={(e) => setDraft({ ...draft, company: e.target.value })}
        />
        <div className="flex items-center gap-2 text-xs text-muted-foreground h-9">
          <Checkbox
            id="contact-draft-consent"
            checked={draft.consent}
            onCheckedChange={(v) => setDraft({ ...draft, consent: v })}
          />
          <label htmlFor="contact-draft-consent" className="cursor-pointer">
            consented
          </label>
        </div>
        <Button size="sm" onClick={addContact} disabled={!draft.phone.trim() || create.isPending}>
          <Plus className="h-4 w-4" /> Add
        </Button>
        <div className="ml-auto flex items-center gap-2">
          <input
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(e) => onFile(e.target.files?.[0])}
          />
          <Button
            size="sm"
            variant="outline"
            onClick={() => fileRef.current?.click()}
            disabled={importCsv.isPending}
          >
            <Upload className="h-4 w-4" /> Import CSV
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setShowDnc((v) => !v)}>
            <Ban className="h-4 w-4" /> Do-not-call ({dnc.data?.length ?? 0})
          </Button>
        </div>
      </div>
      <p className="text-[11px] text-muted-foreground -mt-2">
        CSV headers: name, phone, company, notes, timezone, consent. Numbers on the do-not-call list
        are never dialled, on any agent.
      </p>

      {showDnc && (
        <div className="rounded-lg border border-border p-3 space-y-2">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Do-not-call list
          </div>
          {(dnc.data ?? []).length === 0 && (
            <div className="text-sm text-muted-foreground">Empty.</div>
          )}
          {(dnc.data ?? []).map((e) => (
            <div key={e.phone} className="flex items-center gap-3 text-sm">
              <span className="font-mono">{e.phone}</span>
              <span className="text-xs text-muted-foreground">
                {e.source}
                {e.reason ? ` · ${e.reason}` : ''} · {formatDate(e.created_at)}
              </span>
              <Button
                size="sm"
                variant="ghost"
                className="ml-auto h-7"
                onClick={() => dncMut.remove.mutate(e.phone)}
              >
                Remove
              </Button>
            </div>
          ))}
        </div>
      )}

      {/* Table */}
      {isLoading ? (
        <div className="text-sm text-muted-foreground">Loading…</div>
      ) : contacts.length === 0 ? (
        <div className="text-sm text-muted-foreground py-8 text-center rounded-lg border border-dashed border-border">
          No contacts yet. Add one above or import a CSV.
        </div>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Phone</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="w-[90px]">Attempts</TableHead>
              <TableHead>Next / last</TableHead>
              <TableHead className="w-[120px]" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {contacts.map((c) => (
              <TableRow key={c.id}>
                <TableCell>
                  <div className="font-medium">{c.name}</div>
                  {c.company && <div className="text-xs text-muted-foreground">{c.company}</div>}
                </TableCell>
                <TableCell className="font-mono text-xs">{c.phone}</TableCell>
                <TableCell>
                  <Badge variant="outline" className="capitalize">
                    {contactStatusLabel(c.status)}
                  </Badge>
                  {c.consent && (
                    <span className="ml-2 text-[10px] text-emerald-300 uppercase">consent</span>
                  )}
                </TableCell>
                <TableCell>{c.attempts}</TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  {c.next_attempt_at
                    ? `next ${formatDate(c.next_attempt_at)}`
                    : c.last_attempt_at
                      ? `last ${formatDate(c.last_attempt_at)}`
                      : '—'}
                  {c.last_outcome && (
                    <div className="capitalize">{contactStatusLabel(c.last_outcome)}</div>
                  )}
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-1 justify-end">
                    {agent.mode === 'outbound_sales' && c.status !== 'do_not_call' && (
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-8 w-8"
                        title="Call now"
                        onClick={() => callNow(c)}
                      >
                        <PhoneCall className="h-4 w-4" />
                      </Button>
                    )}
                    {c.status !== 'do_not_call' && (
                      <Button
                        size="icon"
                        variant="ghost"
                        className="h-8 w-8"
                        title="Add to do-not-call"
                        onClick={() => block(c)}
                      >
                        <Ban className="h-4 w-4" />
                      </Button>
                    )}
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-8 w-8 text-muted-foreground hover:text-destructive"
                      title="Delete"
                      onClick={() => remove.mutate(c.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
