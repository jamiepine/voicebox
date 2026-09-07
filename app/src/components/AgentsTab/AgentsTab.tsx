import { useQueryClient } from '@tanstack/react-query';
import { Bot, FlaskConical, Pause, Play, Plus, Search, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useToast } from '@/components/ui/use-toast';
import type { VoiceAgent, VoiceCall } from '@/lib/api/types';
import { BOTTOM_SAFE_AREA_PADDING } from '@/lib/constants/ui';
import {
  agentKeys,
  useAgentRunControl,
  useCall,
  useCreateVoiceAgent,
  useDeleteVoiceAgent,
  useTestWebhook,
  useUpdateVoiceAgent,
  useVoiceAgentStats,
  useVoiceAgents,
} from '@/lib/hooks/useVoiceAgents';
import { cn } from '@/lib/utils/cn';
import { usePlayerStore } from '@/stores/playerStore';
import { AgentForm } from './AgentForm';
import { AnalyticsPanel } from './AnalyticsPanel';
import { AppointmentsPanel } from './AppointmentsPanel';
import { CallConsole } from './CallConsole';
import { CallsPanel } from './CallsPanel';
import { ContactsPanel } from './ContactsPanel';
import { KnowledgePanel } from './KnowledgePanel';
import { SimulateDialog } from './SimulateDialog';
import { MODE_META, StatusDot, statusLabel } from './shared';
import { TicketsPanel } from './TicketsPanel';
import { ToolsPanel } from './ToolsPanel';
import { VersionsPanel } from './VersionsPanel';

export function AgentsTab() {
  const { t } = useTranslation();
  const { toast } = useToast();
  const qc = useQueryClient();
  const { data: agents, isLoading } = useVoiceAgents();
  const create = useCreateVoiceAgent();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [creating, setCreating] = useState(false);
  const isPlayerVisible = !!usePlayerStore((s) => s.audioUrl);

  const filtered = useMemo(() => {
    const list = agents ?? [];
    if (!search.trim()) return list;
    const q = search.toLowerCase();
    return list.filter(
      (a) =>
        a.name.toLowerCase().includes(q) ||
        a.company_name.toLowerCase().includes(q) ||
        MODE_META[a.mode].label.toLowerCase().includes(q),
    );
  }, [agents, search]);

  useEffect(() => {
    if (!agents) return;
    if (selectedId && !agents.some((a) => a.id === selectedId)) setSelectedId(null);
    if (!selectedId && agents.length > 0) setSelectedId(agents[0].id);
  }, [agents, selectedId]);

  const selected = agents?.find((a) => a.id === selectedId) ?? null;

  return (
    <div className="h-full flex gap-0 overflow-hidden -mx-8">
      {/* Left: agent list */}
      <div className="w-[320px] shrink-0 flex flex-col relative overflow-hidden border-r border-border">
        <div className="absolute top-0 left-0 right-0 h-16 bg-gradient-to-b from-background to-transparent z-10 pointer-events-none" />
        <div className="absolute top-0 left-0 right-0 z-20 pl-8 pr-4">
          <div className="flex items-center gap-2 mb-4">
            <h1 className="text-2xl font-bold">{t('nav.agents')}</h1>
            <div className="flex-1" />
            <Button size="icon" onClick={() => setCreating(true)} title="New agent">
              <Plus className="h-4 w-4" />
            </Button>
          </div>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              placeholder="Search agents"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="h-9 pl-8 text-sm rounded-full focus-visible:ring-0 focus-visible:ring-offset-0"
            />
          </div>
        </div>
        <div
          className={cn(
            'flex-1 overflow-y-auto pt-32 pl-8 pr-4 pb-8 space-y-1.5',
            isPlayerVisible && BOTTOM_SAFE_AREA_PADDING,
          )}
        >
          {isLoading && <div className="text-sm text-muted-foreground">Loading…</div>}
          {!isLoading && filtered.length === 0 && (
            <div className="text-sm text-muted-foreground rounded-lg border border-dashed border-border p-4 text-center">
              <Bot className="h-6 w-6 mx-auto mb-2 opacity-40" />
              No agents yet. Create one to start calling.
            </div>
          )}
          {filtered.map((a) => (
            <AgentCard
              key={a.id}
              agent={a}
              selected={a.id === selectedId}
              onSelect={() => setSelectedId(a.id)}
            />
          ))}
        </div>
      </div>

      {/* Right: detail */}
      <div className="flex-1 min-w-0 overflow-hidden">
        {selected ? (
          <AgentDetail key={selected.id} agent={selected} isPlayerVisible={isPlayerVisible} />
        ) : (
          <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
            {isLoading ? '' : 'Select or create an agent.'}
          </div>
        )}
      </div>

      <Dialog open={creating} onOpenChange={setCreating}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>New voice agent</DialogTitle>
            <DialogDescription>
              Pick a mode, a voice, and tell it what it may say. You can edit everything later.
            </DialogDescription>
          </DialogHeader>
          <AgentForm
            submitLabel="Create agent"
            submitting={create.isPending}
            onCancel={() => setCreating(false)}
            onSubmit={async (data) => {
              try {
                const a = await create.mutateAsync(data);
                setCreating(false);
                setSelectedId(a.id);
                qc.invalidateQueries({ queryKey: agentKeys.all });
              } catch (err) {
                toast({
                  title: 'Could not create agent',
                  description: err instanceof Error ? err.message : String(err),
                  variant: 'destructive',
                });
              }
            }}
          />
        </DialogContent>
      </Dialog>
    </div>
  );
}

function AgentCard({
  agent,
  selected,
  onSelect,
}: {
  agent: VoiceAgent;
  selected: boolean;
  onSelect: () => void;
}) {
  const meta = MODE_META[agent.mode];
  const Icon = meta.icon;
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        'w-full text-left rounded-xl border px-3 py-2.5 transition-colors',
        selected ? 'border-accent/50 bg-accent/10' : 'border-border hover:bg-muted/40',
      )}
    >
      <div className="flex items-center gap-2">
        <Icon
          className={cn('h-4 w-4 shrink-0', selected ? 'text-accent' : 'text-muted-foreground')}
        />
        <span className="font-medium truncate">{agent.name}</span>
        <div className="ml-auto flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <StatusDot status={agent.status} running={agent.running} />
          {statusLabel(agent.status, agent.running)}
        </div>
      </div>
      <div className="text-xs text-muted-foreground mt-0.5 truncate">
        {meta.label} · {agent.agent_name} @ {agent.company_name}
      </div>
    </button>
  );
}

function AgentDetail({ agent, isPlayerVisible }: { agent: VoiceAgent; isPlayerVisible: boolean }) {
  const { toast } = useToast();
  const update = useUpdateVoiceAgent();
  const del = useDeleteVoiceAgent();
  const { start, pause } = useAgentRunControl();
  const [tab, setTab] = useState('console');
  const [activeCallId, setActiveCallId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const testWebhook = useTestWebhook(agent.id);
  const { data: stats } = useVoiceAgentStats(agent.id, agent.running || !!activeCallId);
  const { data: activeCall } = useCall(activeCallId, false);
  const isOutbound = agent.mode === 'outbound_sales';

  const toggleRun = async () => {
    try {
      if (agent.status === 'active') await pause.mutateAsync(agent.id);
      else await start.mutateAsync(agent.id);
    } catch (err) {
      toast({
        title: 'Could not change agent state',
        description: err instanceof Error ? err.message : String(err),
        variant: 'destructive',
      });
    }
  };

  const openInConsole = (callId: string) => {
    setActiveCallId(callId);
    setTab('console');
  };

  const meta = MODE_META[agent.mode];
  const Icon = meta.icon;

  return (
    <div className="h-full flex flex-col relative overflow-hidden">
      <div className="absolute top-0 left-0 right-0 h-16 bg-gradient-to-b from-background to-transparent z-10 pointer-events-none" />
      <div className="absolute top-0 left-0 right-0 z-20 px-8">
        <div className="flex items-center gap-3 mb-3">
          <div className="h-10 w-10 rounded-full bg-accent/15 flex items-center justify-center">
            <Icon className="h-5 w-5 text-accent" />
          </div>
          <div className="min-w-0">
            <div className="text-xl font-bold truncate">{agent.name}</div>
            <div className="text-xs text-muted-foreground">
              {meta.label} · speaks as {agent.agent_name} for {agent.company_name}
            </div>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSimulating(true)}
              title="Run a test call with a simulated customer"
            >
              <FlaskConical className="h-4 w-4" /> Test call
            </Button>
            <Button
              variant={agent.status === 'active' ? 'outline' : 'default'}
              size="sm"
              onClick={toggleRun}
              disabled={start.isPending || pause.isPending}
            >
              {agent.status === 'active' ? (
                <>
                  <Pause className="h-4 w-4" /> {isOutbound ? 'Pause dialing' : 'Set inactive'}
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" /> {isOutbound ? 'Start dialing' : 'Activate'}
                </>
              )}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="text-muted-foreground hover:text-destructive"
              onClick={() => setConfirmDelete(true)}
              title="Delete agent"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      <div
        className={cn(
          'flex-1 overflow-y-auto pt-20 px-8 pb-8',
          isPlayerVisible && BOTTOM_SAFE_AREA_PADDING,
        )}
      >
        {/* Stat tiles */}
        <div className="grid grid-cols-6 gap-2 mb-5">
          <Stat label="Contacts" value={stats?.contacts_total ?? 0} />
          <Stat
            label="Calls today"
            value={stats?.calls_today ?? 0}
            sub={`${stats?.calls_total ?? 0} total`}
          />
          <Stat
            label={isOutbound ? 'Booked' : 'Resolved'}
            value={`${Math.round((stats?.resolution_rate ?? 0) * 100)}%`}
            sub={`avg ${stats?.avg_turns ?? 0} turns`}
          />
          <Stat label="Avg score" value={stats?.avg_score ?? '—'} sub="goal achievement" />
          <Stat
            label="Bookings"
            value={stats?.appointments_upcoming ?? 0}
            sub={`${stats?.open_tickets ?? 0} open tickets`}
          />
          <Stat
            label={isOutbound ? 'Dialable now' : 'Status'}
            value={
              isOutbound ? (stats?.next_dialable ?? 0) : statusLabel(agent.status, agent.running)
            }
            sub={`v${agent.version}`}
          />
        </div>

        <Tabs value={tab} onValueChange={setTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="console">Console</TabsTrigger>
            {isOutbound && <TabsTrigger value="contacts">Contacts</TabsTrigger>}
            <TabsTrigger value="knowledge">Knowledge</TabsTrigger>
            <TabsTrigger value="tools">Tools</TabsTrigger>
            <TabsTrigger value="calls">Calls</TabsTrigger>
            <TabsTrigger value="appointments">Bookings</TabsTrigger>
            <TabsTrigger value="tickets">Tickets</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="setup">Setup</TabsTrigger>
            <TabsTrigger value="versions">Versions</TabsTrigger>
          </TabsList>
          <TabsContent value="console">
            <CallConsole
              agent={agent}
              activeCall={(activeCall as VoiceCall | undefined) ?? null}
              onCallChange={setActiveCallId}
            />
          </TabsContent>
          {isOutbound && (
            <TabsContent value="contacts">
              <ContactsPanel agent={agent} onCallStarted={openInConsole} />
            </TabsContent>
          )}
          <TabsContent value="knowledge">
            <KnowledgePanel agentId={agent.id} />
          </TabsContent>
          <TabsContent value="tools">
            <ToolsPanel agent={agent} />
          </TabsContent>
          <TabsContent value="appointments">
            <AppointmentsPanel agent={agent} />
          </TabsContent>
          <TabsContent value="analytics">
            <AnalyticsPanel agent={agent} />
          </TabsContent>
          <TabsContent value="versions">
            <VersionsPanel agent={agent} />
          </TabsContent>
          <TabsContent value="calls">
            <CallsPanel agentId={agent.id} live={agent.running} onOpenInConsole={openInConsole} />
          </TabsContent>
          <TabsContent value="tickets">
            <TicketsPanel agentId={agent.id} />
          </TabsContent>
          <TabsContent value="setup">
            <AgentForm
              key={agent.version}
              initial={agent}
              submitLabel="Save changes"
              submitting={update.isPending}
              onTestWebhook={async () => {
                try {
                  await testWebhook.mutateAsync();
                  toast({
                    title: 'Test delivery queued',
                    description: 'See Analytics → Webhook deliveries for the result.',
                  });
                } catch (err) {
                  toast({
                    title: 'Webhook test failed',
                    description: err instanceof Error ? err.message : String(err),
                    variant: 'destructive',
                  });
                }
              }}
              onSubmit={async (data) => {
                try {
                  await update.mutateAsync({ agentId: agent.id, data });
                  toast({ title: 'Agent saved' });
                } catch (err) {
                  toast({
                    title: 'Could not save',
                    description: err instanceof Error ? err.message : String(err),
                    variant: 'destructive',
                  });
                }
              }}
            />
          </TabsContent>
        </Tabs>
      </div>

      <SimulateDialog
        agent={agent}
        open={simulating}
        onOpenChange={setSimulating}
        onDone={openInConsole}
      />

      <AlertDialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete “{agent.name}”?</AlertDialogTitle>
            <AlertDialogDescription>
              Removes the agent with its contacts, calls, tickets and knowledge. The do-not-call
              list is kept.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={async () => {
                try {
                  await del.mutateAsync(agent.id);
                } catch (err) {
                  toast({
                    title: 'Could not delete',
                    description: err instanceof Error ? err.message : String(err),
                    variant: 'destructive',
                  });
                }
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="rounded-xl border border-border bg-card/40 px-3 py-2.5">
      <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="text-xl font-semibold leading-tight mt-0.5">{value}</div>
      {sub && <div className="text-[11px] text-muted-foreground">{sub}</div>}
    </div>
  );
}
