import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import type {
  ContactCreate,
  ContactUpdate,
  KnowledgeArticleCreate,
  VoiceAgentCreate,
  VoiceAgentUpdate,
} from '@/lib/api/types';

export const agentKeys = {
  all: ['voice-agents'] as const,
  detail: (id: string) => ['voice-agents', id] as const,
  stats: (id: string) => ['voice-agents', id, 'stats'] as const,
  contacts: (id: string, status?: string) =>
    ['voice-agents', id, 'contacts', status ?? 'all'] as const,
  knowledge: (id: string) => ['voice-agents', id, 'knowledge'] as const,
  calls: (id: string) => ['voice-agents', id, 'calls'] as const,
  call: (callId: string) => ['voice-calls', callId] as const,
  tickets: (id?: string) => ['voice-tickets', id ?? 'all'] as const,
  dnc: ['voice-dnc'] as const,
};

export function useVoiceAgents() {
  return useQuery({ queryKey: agentKeys.all, queryFn: () => apiClient.listVoiceAgents() });
}

export function useVoiceAgent(agentId: string | null) {
  return useQuery({
    queryKey: agentKeys.detail(agentId ?? ''),
    queryFn: () => apiClient.getVoiceAgent(agentId as string),
    enabled: !!agentId,
  });
}

export function useVoiceAgentStats(agentId: string | null, live = false) {
  return useQuery({
    queryKey: agentKeys.stats(agentId ?? ''),
    queryFn: () => apiClient.getVoiceAgentStats(agentId as string),
    enabled: !!agentId,
    refetchInterval: live ? 5000 : false,
  });
}

export function useCreateVoiceAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: VoiceAgentCreate) => apiClient.createVoiceAgent(data),
    onSuccess: () => qc.invalidateQueries({ queryKey: agentKeys.all }),
  });
}

export function useUpdateVoiceAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ agentId, data }: { agentId: string; data: VoiceAgentUpdate }) =>
      apiClient.updateVoiceAgent(agentId, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: agentKeys.all }),
  });
}

export function useDeleteVoiceAgent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (agentId: string) => apiClient.deleteVoiceAgent(agentId),
    onSuccess: () => qc.invalidateQueries({ queryKey: agentKeys.all }),
  });
}

export function useAgentRunControl() {
  const qc = useQueryClient();
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: agentKeys.all });
  };
  const start = useMutation({
    mutationFn: (agentId: string) => apiClient.startVoiceAgent(agentId),
    onSuccess: invalidate,
  });
  const pause = useMutation({
    mutationFn: (agentId: string) => apiClient.pauseVoiceAgent(agentId),
    onSuccess: invalidate,
  });
  return { start, pause };
}

export function useContacts(agentId: string | null, status?: string) {
  return useQuery({
    queryKey: agentKeys.contacts(agentId ?? '', status),
    queryFn: () => apiClient.listContacts(agentId as string, { status, limit: 1000 }),
    enabled: !!agentId,
  });
}

export function useContactMutations(agentId: string) {
  const qc = useQueryClient();
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['voice-agents', agentId, 'contacts'] });
    qc.invalidateQueries({ queryKey: agentKeys.stats(agentId) });
  };
  const create = useMutation({
    mutationFn: (data: ContactCreate) => apiClient.createContact(agentId, data),
    onSuccess: invalidate,
  });
  const importCsv = useMutation({
    mutationFn: (file: File) => apiClient.importContactsCsv(agentId, file),
    onSuccess: invalidate,
  });
  const update = useMutation({
    mutationFn: ({ contactId, data }: { contactId: string; data: ContactUpdate }) =>
      apiClient.updateContact(contactId, data),
    onSuccess: invalidate,
  });
  const remove = useMutation({
    mutationFn: (contactId: string) => apiClient.deleteContact(contactId),
    onSuccess: invalidate,
  });
  return { create, importCsv, update, remove };
}

export function useKnowledge(agentId: string | null) {
  return useQuery({
    queryKey: agentKeys.knowledge(agentId ?? ''),
    queryFn: () => apiClient.listKnowledge(agentId as string),
    enabled: !!agentId,
  });
}

export function useKnowledgeMutations(agentId: string) {
  const qc = useQueryClient();
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: agentKeys.knowledge(agentId) });
  };
  const create = useMutation({
    mutationFn: (data: KnowledgeArticleCreate) => apiClient.createKnowledge(agentId, data),
    onSuccess: invalidate,
  });
  const update = useMutation({
    mutationFn: ({
      articleId,
      data,
    }: {
      articleId: string;
      data: Partial<KnowledgeArticleCreate>;
    }) => apiClient.updateKnowledge(articleId, data),
    onSuccess: invalidate,
  });
  const remove = useMutation({
    mutationFn: (articleId: string) => apiClient.deleteKnowledge(articleId),
    onSuccess: invalidate,
  });
  return { create, update, remove };
}

export function useCalls(agentId: string | null, live = false) {
  return useQuery({
    queryKey: agentKeys.calls(agentId ?? ''),
    queryFn: () => apiClient.listCalls(agentId as string, { limit: 200 }),
    enabled: !!agentId,
    refetchInterval: live ? 5000 : false,
  });
}

export function useCall(callId: string | null, live = false) {
  return useQuery({
    queryKey: agentKeys.call(callId ?? ''),
    queryFn: () => apiClient.getCall(callId as string),
    enabled: !!callId,
    refetchInterval: live ? 2000 : false,
  });
}

export function useTickets(agentId?: string, status?: string) {
  return useQuery({
    queryKey: [...agentKeys.tickets(agentId), status ?? 'all'],
    queryFn: () => apiClient.listTickets({ agentId, status, limit: 200 }),
  });
}

export function useUpdateTicket() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      ticketId,
      data,
    }: {
      ticketId: string;
      data: { status?: string; priority?: string; description?: string };
    }) => apiClient.updateTicket(ticketId, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['voice-tickets'] });
      qc.invalidateQueries({ queryKey: agentKeys.all });
    },
  });
}

export function useDoNotCall() {
  return useQuery({ queryKey: agentKeys.dnc, queryFn: () => apiClient.listDoNotCall() });
}

export function useDoNotCallMutations() {
  const qc = useQueryClient();
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: agentKeys.dnc });
    qc.invalidateQueries({ queryKey: agentKeys.all });
  };
  const add = useMutation({
    mutationFn: ({ phone, reason }: { phone: string; reason?: string }) =>
      apiClient.addDoNotCall(phone, reason),
    onSuccess: invalidate,
  });
  const remove = useMutation({
    mutationFn: (phone: string) => apiClient.removeDoNotCall(phone),
    onSuccess: invalidate,
  });
  return { add, remove };
}
