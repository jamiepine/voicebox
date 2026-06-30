import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import type {
  MCPClientBindingListResponse,
  MCPClientBindingUpsert,
} from '@/lib/api/types';

const MCP_BINDINGS_KEY = ['settings', 'mcp', 'bindings'] as const;

/** Manage per-MCP-client voice bindings (Claude Code → Morgan, etc.). */
export function useMCPBindings() {
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: MCP_BINDINGS_KEY,
    queryFn: () => apiClient.listMCPBindings(),
    // Keep fresh while the Settings page is open — the ``last_seen_at``
    // timestamp is useful for confirming an install works, and we want it
    // to tick forward when a client connects.
    refetchInterval: 10_000,
  });

  const upsertMutation = useMutation({
    mutationFn: (data: MCPClientBindingUpsert) =>
      apiClient.upsertMCPBinding(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: MCP_BINDINGS_KEY });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (clientId: string) => apiClient.deleteMCPBinding(clientId),
    onMutate: async (clientId) => {
      await queryClient.cancelQueries({ queryKey: MCP_BINDINGS_KEY });
      const prev =
        queryClient.getQueryData<MCPClientBindingListResponse>(MCP_BINDINGS_KEY);
      if (prev) {
        queryClient.setQueryData<MCPClientBindingListResponse>(
          MCP_BINDINGS_KEY,
          { items: prev.items.filter((b) => b.client_id !== clientId) },
        );
      }
      return { prev };
    },
    onError: (_err, _id, ctx) => {
      if (ctx?.prev) queryClient.setQueryData(MCP_BINDINGS_KEY, ctx.prev);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: MCP_BINDINGS_KEY });
    },
  });

  return {
    bindings: query.data?.items ?? [],
    isLoading: query.isLoading,
    upsert: upsertMutation.mutate,
    upsertAsync: upsertMutation.mutateAsync,
    remove: deleteMutation.mutate,
  };
}
