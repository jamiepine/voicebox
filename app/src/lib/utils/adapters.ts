import type { AdapterInfo } from '@/lib/api/types';

export function adapterDisplayName(adapter: AdapterInfo): string {
  if (adapter.label) return adapter.label;
  return `r${adapter.lora_rank} / ${adapter.epochs}ep / ${adapter.num_samples}s`;
}
