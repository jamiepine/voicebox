// Live on-chain stats for $VOICEBOX, fetched server-side via Helius.
//
// Design goals:
//   • Never throws. Every sub-fetch is isolated; a failure degrades that one
//     metric to `null` and is recorded in `warnings`, so the page always renders.
//   • Cheap. Results are cached in-memory for TOKEN_STATS_CACHE_MS, and holder
//     enumeration is page-capped so a viral token can't blow up a request.
//   • Honest. Numbers come straight from chain reads (Helius RPC) and Jupiter
//     for price — nothing is asserted that can't be verified on Solscan.

import {
  TOKEN_BURN_ADDRESS,
  TOKEN_CONTRACT_ADDRESS,
  TOKEN_CREATOR_ADDRESS,
  TOKEN_DEV_WALLETS,
  TOKEN_INITIAL_SUPPLY,
  TOKEN_LOCKED_ACCOUNTS,
  TOKEN_STATS_CACHE_MS,
} from './constants';

const MINT = TOKEN_CONTRACT_ADDRESS;
const WSOL_MINT = 'So11111111111111111111111111111111111111112';

// Let Next cache the underlying network reads and revalidate them on the same
// cadence as the page (ISR). Keeps /token static-fast and CDN-cacheable while
// staying fresh, instead of forcing the route fully dynamic with `no-store`.
const REVALIDATE_S = Math.round(TOKEN_STATS_CACHE_MS / 1000);

// Public Solana mainnet RPC — used as a fallback so supply/balances/locks work
// without any API key. Rate-limited, but our 10-minute cache keeps us under it.
const PUBLIC_RPC = 'https://api.mainnet-beta.solana.com';

function heliusRpcUrl(): string | null {
  const key = process.env.HELIUS_API_KEY?.trim();
  if (!key) return null;
  return `https://mainnet.helius-rpc.com/?api-key=${key}`;
}

// Standard JSON-RPC reads (supply, balances) — Helius if configured, else public.
function standardRpcUrl(): string {
  return heliusRpcUrl() ?? PUBLIC_RPC;
}

// Holder enumeration needs Helius' DAS `getTokenAccounts` extension; public RPC
// can't do it efficiently. Null when no key — holders degrade to "—".
function dasRpcUrl(): string | null {
  return heliusRpcUrl();
}

export interface TopHolder {
  owner: string;
  amount: number;
  pct: number; // share of current supply, 0–100
}

export interface LockedEntry {
  label: string;
  account: string;
  amount: number | null;
  unlocksAt?: string;
  url?: string;
}

export interface TokenStats {
  /** True only if Helius is configured and the core supply read succeeded. */
  live: boolean;
  decimals: number;
  initialSupply: number;
  /** Current on-chain mint supply (UI amount). */
  totalSupply: number | null;
  /** initialSupply − totalSupply: tokens permanently removed by burns. */
  burned: number | null;
  burnedPct: number | null;
  /** Sum of configured locked accounts. */
  locked: number | null;
  lockedPct: number | null;
  lockedBreakdown: LockedEntry[];
  /** Sum of configured dev/treasury wallets. */
  devBalance: number | null;
  devPct: number | null;
  /** Unique-owner holder count (page-capped; see holdersCapped). */
  holders: number | null;
  holdersCapped: boolean;
  topHolders: TopHolder[];
  /** Spot price in USD (Jupiter). */
  priceUsd: number | null;
  /** priceUsd × totalSupply. */
  marketCapUsd: number | null;
  /** Lifetime pump.fun creator fees earned by the creator wallet, in SOL. */
  creatorRewardsSol: number | null;
  /** creatorRewardsSol × SOL/USD price. */
  creatorRewardsUsd: number | null;
  /** Float supply = total − locked − dev − burned-at-dead-address. */
  circulating: number | null;
  updatedAt: number;
  /** Human-readable notes about anything unconfigured or failed. */
  warnings: string[];
}

// Caching is handled by Next's fetch cache (revalidate per request below), so
// this just aggregates the reads. Never throws — degrades to emptyStats.
export async function getTokenStats(): Promise<TokenStats> {
  try {
    return await buildTokenStats();
  } catch (err) {
    console.error('getTokenStats failed:', err);
    return emptyStats(['Live stats are temporarily unavailable.']);
  }
}

function emptyStats(warnings: string[]): TokenStats {
  return {
    live: false,
    decimals: 6,
    initialSupply: TOKEN_INITIAL_SUPPLY,
    totalSupply: null,
    burned: null,
    burnedPct: null,
    locked: null,
    lockedPct: null,
    lockedBreakdown: TOKEN_LOCKED_ACCOUNTS.map((l) => ({ ...l, amount: null })),
    devBalance: null,
    devPct: null,
    holders: null,
    holdersCapped: false,
    topHolders: [],
    priceUsd: null,
    marketCapUsd: null,
    creatorRewardsSol: null,
    creatorRewardsUsd: null,
    circulating: null,
    updatedAt: Date.now(),
    warnings,
  };
}

async function buildTokenStats(): Promise<TokenStats> {
  const warnings: string[] = [];
  const rpc = standardRpcUrl(); // supply/balances/locks — public RPC if no key
  const das = dasRpcUrl(); // holder enumeration — Helius only

  if (!das) {
    warnings.push('Set HELIUS_API_KEY to enable the holder count & top holders.');
  }

  // Core supply first — everything downstream is a percentage of it.
  const supply = await getTokenSupply(rpc).catch((e) => {
    warnings.push('Could not read token supply.');
    console.error('getTokenSupply:', e);
    return null;
  });

  const decimals = supply?.decimals ?? 6;
  const totalSupply = supply?.uiAmount ?? null;

  // Run the independent reads concurrently.
  const [holderData, devBalance, lockedAmounts, priceUsd, creatorRewardsSol, solPrice] =
    await Promise.all([
    das
      ? getHolders(das).catch((e) => {
          warnings.push('Could not enumerate holders.');
          console.error('getHolders:', e);
          return null;
        })
      : Promise.resolve(null),
    TOKEN_DEV_WALLETS.length
      ? getOwnersBalance(rpc, TOKEN_DEV_WALLETS).catch((e) => {
          warnings.push('Could not read dev wallet balance.');
          console.error('getOwnersBalance(dev):', e);
          return null;
        })
      : Promise.resolve(null),
    TOKEN_LOCKED_ACCOUNTS.length
      ? Promise.all(
          TOKEN_LOCKED_ACCOUNTS.map((l) =>
            getAddressBalance(rpc, l.account)
              .catch(() => null)
              .then((amount) => ({ ...l, amount })),
          ),
        )
      : Promise.resolve(
          [] as Array<(typeof TOKEN_LOCKED_ACCOUNTS)[number] & { amount: number | null }>,
        ),
    getJupiterPrice(MINT).catch(() => {
      warnings.push('Could not read price from Jupiter.');
      return null;
    }),
    getCreatorRewardsSol().catch((e) => {
      warnings.push('Could not read creator rewards.');
      console.error('getCreatorRewardsSol:', e);
      return null;
    }),
    getJupiterPrice(WSOL_MINT).catch(() => null),
  ]);

  if (!TOKEN_DEV_WALLETS.length) warnings.push('No dev/treasury wallet configured.');
  if (!TOKEN_LOCKED_ACCOUNTS.length) warnings.push('No locked accounts configured.');

  const locked =
    lockedAmounts.length && lockedAmounts.some((l) => l.amount != null)
      ? lockedAmounts.reduce((sum, l) => sum + (l.amount ?? 0), 0)
      : lockedAmounts.length
        ? null
        : null;

  const burned =
    totalSupply != null ? Math.max(0, TOKEN_INITIAL_SUPPLY - totalSupply) : null;

  const pct = (n: number | null): number | null =>
    n != null && totalSupply ? (n / totalSupply) * 100 : null;
  const pctOfInitial = (n: number | null): number | null =>
    n != null ? (n / TOKEN_INITIAL_SUPPLY) * 100 : null;

  const marketCapUsd =
    priceUsd != null && totalSupply != null ? priceUsd * totalSupply : null;

  const creatorRewardsUsd =
    creatorRewardsSol != null && solPrice != null
      ? creatorRewardsSol * solPrice
      : null;

  const circulating =
    totalSupply != null
      ? Math.max(0, totalSupply - (locked ?? 0) - (devBalance ?? 0))
      : null;

  const topHolders: TopHolder[] = (holderData?.top ?? []).map((h) => ({
    owner: h.owner,
    amount: h.amount,
    pct: totalSupply ? (h.amount / totalSupply) * 100 : 0,
  }));

  return {
    live: totalSupply != null,
    decimals,
    initialSupply: TOKEN_INITIAL_SUPPLY,
    totalSupply,
    burned,
    burnedPct: pctOfInitial(burned),
    locked,
    lockedPct: pct(locked),
    lockedBreakdown: lockedAmounts.length
      ? lockedAmounts
      : TOKEN_LOCKED_ACCOUNTS.map((l) => ({ ...l, amount: null })),
    devBalance,
    devPct: pct(devBalance),
    holders: holderData?.count ?? null,
    holdersCapped: holderData?.capped ?? false,
    topHolders,
    priceUsd,
    marketCapUsd,
    creatorRewardsSol,
    creatorRewardsUsd,
    circulating,
    updatedAt: Date.now(),
    warnings,
  };
}

// ── Solana / Helius RPC primitives ───────────────────────────────────────────

async function rpcCall<T>(rpc: string, method: string, params: unknown): Promise<T> {
  const res = await fetch(rpc, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    next: { revalidate: REVALIDATE_S },
    body: JSON.stringify({ jsonrpc: '2.0', id: 'voicebox', method, params }),
  });
  if (!res.ok) throw new Error(`RPC ${method} HTTP ${res.status}`);
  const json = (await res.json()) as { result?: T; error?: { message: string } };
  if (json.error) throw new Error(`RPC ${method}: ${json.error.message}`);
  if (json.result === undefined) throw new Error(`RPC ${method}: empty result`);
  return json.result;
}

interface SupplyResult {
  value: { amount: string; decimals: number; uiAmount: number | null };
}
async function getTokenSupply(
  rpc: string,
): Promise<{ uiAmount: number; decimals: number }> {
  const r = await rpcCall<SupplyResult>(rpc, 'getTokenSupply', [MINT]);
  const decimals = r.value.decimals;
  const uiAmount =
    r.value.uiAmount ?? Number(r.value.amount) / 10 ** decimals;
  return { uiAmount, decimals };
}

// Sum a single owner's balance of the mint across all their token accounts.
async function getOwnerBalance(rpc: string, owner: string): Promise<number> {
  const r = await rpcCall<{
    value: Array<{
      account: { data: { parsed: { info: { tokenAmount: { uiAmount: number | null } } } } };
    }>;
  }>(rpc, 'getTokenAccountsByOwner', [
    owner,
    { mint: MINT },
    { encoding: 'jsonParsed' },
  ]);
  return r.value.reduce(
    (sum, a) => sum + (a.account.data.parsed.info.tokenAmount.uiAmount ?? 0),
    0,
  );
}

async function getOwnersBalance(rpc: string, owners: string[]): Promise<number> {
  const balances = await Promise.all(owners.map((o) => getOwnerBalance(rpc, o)));
  return balances.reduce((a, b) => a + b, 0);
}

// Balance for a configured "account" that may be either a token-account address
// or an owner address — try token account first, fall back to owner.
async function getAddressBalance(rpc: string, address: string): Promise<number> {
  try {
    const r = await rpcCall<{
      value: { amount: string; decimals: number; uiAmount: number | null };
    }>(rpc, 'getTokenAccountBalance', [address]);
    return r.value.uiAmount ?? Number(r.value.amount) / 10 ** r.value.decimals;
  } catch {
    // Not a token account — treat it as an owner.
    return getOwnerBalance(rpc, address);
  }
}

// Holder enumeration via Helius DAS getTokenAccounts. Dedupes by owner (one
// owner can hold many token accounts) and ranks the top holders. Page-capped.
const HOLDER_PAGE_LIMIT = 1000;
const HOLDER_MAX_PAGES = 25; // up to 25k accounts before we stop and flag it
const TOP_HOLDERS = 12;

interface HeliusTokenAccount {
  owner: string;
  amount: number; // raw, needs / 10**decimals
}
interface HeliusTokenAccountsPage {
  total: number;
  limit: number;
  page: number;
  token_accounts: HeliusTokenAccount[];
}

async function getHolders(
  rpc: string,
): Promise<{ count: number; capped: boolean; top: Array<{ owner: string; amount: number }> }> {
  const balances = new Map<string, number>(); // owner -> raw amount
  let page = 1;
  let capped = false;
  let decimals = 6;

  // Grab decimals once so we can return UI amounts for the top holders.
  try {
    decimals = (await getTokenSupply(rpc)).decimals;
  } catch {
    /* fall back to 6 */
  }

  for (;;) {
    const res = await rpcCall<HeliusTokenAccountsPage>(rpc, 'getTokenAccounts', {
      mint: MINT,
      page,
      limit: HOLDER_PAGE_LIMIT,
      options: { showZeroBalance: false },
    });
    const accounts = res.token_accounts ?? [];
    for (const a of accounts) {
      if (!a.owner || !a.amount) continue;
      balances.set(a.owner, (balances.get(a.owner) ?? 0) + a.amount);
    }
    if (accounts.length < HOLDER_PAGE_LIMIT) break;
    page += 1;
    if (page > HOLDER_MAX_PAGES) {
      capped = true;
      break;
    }
  }

  const top = [...balances.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, TOP_HOLDERS)
    .map(([owner, raw]) => ({ owner, amount: raw / 10 ** decimals }));

  return { count: balances.size, capped, top };
}

// ── Price (Jupiter, no key required) ─────────────────────────────────────────
async function getJupiterPrice(mint: string): Promise<number | null> {
  const res = await fetch(`https://lite-api.jup.ag/price/v3?ids=${mint}`, {
    next: { revalidate: REVALIDATE_S },
  });
  if (!res.ok) throw new Error(`Jupiter HTTP ${res.status}`);
  const json = (await res.json()) as Record<string, { usdPrice?: number } | undefined>;
  const price = json[mint]?.usdPrice;
  return typeof price === 'number' ? price : null;
}

// ── Creator rewards (pump.fun swap-api) ──────────────────────────────────────
// Lifetime creator fees earned by the creator wallet, in SOL. The swap-api
// returns a daily series with a running `cumulativeCreatorFeeSOL`; the latest
// (max) bucket is the lifetime total. The per-coin endpoint is unreliable, so
// we use the per-creator one.
interface CreatorFeeBucket {
  cumulativeCreatorFeeSOL: string;
}
async function getCreatorRewardsSol(): Promise<number | null> {
  const res = await fetch(
    `https://swap-api.pump.fun/v1/creators/${TOKEN_CREATOR_ADDRESS}/fees?interval=1d`,
    { next: { revalidate: REVALIDATE_S }, headers: { 'User-Agent': 'voicebox.sh' } },
  );
  if (!res.ok) throw new Error(`pump.fun swap-api HTTP ${res.status}`);
  const buckets = (await res.json()) as CreatorFeeBucket[];
  if (!Array.isArray(buckets) || buckets.length === 0) return null;
  // Cumulative is monotonic, but take the max defensively.
  const max = buckets.reduce((m, b) => {
    const v = Number.parseFloat(b.cumulativeCreatorFeeSOL);
    return Number.isFinite(v) && v > m ? v : m;
  }, 0);
  return max;
}
