// Download links for voicebox releases
// These are fallback values - link to releases page if API fails
export const LATEST_VERSION = 'v0.1.0';

export const GITHUB_REPO = 'https://github.com/jamiepine/voicebox';
export const GITHUB_RELEASES_PAGE = `${GITHUB_REPO}/releases`;
export const DONATE_URL = 'https://buymeacoffee.com/jamiepine';
export const SPONSOR_CHECKOUT_URL = 'https://buy.stripe.com/eVqdRad3n16ubcqf201Jm00';
export const SPONSOR_CONTACT_EMAIL = 'jamie@spacedrive.com';

// $VOICEBOX — the official community token on Solana
export const TOKEN_TICKER = '$VOICEBOX';
export const TOKEN_CONTRACT_ADDRESS = 'FpzZHtp5tbvz6xndEtoJHoGEWcT7cFEuscdCh9RApump';
export const TOKEN_PUMP_URL = `https://pump.fun/coin/${TOKEN_CONTRACT_ADDRESS}`;
// Solscan token page — lets anyone inspect supply, holders, and history.
export const TOKEN_SOLSCAN_URL = `https://solscan.io/token/${TOKEN_CONTRACT_ADDRESS}`;
export const TOKEN_TOTAL_SUPPLY = '1B';

// ── Live on-chain tracking config ───────────────────────────────────────────
// Powers the transparency dashboard on /token. Reads are done server-side via
// Helius (HELIUS_API_KEY). Every value below has a safe default so the page
// still renders if something is unset — sections you haven't configured just
// show as "not configured" rather than breaking the build.

/** Mint supply at launch, used to derive burned = initial − current supply. */
export const TOKEN_INITIAL_SUPPLY = 1_000_000_000;

/**
 * pump.fun creator wallet — the address that launched the coin and earns creator
 * fees. Lifetime creator rewards (in SOL) are read from pump.fun's swap-api for
 * this wallet. Defaults to the dev wallet (they're the same here).
 */
export const TOKEN_CREATOR_ADDRESS = envStr(
  'TOKEN_CREATOR_ADDRESS',
  'BSn573bjkQa5iffMg6zA8eb9mHyzewu2ps9R85qNKXC5',
);

/**
 * Dev / treasury wallets to surface as "team holdings". List every address you
 * want counted; balances are summed. Public, read-only — these are already
 * visible on-chain. Override at deploy time with TOKEN_DEV_WALLETS (comma list).
 */
export const TOKEN_DEV_WALLETS: string[] = envList('TOKEN_DEV_WALLETS', [
  'BSn573bjkQa5iffMg6zA8eb9mHyzewu2ps9R85qNKXC5', // Jamie's dev/treasury wallet
]);

/**
 * Locked supply: token accounts whose $VOICEBOX is locked (liquidity lockers,
 * vesting escrows). Each entry is summed into "locked"; unlocksAt is optional
 * copy for the card. Override with TOKEN_LOCKED_ACCOUNTS as a JSON array.
 */
export interface LockedAccount {
  label: string;
  /** The token account or owner address holding the locked $VOICEBOX. */
  account: string;
  /** Human-readable unlock date, e.g. "Unlocks Jun 2027" (optional). */
  unlocksAt?: string;
  /** Optional Solscan/locker link proving the lock. */
  url?: string;
}
export const TOKEN_LOCKED_ACCOUNTS: LockedAccount[] = envJson<LockedAccount[]>(
  'TOKEN_LOCKED_ACCOUNTS',
  [
    // Streamflow locks. `account` is each lock's escrow token account (read for
    // the live balance, so it ticks down only when actually unlocked/withdrawn);
    // `url` is the public Streamflow contract page for verification.
    {
      label: 'Streamflow lock #1',
      account: 'EaPun3ZUk5XiKft2tbvVRXgq8HyXjTmg77kUYYe7Q5HM',
      unlocksAt: 'Unlocks Jun 2027',
      url: 'https://app.streamflow.finance/contract/solana/mainnet/AmzHaDAZWWZPkvN5zC78mQ3QAedH7hHSCEWeYSbSWXu5',
    },
    {
      label: 'Streamflow lock #2',
      account: 'FGK5G4CbtryRdoubPN7u4y3WTYS4vqoepPLpppba92cp',
      unlocksAt: 'Unlocks Jun 2027',
      url: 'https://app.streamflow.finance/contract/solana/mainnet/GfBjWriW8mcJWS9njC2gBBJoJuGRQQzJLRFNg6a12bW8',
    },
    {
      label: 'Streamflow lock #3',
      account: 'ELKMRnDin7w6ht4MkQ6FnDU3LvkDtoYbtR9y3P51pf1N',
      unlocksAt: 'Unlocks Jun 2027',
      url: 'https://app.streamflow.finance/contract/solana/mainnet/3xa49K6b8ChsL5SoPYrAWigwKmoXAge6YCAmUJWM6Ncw',
    },
  ],
);

/**
 * Burn / dead address. The standard SPL incinerator by default. Buyback+burns
 * that reduce mint supply are already captured by initial − current; this is
 * only used to additionally surface anything parked at a dead address.
 */
export const TOKEN_BURN_ADDRESS = envStr(
  'TOKEN_BURN_ADDRESS',
  '1nc1nerator11111111111111111111111111111111',
);

/** How long stats are cached server-side (ms). Keeps us off rate limits. */
export const TOKEN_STATS_CACHE_MS = 1000 * 60 * 10; // 10 minutes

// ── tiny env helpers (server-only; safe in this module, no secrets exposed) ──
function envStr(key: string, fallback: string): string {
  const v = process.env[key];
  return v && v.trim() ? v.trim() : fallback;
}
function envList(key: string, fallback: string[]): string[] {
  const v = process.env[key];
  if (!v || !v.trim()) return fallback;
  return v
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}
function envJson<T>(key: string, fallback: T): T {
  const v = process.env[key];
  if (!v || !v.trim()) return fallback;
  try {
    return JSON.parse(v) as T;
  } catch {
    return fallback;
  }
}

// On-chain transparency log — locks and buyback+burns.
// Add a new entry every time a lock or burn happens; set `txUrl` to its Solscan
// link to make the card a live, verifiable proof. Entries without a txUrl render
// as "proof link pending" — fill them in as soon as the hash is available.
export interface TokenProof {
  kind: 'lock' | 'burn';
  label: string;
  detail: string;
  /** Solscan (or locker) URL proving the action. Empty = pending, shown as such. */
  txUrl: string;
}

export const TOKEN_PROOFS: TokenProof[] = [
  {
    kind: 'lock',
    label: 'Launch liquidity lock',
    detail:
      'Liquidity and a portion of dev holdings were locked at launch (~6.6% top holder), so the supply can be verified on-chain from day one.',
    txUrl: '', // TODO(jamie): add the Solscan/locker link for the launch lock
  },
  {
    kind: 'burn',
    label: 'Buyback & burn',
    detail:
      'Bought $VOICEBOX back from the open market and burned it to a dead address, permanently removing it from supply.',
    txUrl:
      'https://solscan.io/tx/5MjK4CYMBKAewLcjdD6QkM8ctkeG2bjyQhpjNgEumkDbDtoKCVmzKcWwLWsd4QJov8hs5zbGLt3g5vVCp4CBmze5',
  },
  {
    kind: 'burn',
    label: 'Buyback & burn',
    detail:
      'A second buyback and burn — part of an ongoing commitment to keep buying back and reducing supply over time.',
    txUrl: '', // TODO(jamie): add the Solscan link for the second burn
  },
];

export const DOWNLOAD_LINKS = {
  macArm: GITHUB_RELEASES_PAGE,
  macIntel: GITHUB_RELEASES_PAGE,
  windows: GITHUB_RELEASES_PAGE,
  linux: GITHUB_RELEASES_PAGE,
} as const;

// Export function to get dynamic download links
export { getLatestRelease } from './releases';
