import {ArrowUpRight, Coins, Flame, Lock, Users, Wallet} from "lucide-react";
import {
	TOKEN_CONTRACT_ADDRESS,
	TOKEN_CREATOR_ADDRESS,
	TOKEN_SOLSCAN_URL,
	TOKEN_TICKER,
} from "@/lib/constants";
import {getTokenStats, type TokenStats} from "@/lib/token-stats";

// ── formatters ───────────────────────────────────────────────────────────────
function compact(n: number | null): string {
	if (n == null) return "—";
	const abs = Math.abs(n);
	if (abs >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
	if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
	if (abs >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
	return n.toLocaleString("en-US", {maximumFractionDigits: 0});
}

function pct(n: number | null): string {
	if (n == null) return "—";
	if (n > 0 && n < 0.01) return "<0.01%";
	return `${n.toFixed(2)}%`;
}

function usdPrice(n: number | null): string {
	if (n == null) return "—";
	if (n < 0.000001) return `$${n.toExponential(2)}`;
	if (n < 1) return `$${n.toPrecision(3)}`;
	return `$${n.toLocaleString("en-US", {maximumFractionDigits: 2})}`;
}

function usdBig(n: number | null): string {
	if (n == null) return "—";
	if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
	if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
	return `$${n.toLocaleString("en-US", {maximumFractionDigits: 0})}`;
}

function sol(n: number | null): string {
	if (n == null) return "—";
	return `${n.toLocaleString("en-US", {maximumFractionDigits: 2})} SOL`;
}

function shortAddr(a: string): string {
	return a.length > 12 ? `${a.slice(0, 4)}…${a.slice(-4)}` : a;
}

function solscanAccount(a: string): string {
	return `https://solscan.io/account/${a}`;
}

function timeAgo(ts: number): string {
	const secs = Math.max(0, Math.round((Date.now() - ts) / 1000));
	if (secs < 60) return "just now";
	const mins = Math.round(secs / 60);
	if (mins < 60) return `${mins}m ago`;
	const hrs = Math.round(mins / 60);
	return `${hrs}h ago`;
}

export async function TokenStatsSection() {
	const stats = await getTokenStats();
	return <TokenStatsView stats={stats} />;
}

// Hidden for now — flip to true to bring the "fees earned for development"
// card back. The data is still fetched; it's just not rendered.
const SHOW_CREATOR_REWARDS = false;

function TokenStatsView({stats}: {stats: TokenStats}) {
	const cards = [
		{
			icon: Flame,
			label: "Burned",
			value: compact(stats.burned),
			sub: stats.burnedPct != null ? `${pct(stats.burnedPct)} of initial supply` : "Removed from supply forever",
		},
		{
			icon: Lock,
			label: "Locked",
			value: stats.locked != null ? compact(stats.locked) : "Not configured",
			sub: stats.lockedPct != null ? `${pct(stats.lockedPct)} of supply` : "Liquidity & vesting locks",
		},
		{
			icon: Wallet,
			label: "Dev / treasury",
			value: stats.devBalance != null ? compact(stats.devBalance) : "Not configured",
			sub: stats.devPct != null ? `${pct(stats.devPct)} of supply` : "Team-held tokens",
		},
		{
			icon: Users,
			label: "Holders",
			value: stats.holders != null ? `${stats.holdersCapped ? "" : ""}${stats.holders.toLocaleString("en-US")}` : "—",
			sub: stats.holdersCapped ? "counted (capped)" : "unique wallets",
		},
	];

	return (
		<section className="border-t border-border py-20">
			<div className="mx-auto max-w-5xl px-6">
				{/* Header */}
				<div className="text-center mb-12">
					<div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-4">
						Live on-chain stats
					</div>
					<h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">
						Every number, straight from the chain.
					</h2>
					<p className="text-muted-foreground max-w-2xl mx-auto mt-4">
						Supply, holders, burns, locks and team holdings for {TOKEN_TICKER},
						read live from Solana.
					</p>
				</div>

				{/* Supply + market headline */}
				<div className="grid gap-4 sm:grid-cols-3 mb-4">
					<HeadlineStat
						label="Circulating supply"
						value={compact(stats.circulating)}
						sub={
							stats.totalSupply != null
								? `of ${compact(stats.totalSupply)} total`
								: undefined
						}
					/>
					<HeadlineStat label="Price" value={usdPrice(stats.priceUsd)} sub="via Jupiter" />
					<HeadlineStat label="Market cap" value={usdBig(stats.marketCapUsd)} sub="price × supply" />
				</div>

				{/* Stat cards */}
				<div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
					{cards.map((c) => {
						const Icon = c.icon;
						return (
							<div
								key={c.label}
								className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6"
							>
								<Icon className="h-5 w-5 text-accent mb-3" />
								<div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-1">
									{c.label}
								</div>
								<div className="text-2xl font-semibold tracking-tight text-foreground tabular-nums">
									{c.value}
								</div>
								<div className="text-xs text-muted-foreground mt-1">{c.sub}</div>
							</div>
						);
					})}
				</div>

				{/* Creator rewards — pump.fun creator fees, the funding story */}
				{SHOW_CREATOR_REWARDS && stats.creatorRewardsSol != null && (
					<div className="mt-4 rounded-2xl border border-accent/30 bg-gradient-to-b from-accent/[0.08] to-transparent p-8 text-center">
						<div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full border border-accent/30 bg-accent/10">
							<Coins className="h-5 w-5 text-accent" />
						</div>
						<div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
							Fees earned for development
						</div>
						<div className="mt-2 text-4xl font-semibold tracking-tight text-foreground tabular-nums">
							{sol(stats.creatorRewardsSol)}
						</div>
						{stats.creatorRewardsUsd != null && (
							<div className="mt-1 text-sm text-muted-foreground tabular-nums">
								≈ {usdBig(stats.creatorRewardsUsd)}
							</div>
						)}
						<p className="mx-auto mt-4 max-w-md text-sm leading-relaxed text-muted-foreground">
							Lifetime {TOKEN_TICKER} trading fees — the funding that pays for
							full-time work on Voicebox.{" "}
							<a
								href={`https://solscan.io/account/${TOKEN_CREATOR_ADDRESS}`}
								target="_blank"
								rel="noopener noreferrer"
								className="inline-flex items-center gap-0.5 text-foreground/80 hover:text-foreground"
							>
								Verify <ArrowUpRight className="h-3 w-3" />
							</a>
						</p>
					</div>
				)}

				{/* Locked breakdown (only if any configured) */}
				{stats.lockedBreakdown.length > 0 && (
					<div className="mt-4 rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6">
						<div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-4">
							Locked &amp; vesting
						</div>
						<ul className="space-y-3">
							{stats.lockedBreakdown.map((l) => (
								<li
									key={l.account}
									className="flex items-center gap-3 text-sm"
								>
									<Lock className="h-4 w-4 shrink-0 text-accent" />
									<span className="text-foreground/90">{l.label}</span>
									{l.unlocksAt && (
										<span className="text-xs text-muted-foreground">· {l.unlocksAt}</span>
									)}
									<span className="ml-auto font-medium tabular-nums text-foreground">
										{compact(l.amount)}
									</span>
									<a
										href={l.url ?? solscanAccount(l.account)}
										target="_blank"
										rel="noopener noreferrer"
										className="text-muted-foreground hover:text-foreground"
										aria-label="View on Solscan"
									>
										<ArrowUpRight className="h-4 w-4" />
									</a>
								</li>
							))}
						</ul>
					</div>
				)}

				{/* Top holders */}
				{stats.topHolders.length > 0 && (
					<div className="mt-4 rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6">
						<div className="flex items-center justify-between mb-4">
							<div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
								Top holders
							</div>
							<a
								href={`${TOKEN_SOLSCAN_URL}#holders`}
								target="_blank"
								rel="noopener noreferrer"
								className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
							>
								All holders <ArrowUpRight className="h-3 w-3" />
							</a>
						</div>
						<ul className="divide-y divide-border/60">
							{stats.topHolders.map((h, i) => (
								<li
									key={h.owner}
									className="flex items-center gap-3 py-2.5 text-sm"
								>
									<span className="w-5 text-xs text-muted-foreground tabular-nums">
										{i + 1}
									</span>
									<a
										href={solscanAccount(h.owner)}
										target="_blank"
										rel="noopener noreferrer"
										className="font-mono text-foreground/90 hover:text-foreground hover:underline"
									>
										{shortAddr(h.owner)}
									</a>
									<span className="ml-auto tabular-nums text-foreground">
										{compact(h.amount)}
									</span>
									<span className="w-16 text-right tabular-nums text-muted-foreground">
										{pct(h.pct)}
									</span>
								</li>
							))}
						</ul>
					</div>
				)}

				{/* Footer: provenance + freshness */}
				<div className="mt-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-muted-foreground">
					<span>
						{stats.live ? (
							<>Updated {timeAgo(stats.updatedAt)} · data via Helius &amp; Jupiter</>
						) : (
							<>Live stats unavailable right now — verify on Solscan.</>
						)}
					</span>
					<a
						href={TOKEN_SOLSCAN_URL}
						target="_blank"
						rel="noopener noreferrer"
						className="inline-flex items-center gap-1.5 hover:text-foreground transition-colors"
					>
						<span className="font-mono">{shortAddr(TOKEN_CONTRACT_ADDRESS)}</span>
						Inspect on Solscan <ArrowUpRight className="h-3.5 w-3.5" />
					</a>
				</div>
			</div>
		</section>
	);
}

function HeadlineStat({
	label,
	value,
	sub,
}: {
	label: string;
	value: string;
	sub?: string;
}) {
	return (
		<div className="rounded-2xl border border-border bg-card/60 backdrop-blur-sm p-6 text-center">
			<div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground mb-2">
				{label}
			</div>
			<div className="text-3xl font-semibold tracking-tight text-foreground tabular-nums">
				{value}
			</div>
			{sub && <div className="text-xs text-muted-foreground mt-1">{sub}</div>}
		</div>
	);
}
