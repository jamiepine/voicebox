import {
	ArrowUpRight,
	Check,
	Cloud,
	Flame,
	Heart,
	Lock,
	Rocket,
	ShieldCheck,
} from "lucide-react";
import type {Metadata} from "next";
import {Footer} from "@/components/Footer";
import {Navbar} from "@/components/Navbar";
import {TokenSection} from "@/components/TokenSection";
import {TokenStatsSection} from "@/components/TokenStats";
import {
	TOKEN_PROOFS,
	TOKEN_SOLSCAN_URL,
	TOKEN_TICKER,
} from "@/lib/constants";

export const metadata: Metadata = {
	title: `${TOKEN_TICKER} — The official Voicebox token`,
	description: `${TOKEN_TICKER} is the official community token for Voicebox on Solana. Entirely optional — Voicebox is, and always will be, free and open source.`,
	openGraph: {
		title: `${TOKEN_TICKER} on Solana`,
		description: `The official community token for Voicebox. Optional, just for fun — Voicebox stays free and open source.`,
		type: "website",
		url: "https://voicebox.sh/token",
		images: [{url: "/og.webp", width: 1200, height: 630}],
	},
};

// Re-fetch live on-chain stats at most every 10 minutes (matches the server
// cache in token-stats.ts). Keeps the page static-fast while staying fresh.
export const revalidate = 600;

const USE_OF_FUNDS = [
	{
		icon: Rocket,
		title: "Full-time development",
		body: "The token is the equivalent of a salary — it lets me work on Voicebox every day instead of squeezing it around other work.",
	},
	{
		icon: Cloud,
		title: "Mobile + cloud backup & sync",
		body: "Shipping the mobile app and encrypted cloud backup/sync so your generations and captures are safe and available anywhere.",
	},
	{
		icon: Heart,
		title: "More engines, more hardware",
		body: "Adding TTS engines and broadening GPU / OS support so Voicebox runs great on whatever you've got.",
	},
];

const FAQ = [
	{
		q: "Do I need the token to use Voicebox?",
		a: "No. Voicebox is free and open source, and every feature works without ever touching the token. It exists purely for supporters who want to back the project and have some fun.",
	},
	{
		q: "Is this an investment?",
		a: "No. $VOICEBOX is a community token, not a security or a promise of returns. There is no roadmap of financial milestones, and nothing here is financial advice. Only spend what you're comfortable with.",
	},
	{
		q: "How do I buy it?",
		a: "Copy the contract address above, then buy on pump.fun with a Solana wallet. Always verify the address matches the one on this page — impersonators are common.",
	},
	{
		q: "Does buying it fund development?",
		a: "Yes — going full-time on Voicebox is funded by the token, alongside donations. The surest way to support the project either way is to use it, star the repo, and tell people about it.",
	},
];

export default function TokenPage() {
	return (
		<>
			<Navbar />

			{/* Top padding clears the fixed navbar; TokenSection carries the
			    header, contract address, and buy CTA. */}
			<main className="pt-16">
				<TokenSection />

				{/* ── Live on-chain stats ──────────────────────────────────── */}
				<TokenStatsSection />

				{/* ── Why a token ──────────────────────────────────────────── */}
				<section className="border-t border-border py-20">
					<div className="mx-auto max-w-3xl px-6">
						<div className="text-center mb-12">
							<div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-4">
								Why a token
							</div>
							<h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">
								So I can build this full-time.
							</h2>
							<p className="text-muted-foreground max-w-2xl mx-auto mt-4">
								Voicebox grew to over a million downloads with zero marketing —
								but donations alone never made full-time work sustainable.{" "}
								{TOKEN_TICKER} changed that overnight, and it's already
								accelerating everything below. The app stays{" "}
								<b className="text-foreground">free, open source, and local-first</b>{" "}
								— forever.
							</p>
						</div>

						<div className="grid gap-4 sm:grid-cols-3">
							{USE_OF_FUNDS.map((item) => {
								const Icon = item.icon;
								return (
									<div
										key={item.title}
										className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6"
									>
										<Icon className="h-5 w-5 text-accent mb-3" />
										<h3 className="text-[15px] font-semibold text-foreground mb-2">
											{item.title}
										</h3>
										<p className="text-sm leading-relaxed text-muted-foreground">
											{item.body}
										</p>
									</div>
								);
							})}
						</div>
					</div>
				</section>

				{/* ── Holder utility ───────────────────────────────────────── */}
				<section className="border-t border-border py-20">
					<div className="mx-auto max-w-3xl px-6">
						<div className="rounded-2xl border-2 border-accent/40 bg-card/60 backdrop-blur-sm p-8 md:p-10 shadow-[0_8px_40px_hsl(43_60%_50%/0.08)]">
							<div className="flex items-center gap-2 mb-4">
								<Cloud className="h-5 w-5 text-accent" />
								<span className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent">
									Holder perk · coming soon
								</span>
							</div>
							<h2 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground mb-3">
								Cloud backup & sync — free for holders.
							</h2>
							<p className="text-muted-foreground leading-relaxed">
								Encrypted cloud backup and sync (and the mobile cloud) will be a
								paid service — roughly{" "}
								<b className="text-foreground">$12/year</b> for everyone else, and{" "}
								<b className="text-foreground">free for {TOKEN_TICKER} holders</b>.
								Generate on the go, keep your captures and generations safe, and
								pick up on any device.
							</p>
						</div>
					</div>
				</section>

				{/* ── Official vs community ────────────────────────────────── */}
				<section className="border-t border-border py-20">
					<div className="mx-auto max-w-3xl px-6">
						<div className="rounded-2xl border border-border bg-card/40 backdrop-blur-sm p-8">
							<div className="flex items-center gap-2 mb-4">
								<ShieldCheck className="h-5 w-5 text-accent" />
								<h2 className="text-xl md:text-2xl font-semibold tracking-tight text-foreground">
									One official token. Accept no substitutes.
								</h2>
							</div>
							<ul className="space-y-3">
								<ProofRow text={`${TOKEN_TICKER} is the only official Voicebox token. The mint address on this page is the single source of truth — always verify it.`} />
								<ProofRow text="My other projects (including Spacedrive) will never have an official token. This is the only one I'll ever make." />
								<ProofRow text="I deployed it myself so liquidity can be locked and the trajectory controlled — and I no longer claim fees on any other community tokens." />
							</ul>
						</div>
					</div>
				</section>

				{/* ── Good to know ─────────────────────────────────────────── */}
				<section className="border-t border-border py-20">
					<div className="mx-auto max-w-3xl px-6">
						<div className="text-center mb-12">
							<div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-4">
								Good to know
							</div>
							<h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">
								Optional, just for fun.
							</h2>
						</div>

						<div className="grid gap-4 sm:grid-cols-2">
							{FAQ.map((item) => (
								<div
									key={item.q}
									className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6"
								>
									<h3 className="text-[15px] font-semibold text-foreground mb-2">
										{item.q}
									</h3>
									<p className="text-sm leading-relaxed text-muted-foreground">
										{item.a}
									</p>
								</div>
							))}
						</div>

						<p className="text-center text-xs text-muted-foreground/70 mt-10 max-w-2xl mx-auto">
							{TOKEN_TICKER} is a community token with no affiliation to any
							exchange or financial product. Nothing on this page is financial
							advice. Verify the contract address before buying.
						</p>
					</div>
				</section>
			</main>

			<Footer />
		</>
	);
}

function ProofRow({text}: {text: string}) {
	return (
		<li className="flex items-start gap-3 text-sm text-foreground/90">
			<Check className="h-5 w-5 shrink-0 text-accent mt-px" />
			<span>{text}</span>
		</li>
	);
}
