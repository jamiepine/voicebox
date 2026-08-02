import {Coins} from "lucide-react";
import type {Metadata} from "next";
import Link from "next/link";
import {Footer} from "@/components/Footer";
import {Navbar} from "@/components/Navbar";
import {PricingTiers} from "@/components/PricingTiers";

export const metadata: Metadata = {
	title: "Pricing — Voicebox",
	description:
		"Voicebox is free and open source forever. Optional, end-to-end encrypted cloud backup & sync — free for $VOICEBOX holders.",
	openGraph: {
		title: "Voicebox Pricing",
		description:
			"The app is free forever. Cloud backup & sync is an optional add-on — free for $VOICEBOX holders.",
		type: "website",
		url: "https://voicebox.sh/pricing",
		images: [{url: "/og.webp", width: 1200, height: 630}],
	},
};

const FAQ = [
	{
		q: "Is the app really free?",
		a: "Yes — Voicebox is free and open source, forever. Cloning, dictation, every TTS engine, MCP, personalities: all of it runs locally with no account. The paid plans only add optional cloud backup & sync.",
	},
	{
		q: "What's encrypted in the cloud?",
		a: "Everything. Your profiles, generations, and captures are end-to-end encrypted on your device before upload. The server stores only ciphertext and can never read your data.",
	},
	{
		q: "Do $VOICEBOX holders really get Cloud free?",
		a: "Yes. Holding the token unlocks the Cloud tier at no cost. The app itself is free regardless — the token is an optional way to support the project.",
	},
	{
		q: "What counts toward storage?",
		a: "Your encrypted objects — generated audio, the original audio kept with each capture, and profile data. Plans differ mainly on storage, device count, and version-history length.",
	},
	{
		q: "Can I cancel anytime?",
		a: "Yes. Cloud is a subscription you can cancel whenever you like; your local library always stays on your machine and keeps working.",
	},
];

export default function PricingPage() {
	return (
		<>
			<Navbar />

			{/* ── Hero ─────────────────────────────────────────────────── */}
			<section className="relative pt-32 pb-12">
				<div className="hero-glow hero-glow-fade pointer-events-none absolute inset-0 -top-32">
					<div className="absolute left-1/2 top-0 -translate-x-1/2 w-[900px] h-[460px] rounded-full bg-accent/12 blur-[140px]" />
				</div>
				<div className="relative mx-auto max-w-4xl px-6 text-center">
					<div className="fade-in mb-4 text-[11px] font-semibold uppercase tracking-[0.22em] text-accent">
						Pricing
					</div>
					<h1 className="fade-in text-5xl font-bold tracking-tighter text-foreground md:text-6xl">
						The app is free. Forever.
					</h1>
					<p className="fade-in mx-auto mt-6 max-w-2xl text-lg text-muted-foreground">
						Everything that makes Voicebox great runs locally at no cost. Pay
						only if you want optional, encrypted cloud backup & sync — and
						holders get that free.
					</p>
				</div>
			</section>

			{/* ── Tiers (with monthly/annual toggle) ───────────────────── */}
			<section className="pb-8">
				<PricingTiers />
			</section>

			{/* ── Holder callout ───────────────────────────────────────── */}
			<section className="py-12">
				<div className="mx-auto max-w-3xl px-6">
					<Link
						href="/token"
						className="group flex flex-col items-center gap-3 rounded-2xl border border-accent/30 bg-card/40 backdrop-blur-sm px-6 py-8 text-center transition-colors hover:border-accent/50"
					>
						<Coins className="h-6 w-6 text-accent" />
						<h2 className="text-xl md:text-2xl font-semibold tracking-tight text-foreground">
							Hold $VOICEBOX, get Cloud free.
						</h2>
						<p className="max-w-xl text-sm text-muted-foreground">
							The token is an optional way to back the project — and holders get
							the Cloud tier at no cost. Learn how it works and verify everything
							on-chain.
						</p>
						<span className="mt-1 text-sm font-medium text-accent group-hover:underline underline-offset-4">
							View the token →
						</span>
					</Link>
				</div>
			</section>

			{/* ── FAQ ──────────────────────────────────────────────────── */}
			<section className="border-t border-border py-20">
				<div className="mx-auto max-w-3xl px-6">
					<div className="text-center mb-12">
						<h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">
							Questions
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
						Cloud pricing and limits are not final — they'll be confirmed at
						launch.
					</p>
				</div>
			</section>

			<Footer />
		</>
	);
}
