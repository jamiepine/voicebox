import {ArrowRight, Cloud, KeyRound, Lock, ShieldCheck} from "lucide-react";
import type {Metadata} from "next";
import Link from "next/link";
import {Footer} from "@/components/Footer";
import {Navbar} from "@/components/Navbar";
import {CLOUD_FEATURES, CLOUD_NOTIFY_URL} from "@/lib/pricing";

export const metadata: Metadata = {
	title: "Cloud Backup & Sync — Voicebox",
	description:
		"End-to-end encrypted backup and sync for your Voicebox library. We can't read your data — only your devices can. Optional, local-first, free for $VOICEBOX holders.",
	openGraph: {
		title: "Voicebox Cloud — encrypted backup & sync",
		description:
			"End-to-end encrypted backup and sync across desktop and mobile. The server is blind — only your devices can decrypt.",
		type: "website",
		url: "https://voicebox.sh/cloud",
		images: [{url: "/og.webp", width: 1200, height: 630}],
	},
};

const STEPS = [
	{
		icon: Lock,
		title: "Encrypted on your device",
		body: "Profiles, generations, and captures are encrypted locally with keys only you hold — before anything is uploaded.",
	},
	{
		icon: Cloud,
		title: "Stored as opaque blobs",
		body: "The server keeps your encrypted objects and a sync feed. It can route and store them, but never decrypt them.",
	},
	{
		icon: KeyRound,
		title: "Only your devices decrypt",
		body: "Each device unwraps your master key on pairing. A recovery phrase you control lets you restore everything to a new one.",
	},
];

export default function CloudPage() {
	return (
		<>
			<Navbar />

			{/* ── Hero ─────────────────────────────────────────────────── */}
			<section className="relative pt-32 pb-16">
				<div className="hero-glow hero-glow-fade pointer-events-none absolute inset-0 -top-32">
					<div className="absolute left-1/2 top-0 -translate-x-1/2 w-[900px] h-[500px] rounded-full bg-accent/12 blur-[140px]" />
				</div>

				<div className="relative mx-auto max-w-4xl px-6 text-center">
					<div className="fade-in mb-6 inline-flex items-center gap-2 rounded-full border border-border/60 bg-card/40 px-3 py-1">
						<Cloud className="h-3.5 w-3.5 text-accent" />
						<span className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
							Voicebox Cloud · coming soon
						</span>
					</div>

					<h1 className="fade-in text-5xl font-bold tracking-tighter leading-[0.95] text-foreground md:text-6xl lg:text-7xl">
						Your studio, backed up and in sync.
					</h1>

					<p className="fade-in mx-auto mt-6 max-w-2xl text-lg text-muted-foreground md:text-xl">
						Optional, end-to-end encrypted backup and sync for your entire
						Voicebox library. We can't read a byte of it — only your devices
						can. Free for{" "}
						<Link href="/token" className="text-foreground underline-offset-4 hover:underline">
							$VOICEBOX
						</Link>{" "}
						holders.
					</p>

					<div className="fade-in mt-10 flex flex-row items-center justify-center gap-3 sm:gap-4">
						<Link
							href="/pricing"
							className="rounded-full bg-accent px-8 py-3.5 text-sm font-semibold uppercase tracking-wider text-white shadow-[0_4px_20px_hsl(43_60%_50%/0.3),inset_0_2px_0_rgba(255,255,255,0.2),inset_0_-2px_0_rgba(0,0,0,0.1)] transition-all hover:bg-accent-faint"
						>
							See pricing
						</Link>
						<a
							href={CLOUD_NOTIFY_URL}
							target="_blank"
							rel="noopener noreferrer"
							className="flex items-center gap-2 rounded-full border border-border/60 bg-card/40 backdrop-blur-sm px-6 py-3 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground hover:border-border"
						>
							Get notified
							<ArrowRight className="h-4 w-4" />
						</a>
					</div>
				</div>
			</section>

			{/* ── Features ─────────────────────────────────────────────── */}
			<section className="border-t border-border py-20">
				<div className="mx-auto max-w-5xl px-6">
					<div className="grid gap-4 md:grid-cols-3">
						{CLOUD_FEATURES.map((f) => (
							<div
								key={f.title}
								className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6"
							>
								<h3 className="text-[15px] font-semibold text-foreground mb-2">
									{f.title}
								</h3>
								<p className="text-sm leading-relaxed text-muted-foreground">
									{f.body}
								</p>
							</div>
						))}
					</div>
				</div>
			</section>

			{/* ── How it works ─────────────────────────────────────────── */}
			<section className="border-t border-border py-20">
				<div className="mx-auto max-w-4xl px-6">
					<div className="text-center mb-12">
						<div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-4">
							How it works
						</div>
						<h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">
							Zero-knowledge by design.
						</h2>
					</div>

					<div className="grid gap-4 md:grid-cols-3">
						{STEPS.map((step, i) => {
							const Icon = step.icon;
							return (
								<div
									key={step.title}
									className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6"
								>
									<div className="flex items-center gap-3 mb-3">
										<Icon className="h-5 w-5 text-accent" />
										<span className="font-mono text-xs text-muted-foreground/60">
											0{i + 1}
										</span>
									</div>
									<h3 className="text-[15px] font-semibold text-foreground mb-2">
										{step.title}
									</h3>
									<p className="text-sm leading-relaxed text-muted-foreground">
										{step.body}
									</p>
								</div>
							);
						})}
					</div>
				</div>
			</section>

			{/* ── Trust callout ────────────────────────────────────────── */}
			<section className="border-t border-border py-20">
				<div className="mx-auto max-w-3xl px-6">
					<div className="rounded-2xl border-2 border-accent/40 bg-card/60 backdrop-blur-sm p-8 md:p-10 text-center shadow-[0_8px_40px_hsl(43_60%_50%/0.08)]">
						<ShieldCheck className="h-7 w-7 text-accent mx-auto mb-4" />
						<h2 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground mb-3">
							We can't see your data. That's the point.
						</h2>
						<p className="text-muted-foreground leading-relaxed max-w-2xl mx-auto">
							Voicebox is local-first and privacy-first. The cloud keeps that
							promise: your library is encrypted before it leaves your device,
							the server stores only ciphertext, and the keys never leave your
							control. Same philosophy as the app — just backed up.
						</p>
						<div className="mt-8 flex flex-row items-center justify-center gap-3">
							<Link
								href="/pricing"
								className="rounded-full bg-accent px-6 py-3 text-sm font-semibold text-white shadow-[0_4px_20px_hsl(43_60%_50%/0.3)] transition-all hover:bg-accent-faint"
							>
								See pricing
							</Link>
							<Link
								href="/token"
								className="rounded-full border border-border/60 bg-card/40 px-6 py-3 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground hover:border-border"
							>
								Free for holders →
							</Link>
						</div>
					</div>
				</div>
			</section>

			<Footer />
		</>
	);
}
