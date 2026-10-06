"use client";

import {Github} from "lucide-react";
import {useEffect, useState} from "react";
import {AgentIntegration} from "@/components/AgentIntegration";
import {ApiSection} from "@/components/ApiSection";
import {CaptureSection} from "@/components/CaptureSection";
import {ControlUI} from "@/components/ControlUI";
import {Features} from "@/components/Features";
import {Footer} from "@/components/Footer";
import {Navbar} from "@/components/Navbar";
import {Personalities} from "@/components/Personalities";
import {AppleIcon, LinuxIcon, WindowsIcon} from "@/components/PlatformIcons";
import {SupportedModels} from "@/components/SupportedModels";
import {Testimonials} from "@/components/Testimonials";
import {TokenTeaser} from "@/components/TokenTeaser";
import {TutorialsSection} from "@/components/TutorialsSection";
import {VoiceCreator} from "@/components/VoiceCreator";
import {GITHUB_REPO} from "@/lib/constants";

export default function Home() {
	const [version, setVersion] = useState<string | null>(null);
	const [totalDownloads, setTotalDownloads] = useState<number | null>(null);

	useEffect(() => {
		fetch("/api/releases")
			.then((res) => {
				if (!res.ok) throw new Error("Failed to fetch releases");
				return res.json();
			})
			.then((data) => {
				if (data.version) setVersion(data.version);
				if (data.totalDownloads != null) setTotalDownloads(data.totalDownloads);
			})
			.catch((error) => {
				console.error("Failed to fetch release info:", error);
			});
	}, []);

	return (
		<>
			<Navbar />

			{/* ── Hero Section ─────────────────────────────────────────────── */}
			<section className="relative pt-32 pb-16">
				{/* Background glow */}
				<div className="hero-glow hero-glow-fade pointer-events-none absolute inset-0 -top-32">
					<div className="absolute left-1/2 top-0 -translate-x-1/2 w-[800px] h-[600px] rounded-full bg-accent/15 blur-[150px]" />
					<div className="absolute left-1/2 top-12 -translate-x-1/2 w-[500px] h-[400px] rounded-full bg-accent/10 blur-[80px]" />
				</div>

				<div className="relative mx-auto max-w-7xl px-6 text-center">
					{/* Logo */}
					<div
						className="fade-in mx-auto mb-8 h-[120px] w-[120px] md:h-[160px] md:w-[160px]"
						style={{animationDelay: "0ms"}}
					>
						{/* eslint-disable-next-line @next/next/no-img-element */}
						<img
							src="/voicebox-logo-app.webp"
							alt="Voicebox"
							className="h-full w-full object-contain"
						/>
					</div>

					{/* Kicker */}
					<div
						className="fade-in mb-6 text-[11px] font-semibold uppercase tracking-[0.22em] text-accent"
						style={{animationDelay: "50ms"}}
					>
						The open-source AI voice studio
					</div>

					{/* Headline */}
					<div className="fade-in relative" style={{animationDelay: "100ms"}}>
						<h1 className="text-5xl font-bold tracking-tighter leading-[0.9] text-foreground md:text-7xl lg:text-8xl">
						Clone, dictate and create.
						</h1>
					</div>

					{/* Subtitle */}
					<p
						className="fade-in mx-auto mt-6 max-w-2xl text-lg text-muted-foreground md:text-xl"
						style={{animationDelay: "200ms"}}
					>
						Clone voices, generate speech across seven TTS engines, dictate into
						any app, and talk to agents in voices you own. A free and local alternative
						to ElevenLabs and WisprFlow, running{" "}
						<b className="text-white">entirely on your machine.</b>
					</p>

					{/* CTAs */}
					<div
						className="fade-in mt-10 flex flex-row items-center justify-center gap-3 sm:gap-4"
						style={{animationDelay: "300ms"}}
					>
						<a
							href="/download"
							className="rounded-full bg-accent px-8 py-3.5 text-sm font-semibold uppercase tracking-wider text-white shadow-[0_4px_20px_hsl(43_60%_50%/0.3),inset_0_2px_0_rgba(255,255,255,0.2),inset_0_-2px_0_rgba(0,0,0,0.1)] transition-all hover:bg-accent-faint active:shadow-[0_2px_10px_hsl(43_60%_50%/0.3),inset_0_4px_8px_rgba(0,0,0,0.3)]"
						>
							Download
						</a>
						<a
							href={GITHUB_REPO}
							target="_blank"
							rel="noopener noreferrer"
							className="flex items-center gap-2 rounded-full border border-border/60 bg-card/40 backdrop-blur-sm px-6 py-3 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground hover:border-border"
						>
							<Github className="h-4 w-4" />
							View on GitHub
						</a>
					</div>

					{/* Version + downloads */}
					<p
						className="fade-in mt-4 text-xs text-muted-foreground/50"
						style={{animationDelay: "400ms"}}
					>
						{version ?? ""}
						{version && totalDownloads != null ? " \u00b7 " : ""}
						{totalDownloads != null
							? `${totalDownloads.toLocaleString()} downloads`
							: ""}
						{version || totalDownloads != null ? " \u00b7 " : ""}
						macOS, Windows, Linux
					</p>
				</div>

				{/* ── ControlUI mockup ─────────────────────────────────────── */}
				<div className="mt-16">
					<ControlUI />
				</div>
			</section>

			{/* ── Features ─────────────────────────────────────────────── */}
			<Features />

			{/* ── Voice Creator ────────────────────────────────────────── */}
			<VoiceCreator />

			{/* ── Capture (dictation + STT + play as voice) ───────────── */}
			<CaptureSection />

			{/* ── Agent integration (speak primitive + MCP) ───────────── */}
			<AgentIntegration />

			{/* ── Personalities (Compose / Rewrite / Respond) ──────────── */}
			<Personalities />

			{/* ── API Section ──────────────────────────────────────────── */}
			<ApiSection />

			{/* ── Tutorials ────────────────────────────────────────────── */}
			<TutorialsSection />

			{/* ── Supported models ─────────────────────────────────────── */}
			<SupportedModels />

			{/* ── Testimonials ─────────────────────────────────────────── */}
			<Testimonials />

			{/* ── Download Section ─────────────────────────────────────── */}
			<section id="download" className="border-t border-border py-24">
				<div className="mx-auto max-w-4xl px-6">
					<div className="text-center mb-12">
						<h2 className="text-3xl font-semibold tracking-tight text-foreground md:text-4xl mb-4">
							Download Voicebox
						</h2>
						<p className="text-muted-foreground">
							Available for macOS, Windows, and Linux. No dependencies required.
						</p>
					</div>

					<div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-2xl mx-auto">
						{/* macOS ARM */}
						<a
							href="/download?platform=macArm"
							className="flex items-center rounded-xl border border-border bg-card/60 backdrop-blur-sm px-5 py-4 transition-all hover:border-accent/30 hover:bg-card group"
						>
							<AppleIcon className="h-6 w-6 shrink-0 text-muted-foreground group-hover:text-foreground transition-colors" />
							<div className="ml-4">
								<div className="text-sm font-medium">macOS</div>
								<div className="text-xs text-muted-foreground">
									Apple Silicon (ARM)
								</div>
							</div>
						</a>

						{/* macOS Intel */}
						<a
							href="/download?platform=macIntel"
							className="flex items-center rounded-xl border border-border bg-card/60 backdrop-blur-sm px-5 py-4 transition-all hover:border-accent/30 hover:bg-card group"
						>
							<AppleIcon className="h-6 w-6 shrink-0 text-muted-foreground group-hover:text-foreground transition-colors" />
							<div className="ml-4">
								<div className="text-sm font-medium">macOS</div>
								<div className="text-xs text-muted-foreground">Intel (x64)</div>
							</div>
						</a>

						{/* Windows */}
						<a
							href="/download?platform=windows"
							className="flex items-center rounded-xl border border-border bg-card/60 backdrop-blur-sm px-5 py-4 transition-all hover:border-accent/30 hover:bg-card group"
						>
							<WindowsIcon className="h-6 w-6 shrink-0 text-muted-foreground group-hover:text-foreground transition-colors" />
							<div className="ml-4">
								<div className="text-sm font-medium">Windows</div>
								<div className="text-xs text-muted-foreground">
									64-bit (MSI)
								</div>
							</div>
						</a>

						{/* Linux */}
						<a
							href="/linux-install"
							className="flex items-center rounded-xl border border-border bg-card/60 backdrop-blur-sm px-5 py-4 transition-all hover:border-accent/30 hover:bg-card group"
						>
							<LinuxIcon className="h-6 w-6 shrink-0 text-muted-foreground group-hover:text-foreground transition-colors" />
							<div className="ml-4">
								<div className="text-sm font-medium">Linux</div>
								<div className="text-xs text-muted-foreground">
									Build from source
								</div>
							</div>
						</a>
					</div>

					{/* GitHub link */}
					<div className="mt-6 text-center">
						<a
							href={`${GITHUB_REPO}/releases`}
							target="_blank"
							rel="noopener noreferrer"
							className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
						>
							<Github className="h-4 w-4" />
							View all releases on GitHub
						</a>
					</div>
				</div>
			</section>

			{/* ── $VOICEBOX token (teaser → /token) ─────────────────────── */}
			<TokenTeaser />

			{/* ── Footer ───────────────────────────────────────────────── */}
			<Footer />
		</>
	);
}
