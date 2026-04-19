"use client";

import {
	Github,
	Globe,
	Languages,
	MessageSquare,
	SlidersHorizontal,
	Zap,
} from "lucide-react";
import {useEffect, useState} from "react";
import {ApiSection} from "@/components/ApiSection";
import {ControlUI} from "@/components/ControlUI";
import {Features} from "@/components/Features";
import {Footer} from "@/components/Footer";
import {Navbar} from "@/components/Navbar";
import {AppleIcon, LinuxIcon, WindowsIcon} from "@/components/PlatformIcons";
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

					{/* Headline */}
					<div className="fade-in relative" style={{animationDelay: "100ms"}}>
						<h1 className="text-5xl font-bold tracking-tighter leading-[0.9] text-foreground md:text-7xl lg:text-8xl">
							Clone any voice, in seconds.
						</h1>
					</div>

					{/* Subtitle */}
					<p
						className="fade-in mx-auto mt-6 max-w-2xl text-lg text-muted-foreground md:text-xl"
						style={{animationDelay: "200ms"}}
					>
						Open source voice cloning studio with support for multiple TTS
						engines. Clone any voice, generate natural speech, and compose
						multi-voice projects. All running{" "}
						<b className="text-white">locally on your machine.</b>
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

			{/* ── Tutorials ────────────────────────────────────────────── */}
			<TutorialsSection />

			{/* ── API Section ──────────────────────────────────────────── */}
			<ApiSection />

			{/* ── Models ─────────────────────────────────────────────────── */}
			<section id="about" className="border-t border-border py-24">
				<div className="mx-auto max-w-5xl px-6">
					<div className="text-center mb-14">
						<h2 className="text-3xl font-semibold tracking-tight text-foreground md:text-4xl mb-4">
							Multi-Engine Architecture
						</h2>
						<p className="text-muted-foreground max-w-2xl mx-auto">
							Choose the right model for every job. All models run locally on
							your hardware — download once, use forever.
						</p>
					</div>

					<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
						{/* Qwen3-TTS */}
						<div className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-6 transition-colors hover:border-accent/30">
							<div className="flex items-start justify-between mb-3">
								<div>
									<h3 className="text-base font-semibold text-foreground">
										Qwen3-TTS
									</h3>
									<span className="text-xs text-muted-foreground/60">
										by Alibaba
									</span>
								</div>
								<div className="flex gap-1.5">
									<span className="text-[10px] px-2 py-0.5 rounded-full border border-border bg-background text-muted-foreground">
										1.7B
									</span>
									<span className="text-[10px] px-2 py-0.5 rounded-full border border-border bg-background text-muted-foreground">
										0.6B
									</span>
								</div>
							</div>
							<p className="text-sm text-muted-foreground leading-relaxed mb-4">
								High-quality multilingual voice cloning with natural prosody.
								The only engine with delivery instructions — control tone, pace,
								and emotion with natural language.
							</p>
							<div className="flex flex-wrap gap-2">
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									<Globe className="h-3 w-3" />
									10 languages
								</span>
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									<MessageSquare className="h-3 w-3" />
									Delivery instructions
								</span>
							</div>
						</div>

						{/* Chatterbox */}
						<div className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-6 transition-colors hover:border-accent/30">
							<div className="flex items-start justify-between mb-3">
								<div>
									<h3 className="text-base font-semibold text-foreground">
										Chatterbox
									</h3>
									<span className="text-xs text-muted-foreground/60">
										by Resemble AI
									</span>
								</div>
							</div>
							<p className="text-sm text-muted-foreground leading-relaxed mb-4">
								Production-grade voice cloning with the broadest language
								support. 23 languages with zero-shot cloning and emotion
								exaggeration control.
							</p>
							<div className="flex flex-wrap gap-2">
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									<Languages className="h-3 w-3" />
									23 languages
								</span>
							</div>
						</div>

						{/* Chatterbox Turbo */}
						<div className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-6 transition-colors hover:border-accent/30">
							<div className="flex items-start justify-between mb-3">
								<div>
									<h3 className="text-base font-semibold text-foreground">
										Chatterbox Turbo
									</h3>
									<span className="text-xs text-muted-foreground/60">
										by Resemble AI
									</span>
								</div>
								<span className="text-[10px] px-2 py-0.5 rounded-full border border-border bg-background text-muted-foreground">
									350M
								</span>
							</div>
							<p className="text-sm text-muted-foreground leading-relaxed mb-4">
								Lightweight and fast. Supports paralinguistic tags — embed
								[laugh], [sigh], [gasp] and more directly in your text for
								expressive, natural speech.
							</p>
							<div className="flex flex-wrap gap-2">
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									<Zap className="h-3 w-3" />
									350M params
								</span>
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									<MessageSquare className="h-3 w-3" />
									[laugh] [sigh] tags
								</span>
							</div>
						</div>

						{/* LuxTTS */}
						<div className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-6 transition-colors hover:border-accent/30">
							<div className="flex items-start justify-between mb-3">
								<div>
									<h3 className="text-base font-semibold text-foreground">
										LuxTTS
									</h3>
									<span className="text-xs text-muted-foreground/60">
										by ZipVoice
									</span>
								</div>
							</div>
							<p className="text-sm text-muted-foreground leading-relaxed mb-4">
								Ultra-fast, CPU-friendly voice cloning at 48kHz. Exceeds 150x
								realtime on CPU with ~1GB VRAM. The fastest engine for quick
								iterations.
							</p>
							<div className="flex flex-wrap gap-2">
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									<Zap className="h-3 w-3" />
									150x realtime
								</span>
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									48kHz output
								</span>
							</div>
						</div>

						{/* Qwen CustomVoice */}
						<div className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-6 transition-colors hover:border-accent/30">
							<div className="flex items-start justify-between mb-3">
								<div>
									<h3 className="text-base font-semibold text-foreground">
										Qwen CustomVoice
									</h3>
									<span className="text-xs text-muted-foreground/60">
										by Alibaba
									</span>
								</div>
								<div className="flex gap-1.5">
									<span className="text-[10px] px-2 py-0.5 rounded-full border border-border bg-background text-muted-foreground">
										1.7B
									</span>
									<span className="text-[10px] px-2 py-0.5 rounded-full border border-border bg-background text-muted-foreground">
										0.6B
									</span>
								</div>
							</div>
							<p className="text-sm text-muted-foreground leading-relaxed mb-4">
								Nine premium preset speakers with natural-language style
								control. Tell the model how to deliver — "speak slowly with
								warmth", "authoritative and clear" — and it adapts tone,
								emotion, and pace.
							</p>
							<div className="flex flex-wrap gap-2">
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									<SlidersHorizontal className="h-3 w-3" />
									Instruct control
								</span>
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									<Globe className="h-3 w-3" />
									10 languages
								</span>
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									9 preset voices
								</span>
							</div>
						</div>

						{/* HumeAI TADA */}
						<div className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-6 transition-colors hover:border-accent/30">
							<div className="flex items-start justify-between mb-3">
								<div>
									<h3 className="text-base font-semibold text-foreground">
										TADA
									</h3>
									<span className="text-xs text-muted-foreground/60">
										by Hume AI
									</span>
								</div>
								<div className="flex gap-1.5">
									<span className="text-[10px] px-2 py-0.5 rounded-full border border-border bg-background text-muted-foreground">
										3B
									</span>
									<span className="text-[10px] px-2 py-0.5 rounded-full border border-border bg-background text-muted-foreground">
										1B
									</span>
								</div>
							</div>
							<p className="text-sm text-muted-foreground leading-relaxed mb-4">
								Speech-language model with text-acoustic dual alignment. Built
								for long-form generation — produces 700s+ of coherent audio
								without drift. Multilingual at 3B, English-focused at 1B.
							</p>
							<div className="flex flex-wrap gap-2">
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									<Globe className="h-3 w-3" />
									10 languages
								</span>
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									Long-form coherent
								</span>
							</div>
						</div>

						{/* Kokoro 82M */}
						<div className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-6 transition-colors hover:border-accent/30">
							<div className="flex items-start justify-between mb-3">
								<div>
									<h3 className="text-base font-semibold text-foreground">
										Kokoro
									</h3>
									<span className="text-xs text-muted-foreground/60">
										by hexgrad · Apache 2.0
									</span>
								</div>
								<span className="text-[10px] px-2 py-0.5 rounded-full border border-border bg-background text-muted-foreground">
									82M
								</span>
							</div>
							<p className="text-sm text-muted-foreground leading-relaxed mb-4">
								Tiny 82M-parameter TTS that runs at CPU realtime with negligible
								VRAM. Pre-built voice styles instead of cloning — pick a voice,
								type, generate. Smallest footprint of any engine.
							</p>
							<div className="flex flex-wrap gap-2">
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									<Zap className="h-3 w-3" />
									CPU realtime
								</span>
								<span className="flex items-center gap-1 text-[11px] text-muted-foreground/70">
									Preset voices
								</span>
							</div>
						</div>
					</div>
				</div>
			</section>

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

			{/* ── Footer ───────────────────────────────────────────────── */}
			<Footer />
		</>
	);
}
