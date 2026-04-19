"use client";

import {AppWindow, Code2, Gamepad2, Terminal, Wrench} from "lucide-react";

type Endpoint = {
	method: "POST" | "GET" | "DELETE" | "PATCH";
	path: string;
	label: string;
};

const ENDPOINTS: Endpoint[] = [
	{method: "POST", path: "/generate", label: "Generate speech"},
	{method: "POST", path: "/generate/{id}/cancel", label: "Cancel a generation"},
	{method: "GET", path: "/profiles", label: "List voice profiles"},
	{method: "POST", path: "/profiles", label: "Create a new profile"},
	{method: "GET", path: "/models/status", label: "Model catalog & state"},
	{method: "GET", path: "/history", label: "Past generations"},
	{method: "GET", path: "/health", label: "Server health"},
];

const METHOD_STYLES: Record<Endpoint["method"], string> = {
	POST: "bg-accent/10 text-accent border-accent/20",
	GET: "bg-muted text-muted-foreground border-border",
	DELETE: "bg-red-500/10 text-red-400 border-red-500/20",
	PATCH: "bg-blue-500/10 text-blue-400 border-blue-500/20",
};

const CURL_SNIPPET = `curl -X POST http://127.0.0.1:17493/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Welcome to the game, player one.",
    "profile_id": "b3f1c2d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
    "engine": "qwen_custom_voice",
    "instruct": "warm, slow, cinematic"
  }' \\
  --output line.wav`;

const USE_CASES = [
	{
		icon: Gamepad2,
		title: "Games",
		description:
			"Generate NPC dialogue on the fly, localize characters into new languages, or ship expressive voice lines without a studio.",
	},
	{
		icon: AppWindow,
		title: "Apps & agents",
		description:
			"Give your app or AI agent a voice. Real-time narration, accessibility readouts, voice replies — all running on the user's machine.",
	},
	{
		icon: Wrench,
		title: "Scripts & tools",
		description:
			"Batch-generate audiobook chapters, automate podcast intros, or wire Voicebox into your Stream Deck. It's just a localhost URL.",
	},
];

export function ApiSection() {
	return (
		<section id="api" className="border-t border-border py-24">
			<div className="mx-auto max-w-6xl px-6">
				{/* Header */}
				<div className="text-center mb-14">
					<div className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-card/40 backdrop-blur-sm px-3 py-1 mb-4">
						<Code2 className="h-3 w-3 text-accent" />
						<span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
							Built-in REST API
						</span>
					</div>
					<h2 className="text-3xl font-semibold tracking-tight text-foreground md:text-4xl mb-4">
						Your local voice API
					</h2>
					<p className="text-muted-foreground max-w-2xl mx-auto">
						Every engine you download becomes a REST endpoint on your machine.
						Build apps, games, and voice tools with full programmatic control —
						no API keys, no rate limits, no per-character fees.
					</p>
				</div>

				{/* Main panel: endpoints + code snippet */}
				<div className="grid grid-cols-1 lg:grid-cols-5 gap-5 mb-14">
					{/* Endpoint reference */}
					<div className="lg:col-span-3 rounded-xl border border-border bg-card/60 backdrop-blur-sm overflow-hidden">
						<div className="flex items-center justify-between px-5 py-3 border-b border-border/60 bg-card/40">
							<div className="flex items-center gap-2">
								<div className="flex gap-1">
									<div className="h-2 w-2 rounded-full bg-muted-foreground/30" />
									<div className="h-2 w-2 rounded-full bg-muted-foreground/30" />
									<div className="h-2 w-2 rounded-full bg-muted-foreground/30" />
								</div>
								<span className="text-xs font-medium text-foreground ml-2">
									API Reference
								</span>
							</div>
							<code className="text-[10px] bg-background border border-border px-1.5 py-0.5 rounded font-mono text-muted-foreground">
								http://127.0.0.1:17493
							</code>
						</div>
						<div className="px-5 py-4 space-y-1">
							{ENDPOINTS.map((ep) => (
								<div
									key={`${ep.method}-${ep.path}`}
									className="flex items-center gap-3 py-1.5 group"
								>
									<span
										className={`text-[10px] font-mono font-semibold w-12 text-center rounded px-1 py-0.5 border ${METHOD_STYLES[ep.method]}`}
									>
										{ep.method}
									</span>
									<code className="text-xs font-mono text-foreground/90">
										{ep.path}
									</code>
									<span className="text-xs text-muted-foreground/60 ml-auto">
										{ep.label}
									</span>
								</div>
							))}
						</div>
						<div className="border-t border-border/60 px-5 py-3 bg-card/40">
							<a
								href="http://127.0.0.1:17493/docs"
								target="_blank"
								rel="noopener noreferrer"
								className="text-xs text-accent hover:underline"
							>
								See the full OpenAPI reference at{" "}
								<code className="font-mono">/docs</code> when Voicebox is running
								→
							</a>
						</div>
					</div>

					{/* Code snippet */}
					<div className="lg:col-span-2 rounded-xl border border-border bg-card/60 backdrop-blur-sm overflow-hidden flex flex-col">
						<div className="flex items-center gap-2 px-4 py-3 border-b border-border/60 bg-card/40">
							<Terminal className="h-3.5 w-3.5 text-muted-foreground" />
							<span className="text-xs font-medium text-foreground">
								Generate a line
							</span>
							<span className="ml-auto text-[10px] text-muted-foreground/50 font-mono">
								curl
							</span>
						</div>
						<pre className="flex-1 p-4 text-[11px] font-mono text-muted-foreground/90 leading-relaxed overflow-x-auto whitespace-pre">
							<code>{CURL_SNIPPET}</code>
						</pre>
					</div>
				</div>

				{/* Use cases */}
				<div className="grid grid-cols-1 md:grid-cols-3 gap-4">
					{USE_CASES.map((uc) => {
						const Icon = uc.icon;
						return (
							<div
								key={uc.title}
								className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-5 transition-colors hover:border-accent/30"
							>
								<div className="flex items-center gap-2 mb-2">
									<Icon className="h-4 w-4 text-accent" />
									<h3 className="text-[15px] font-medium text-foreground">
										{uc.title}
									</h3>
								</div>
								<p className="text-sm leading-relaxed text-muted-foreground">
									{uc.description}
								</p>
							</div>
						);
					})}
				</div>

				{/* Bottom bar: key selling points */}
				<div className="mt-10 flex flex-wrap items-center justify-center gap-x-8 gap-y-2 text-xs text-muted-foreground">
					<span className="flex items-center gap-1.5">
						<span className="h-1.5 w-1.5 rounded-full bg-accent" />
						No API keys
					</span>
					<span className="flex items-center gap-1.5">
						<span className="h-1.5 w-1.5 rounded-full bg-accent" />
						No rate limits
					</span>
					<span className="flex items-center gap-1.5">
						<span className="h-1.5 w-1.5 rounded-full bg-accent" />
						No per-character fees
					</span>
					<span className="flex items-center gap-1.5">
						<span className="h-1.5 w-1.5 rounded-full bg-accent" />
						Works offline
					</span>
					<span className="flex items-center gap-1.5">
						<span className="h-1.5 w-1.5 rounded-full bg-accent" />
						Your audio, your machine
					</span>
				</div>
			</div>
		</section>
	);
}
