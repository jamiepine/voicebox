"use client";

import {ArrowUpRight, Check, Coins, Copy} from "lucide-react";
import {useState} from "react";
import {
	TOKEN_CONTRACT_ADDRESS,
	TOKEN_PUMP_URL,
	TOKEN_TICKER,
} from "@/lib/constants";

export function TokenSection() {
	const [copied, setCopied] = useState(false);

	const handleCopy = async () => {
		try {
			await navigator.clipboard.writeText(TOKEN_CONTRACT_ADDRESS);
			setCopied(true);
			setTimeout(() => setCopied(false), 2000);
		} catch {
			// Clipboard unavailable (e.g. insecure context) — silently no-op.
		}
	};

	return (
		<section id="token" className="border-t border-border py-24">
			<div className="relative mx-auto max-w-4xl px-6">
				{/* Subtle accent glow */}
				<div className="pointer-events-none absolute inset-0 -z-10 flex justify-center">
					<div className="h-[260px] w-[520px] rounded-full bg-accent/10 blur-[140px]" />
				</div>

				{/* Header */}
				<div className="text-center mb-10">
					<div className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-card/40 backdrop-blur-sm px-3 py-1 mb-4">
						<Coins className="h-3 w-3 text-accent" />
						<span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
							Official token
						</span>
					</div>
					<h2 className="text-3xl font-semibold tracking-tight text-foreground md:text-4xl mb-4">
						{TOKEN_TICKER} on Solana
					</h2>
					<p className="text-muted-foreground max-w-2xl mx-auto">
						The official {TOKEN_TICKER} token for supporters who want to back
						the project and have some fun. Voicebox is and always will be{" "}
						<b className="text-foreground">free and open source</b> — the token
						is entirely optional and not required to use anything here.
					</p>
				</div>

				{/* Contract address + CTA */}
				<div className="mx-auto max-w-2xl rounded-2xl border border-border bg-card/60 backdrop-blur-sm p-5 sm:p-6">
					<div className="flex items-center gap-2 mb-3">
						<span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
							Contract address
						</span>
						<span className="ml-auto inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-background/60 px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
							<span className="h-1.5 w-1.5 rounded-full bg-accent" />
							Solana
						</span>
					</div>

					<div className="flex flex-col gap-3 sm:flex-row sm:items-center">
						{/* Address bar */}
						<button
							type="button"
							onClick={handleCopy}
							title="Copy contract address"
							aria-label={
								copied ? "Contract address copied" : "Copy contract address"
							}
							className="group flex min-w-0 flex-1 items-center gap-3 rounded-xl border border-border bg-background/60 px-4 py-3 text-left transition-colors hover:border-accent/40"
						>
							<code className="min-w-0 flex-1 truncate font-mono text-xs text-foreground/90 sm:text-sm">
								{TOKEN_CONTRACT_ADDRESS}
							</code>
							{copied ? (
								<span className="inline-flex shrink-0 items-center gap-1.5 text-xs font-medium text-accent">
									<Check className="h-4 w-4" />
									Copied
								</span>
							) : (
								<span className="inline-flex shrink-0 items-center gap-1.5 text-xs font-medium text-muted-foreground transition-colors group-hover:text-foreground">
									<Copy className="h-4 w-4" />
									Copy
								</span>
							)}
						</button>

						{/* Buy CTA */}
						<a
							href={TOKEN_PUMP_URL}
							target="_blank"
							rel="noopener noreferrer"
							className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-accent px-5 py-3 text-sm font-semibold text-white shadow-[0_4px_20px_hsl(43_60%_50%/0.3),inset_0_2px_0_rgba(255,255,255,0.2),inset_0_-2px_0_rgba(0,0,0,0.1)] transition-all hover:bg-accent-faint"
						>
							Buy on pump.fun
							<ArrowUpRight className="h-4 w-4" />
						</a>
					</div>
				</div>
			</div>
		</section>
	);
}
