import {ArrowUpRight, Coins} from "lucide-react";
import {TOKEN_TICKER} from "@/lib/constants";

/**
 * Compact teaser shown near the bottom of the landing page. The full token
 * details (contract address, buy CTA, disclaimers) live on the dedicated
 * /token page — this just points there.
 */
export function TokenTeaser() {
	return (
		<section className="border-t border-border py-20">
			<div className="mx-auto max-w-4xl px-6">
				<div className="relative flex flex-col items-center gap-5 overflow-hidden rounded-2xl border border-border bg-card/40 backdrop-blur-sm px-6 py-10 text-center">
					{/* Subtle accent glow */}
					<div className="pointer-events-none absolute inset-0 -z-10 flex justify-center">
						<div className="h-[200px] w-[420px] rounded-full bg-accent/10 blur-[130px]" />
					</div>

					<div className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-card/40 px-3 py-1">
						<Coins className="h-3 w-3 text-accent" />
						<span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
							Official token
						</span>
					</div>

					<h2 className="text-2xl font-semibold tracking-tight text-foreground md:text-3xl">
						{TOKEN_TICKER} on Solana
					</h2>

					<p className="max-w-xl text-muted-foreground">
						An optional way to back the project and have some fun. Voicebox is and
						always will be{" "}
						<b className="text-foreground">free and open source</b> — the token is
						not required to use anything here.
					</p>

					<a
						href="/token"
						className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 text-sm font-semibold text-white shadow-[0_4px_20px_hsl(43_60%_50%/0.3),inset_0_2px_0_rgba(255,255,255,0.2),inset_0_-2px_0_rgba(0,0,0,0.1)] transition-all hover:bg-accent-faint"
					>
						Learn about {TOKEN_TICKER}
						<ArrowUpRight className="h-4 w-4" />
					</a>
				</div>
			</div>
		</section>
	);
}
