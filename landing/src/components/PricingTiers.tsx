"use client";

import {Check} from "lucide-react";
import {useState} from "react";
import {
	annualSavingsPercent,
	type BillingPeriod,
	PRICING_TIERS,
} from "@/lib/pricing";

const MAX_ANNUAL_SAVINGS = Math.max(
	...PRICING_TIERS.map((t) => annualSavingsPercent(t)),
);

export function PricingTiers() {
	const [period, setPeriod] = useState<BillingPeriod>("annual");

	return (
		<div className="mx-auto max-w-6xl px-6">
			{/* Billing toggle */}
			<div className="mb-10 flex items-center justify-center">
				<div className="inline-flex items-center gap-1 rounded-full border border-border bg-card/40 p-1">
					<ToggleButton
						active={period === "monthly"}
						onClick={() => setPeriod("monthly")}
					>
						Monthly
					</ToggleButton>
					<ToggleButton
						active={period === "annual"}
						onClick={() => setPeriod("annual")}
					>
						Annual
						{MAX_ANNUAL_SAVINGS > 0 ? (
							<span
								className={`ml-1.5 rounded-full px-1.5 py-0.5 text-[10px] font-semibold transition-colors ${
									period === "annual"
										? "bg-white/25 text-white"
										: "bg-accent/15 text-accent"
								}`}
							>
								Save {MAX_ANNUAL_SAVINGS}%
							</span>
						) : null}
					</ToggleButton>
				</div>
			</div>

			<div className="grid items-start gap-4 md:grid-cols-3">
				{PRICING_TIERS.map((tier) => {
					const isFree = tier.monthly === 0 && tier.annual === 0;
					const amount = period === "monthly" ? tier.monthly : tier.annual;
					const unit = period === "monthly" ? "/month" : "/year";
					const isExternal = tier.cta.href.startsWith("http");

					return (
						<div
							key={tier.id}
							className={`flex flex-col rounded-2xl border bg-card/50 backdrop-blur-sm p-7 ${
								tier.highlighted
									? "border-2 border-accent/50 shadow-[0_8px_40px_hsl(43_60%_50%/0.1)] md:-mt-2"
									: "border-border"
							}`}
						>
							<div className="mb-1 flex items-center justify-between gap-2">
								<h2 className="text-lg font-semibold text-foreground">
									{tier.name}
								</h2>
								{tier.badge ? (
									<span
										className={`rounded-full px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${
											tier.highlighted
												? "bg-accent/15 text-accent"
												: "border border-border/60 text-muted-foreground"
										}`}
									>
										{tier.badge}
									</span>
								) : null}
							</div>

							<p className="mb-5 min-h-[2.5rem] text-sm text-muted-foreground">
								{tier.tagline}
							</p>

							<div className="mb-1 flex items-baseline gap-1">
								<span className="text-4xl font-bold tracking-tight text-foreground">
									${amount}
								</span>
								{!isFree ? (
									<span className="text-sm text-muted-foreground">{unit}</span>
								) : null}
							</div>
							<p className="mb-6 min-h-[1rem] text-xs text-muted-foreground/70">
								{isFree
									? tier.priceNote
									: period === "annual"
										? tier.priceNote
										: `Billed monthly · ${tier.priceNote ?? ""}`}
							</p>

							<a
								href={tier.cta.href}
								{...(isExternal
									? {target: "_blank", rel: "noopener noreferrer"}
									: {})}
								className={`mb-7 inline-flex items-center justify-center rounded-full px-5 py-3 text-sm font-semibold transition-all ${
									tier.highlighted
										? "bg-accent text-white shadow-[0_4px_20px_hsl(43_60%_50%/0.3)] hover:bg-accent-faint"
										: "border border-border/60 bg-card/40 text-foreground hover:border-accent/40"
								}`}
							>
								{tier.cta.label}
							</a>

							<ul className="space-y-3">
								{tier.features.map((feature) => (
									<li
										key={feature}
										className="flex items-start gap-3 text-sm text-foreground/90"
									>
										<Check className="h-4 w-4 shrink-0 text-accent mt-0.5" />
										<span>{feature}</span>
									</li>
								))}
							</ul>
						</div>
					);
				})}
			</div>
		</div>
	);
}

function ToggleButton({
	active,
	onClick,
	children,
}: {
	active: boolean;
	onClick: () => void;
	children: React.ReactNode;
}) {
	return (
		<button
			type="button"
			onClick={onClick}
			className={`flex items-center rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
				active
					? "bg-accent text-white"
					: "text-muted-foreground hover:text-foreground"
			}`}
		>
			{children}
		</button>
	);
}
