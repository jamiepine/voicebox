// Pricing + cloud data — single source of truth for /pricing and /cloud.
//
// DRAFT: the cloud service is not live yet and the pricing model is not final.
// $12/yr is a launch/intro figure; tier limits below are placeholders to shape
// the page — tune them once the model is decided. Keeping it all here means the
// pages update by editing this file only.

export const CLOUD_STATUS = "coming-soon" as const; // → flips to "live" at launch
export const CLOUD_PRICE_YEARLY = 12; // launch price for the Cloud tier (USD/yr)

/** Where "get notified" CTAs point until there's a real signup flow. */
export const CLOUD_NOTIFY_URL = "https://x.com/VoiceboxAI";

export type BillingPeriod = "monthly" | "annual";

export interface PricingTier {
	id: string;
	name: string;
	tagline: string;
	/** USD per month. 0 = free. */
	monthly: number;
	/** USD per year. 0 = free. Set below 12×monthly to reward annual. */
	annual: number;
	priceNote?: string;
	/** Visually emphasize this tier (the headline plan). */
	highlighted?: boolean;
	/** Status pill, e.g. "Available now" / "Coming soon". */
	badge?: string;
	cta: {label: string; href: string};
	features: string[];
}

export const PRICING_TIERS: PricingTier[] = [
	{
		id: "local",
		name: "Local",
		tagline: "The full app, free forever.",
		monthly: 0,
		annual: 0,
		priceNote: "No account required",
		badge: "Available now",
		cta: {label: "Download Voicebox", href: "/download"},
		features: [
			"Voice cloning across every TTS engine",
			"Dictation & Capture (audio kept alongside transcript)",
			"MCP / agent integration & personalities",
			"Unlimited local generations & captures",
			"100% open source, runs entirely on your machine",
		],
	},
	{
		id: "cloud",
		name: "Cloud",
		tagline: "Backup & sync for everything you make.",
		monthly: 2, // placeholder
		annual: CLOUD_PRICE_YEARLY, // $12 launch price (≈50% off monthly)
		priceNote: "Launch price · free for $VOICEBOX holders",
		highlighted: true,
		badge: "Coming soon",
		cta: {label: "Get notified", href: CLOUD_NOTIFY_URL},
		features: [
			"Everything in Local",
			"End-to-end encrypted backup — we can't read it",
			"Sync across desktop & mobile",
			"25 GB encrypted storage", // placeholder
			"Up to 5 devices", // placeholder
			"30-day version history", // placeholder
		],
	},
	{
		id: "studio",
		name: "Studio",
		tagline: "For power users and professionals.",
		monthly: 6, // placeholder
		annual: 48, // placeholder
		priceNote: "Placeholder — pricing TBD",
		badge: "Coming soon",
		cta: {label: "Get notified", href: CLOUD_NOTIFY_URL},
		features: [
			"Everything in Cloud",
			"250 GB encrypted storage", // placeholder
			"Unlimited devices", // placeholder
			"1-year version history", // placeholder
			"Priority support",
		],
	},
];

/** Annual savings vs paying 12× the monthly price, as a whole percent (0 if none). */
export function annualSavingsPercent(tier: PricingTier): number {
	if (tier.monthly <= 0 || tier.annual <= 0) return 0;
	const full = tier.monthly * 12;
	return Math.max(0, Math.round((1 - tier.annual / full) * 100));
}

export interface CloudFeature {
	title: string;
	body: string;
}

export const CLOUD_FEATURES: CloudFeature[] = [
	{
		title: "End-to-end encrypted",
		body: "Everything is encrypted on your device before it leaves it. The server only ever stores opaque blobs — it literally cannot read your voices, generations, or captures.",
	},
	{
		title: "Back up everything",
		body: "Voice profiles, generations, and captures — including the original audio Voicebox keeps alongside each transcript — safe off your machine.",
	},
	{
		title: "Sync across devices",
		body: "Pick up on desktop or mobile. Your library follows you, encrypted in transit and at rest, with a per-device key.",
	},
	{
		title: "Generate on the go",
		body: "The mobile app pairs with your library so you can dictate and generate anywhere, then find it all waiting back at your desk.",
	},
	{
		title: "You hold the keys",
		body: "A recovery phrase you control is the root of your encryption. Lose your devices and you can still restore — but no one else, including us, ever can.",
	},
	{
		title: "Optional & local-first",
		body: "Voicebox works fully offline without an account. Cloud is an add-on for when you want backup and sync — never a requirement.",
	},
];
