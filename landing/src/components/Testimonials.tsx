"use client";

import {Heart, Quote} from "lucide-react";
import {DONATE_URL} from "@/lib/constants";

type Testimonial = {
	quote: string;
	author: string;
};

/** Verbatim supporter messages from Buy Me a Coffee. */
const TESTIMONIALS: Testimonial[] = [
	{
		quote:
			"Cloning my own voice was a snap — and now I can hear my reminders and to-dos from my digital doppelgänger. Very cool.",
		author: "jimzip",
	},
	{
		quote: "It's better than most other paid services.",
		author: "Peiming Pai",
	},
	{
		quote:
			"I'm using this great tool for my multimedia course project — you made it into the classrooms!",
		author: "theanoma.ly",
	},
	{
		quote:
			"This app is fantastic! I use it to learn languages and for my learning materials. Congratulations, it's great!",
		author: "Kevin Serrano",
	},
	{
		quote:
			"Absolutely amazing! The learning curve was very short. Thank you for a great program and for making it free.",
		author: "DJWhy",
	},
	{
		quote: "First engine I tried, zero config. It worked! Amazing.",
		author: "Fitz",
	},
	{
		quote:
			"This is great for people who are uncomfortable with advocating for themselves in public. Thanks for making it.",
		author: "creativeaction.ca",
	},
	{
		quote: "Thanks for this. It's a life-saver!",
		author: "The Cowboy Movie Channel",
	},
	{
		quote: "Fantastic open-source app!",
		author: "Mitja",
	},
];

export function Testimonials() {
	return (
		<section id="testimonials" className="border-t border-border py-24">
			<div className="mx-auto max-w-6xl px-6">
				{/* Header */}
				<div className="text-center mb-14">
					<div className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-card/40 backdrop-blur-sm px-3 py-1 mb-4">
						<Heart className="h-3 w-3 text-accent" />
						<span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
							Loved by users
						</span>
					</div>
					<h2 className="text-3xl font-semibold tracking-tight text-foreground md:text-4xl mb-4">
						What people are saying
					</h2>
					<p className="text-muted-foreground max-w-2xl mx-auto">
						Voicebox has passed 1M+ downloads. Here's a handful of notes from
						the people using it every day.
					</p>
				</div>

				{/* Masonry-style columns so cards flow naturally regardless of length */}
				<div className="columns-1 gap-4 sm:columns-2 lg:columns-3 [&>*]:mb-4">
					{TESTIMONIALS.map((t) => (
						<figure
							key={t.author}
							className="break-inside-avoid rounded-xl border border-border bg-card/60 backdrop-blur-sm p-5 transition-colors hover:border-accent/30"
						>
							<Quote className="h-4 w-4 text-accent/60 mb-3" />
							<blockquote className="text-sm leading-relaxed text-foreground/90">
								{t.quote}
							</blockquote>
							<figcaption className="mt-4 text-xs font-medium text-muted-foreground">
								{t.author}
							</figcaption>
						</figure>
					))}
				</div>

				{/* Attribution + soft CTA */}
				<div className="mt-10 text-center">
					<a
						href={DONATE_URL}
						target="_blank"
						rel="noopener noreferrer"
						className="text-xs text-muted-foreground/70 hover:text-foreground transition-colors"
					>
						From supporters on Buy Me a Coffee →
					</a>
				</div>
			</div>
		</section>
	);
}
