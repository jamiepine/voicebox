"use client";

import {Play, Youtube} from "lucide-react";

type Tutorial = {
	id: string;
	title: string;
	author: string;
	thumbnail: string;
};

const TUTORIALS: (Tutorial | null)[] = [
	{
		id: "sisnzgc73zc",
		title: "Free AI Voice Generator on Your PC (Clones Any Voice)",
		author: "Kevin Stratvert",
		thumbnail: "/tutorials/sisnzgc73zc.jpg",
	},
	{
		id: "woQe90k7g3c",
		title: "NEW Voicebox DESTROYS ElevenLabs?",
		author: "Julian Goldie SEO",
		thumbnail: "/tutorials/woQe90k7g3c.jpg",
	},
	{
		id: "kqxqjRsdD5E",
		title: "This Open-Source TTS App Sounds Scary Good (And It's Free)",
		author: "Dave Swift",
		thumbnail: "/tutorials/kqxqjRsdD5E.jpg",
	},
	{
		id: "yu9QHqOEqqA",
		title: "This FREE AI Tool Just Destroyed ElevenLabs Voice Cloning (VoiceBox)",
		author: "Danish Sofi",
		thumbnail: "/tutorials/yu9QHqOEqqA.jpg",
	},
	{
		id: "RRRBxNXgeKQ",
		title: "Get Started with Voicebox: Open-Source Alternative to ElevenLabs Tutorial",
		author: "StinkyScrublet",
		thumbnail: "/tutorials/RRRBxNXgeKQ.jpg",
	},
	{
		id: "PyMx4L9mky4",
		title: "Free AI Voice Generator (Clones Any Voice)",
		author: "mikbes",
		thumbnail: "/tutorials/PyMx4L9mky4.jpg",
	},
];

function TutorialCard({tutorial}: {tutorial: Tutorial}) {
	return (
		<a
			href={`https://www.youtube.com/watch?v=${tutorial.id}`}
			target="_blank"
			rel="noopener noreferrer"
			className="group rounded-xl border border-border bg-card/60 backdrop-blur-sm overflow-hidden transition-all hover:border-accent/30 hover:bg-card"
		>
			<div className="relative aspect-video overflow-hidden bg-muted">
				{/* eslint-disable-next-line @next/next/no-img-element */}
				<img
					src={tutorial.thumbnail}
					alt={tutorial.title}
					className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-105"
				/>
				{/* Gradient overlay */}
				<div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/0 to-black/0" />
				{/* Play button overlay */}
				<div className="absolute inset-0 flex items-center justify-center">
					<div className="flex h-14 w-14 items-center justify-center rounded-full bg-black/50 backdrop-blur-md border border-white/20 transition-all group-hover:scale-110 group-hover:bg-accent/90 group-hover:border-accent">
						<Play className="h-5 w-5 text-white fill-white ml-0.5" />
					</div>
				</div>
				{/* YouTube badge */}
				<div className="absolute top-3 right-3 flex items-center gap-1 rounded bg-black/60 backdrop-blur-sm px-2 py-1">
					<Youtube className="h-3 w-3 text-white" />
					<span className="text-[10px] font-medium text-white uppercase tracking-wider">
						YouTube
					</span>
				</div>
			</div>
			<div className="p-4">
				<h3 className="text-sm font-medium text-foreground line-clamp-2 leading-snug mb-1.5 group-hover:text-accent transition-colors">
					{tutorial.title}
				</h3>
				<p className="text-xs text-muted-foreground">{tutorial.author}</p>
			</div>
		</a>
	);
}

function TutorialPlaceholder() {
	return (
		<div className="rounded-xl border border-dashed border-border/60 bg-card/30 backdrop-blur-sm overflow-hidden">
			<div className="relative aspect-video overflow-hidden bg-gradient-to-br from-card via-muted/20 to-card">
				<div className="absolute inset-0 flex items-center justify-center">
					<div className="flex h-14 w-14 items-center justify-center rounded-full border border-border/40 bg-card/40">
						<Play className="h-5 w-5 text-muted-foreground/40 fill-muted-foreground/40 ml-0.5" />
					</div>
				</div>
			</div>
			<div className="p-4">
				<div className="h-3 w-3/4 rounded bg-muted-foreground/10 mb-2" />
				<div className="h-2.5 w-1/3 rounded bg-muted-foreground/10" />
				<p className="text-[11px] text-muted-foreground/50 mt-3 uppercase tracking-wider">
					Coming soon
				</p>
			</div>
		</div>
	);
}

export function TutorialsSection() {
	return (
		<section id="tutorials" className="border-t border-border py-24">
			<div className="mx-auto max-w-6xl px-6">
				<div className="text-center mb-14">
					<div className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-card/40 backdrop-blur-sm px-3 py-1 mb-4">
						<Youtube className="h-3 w-3 text-accent" />
						<span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
							Video tutorials
						</span>
					</div>
					<h2 className="text-3xl font-semibold tracking-tight text-foreground md:text-4xl mb-4">
						Learn by watching
					</h2>
					<p className="text-muted-foreground max-w-2xl mx-auto">
						Walkthroughs from the community covering setup, voice cloning, and
						production workflows.
					</p>
				</div>

				<div className="grid grid-cols-1 md:grid-cols-3 gap-5">
					{TUTORIALS.map((tutorial, i) =>
						tutorial ? (
							<TutorialCard key={tutorial.id} tutorial={tutorial} />
						) : (
							<TutorialPlaceholder key={`placeholder-${i}`} />
						),
					)}
				</div>
			</div>
		</section>
	);
}
