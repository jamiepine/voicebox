import type {Metadata} from "next";
import Link from "next/link";
import {Footer} from "@/components/Footer";
import {Navbar} from "@/components/Navbar";
import {formatDate, listPosts} from "@/lib/blog";

export const metadata: Metadata = {
	title: "Blog — Voicebox",
	description: "Notes from building Voicebox — the open-source AI voice studio.",
	openGraph: {
		title: "Voicebox Blog",
		description: "Notes from building Voicebox — the open-source AI voice studio.",
		type: "website",
		url: "https://voicebox.sh/blog",
		images: [{url: "/og.webp", width: 1200, height: 630}],
	},
};

export default function BlogIndexPage() {
	const posts = listPosts();

	return (
		<>
			<Navbar />

			<main className="mx-auto w-full max-w-3xl px-6 pt-32 pb-20">
				<div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-4">
					Blog
				</div>
				<h1 className="text-4xl md:text-5xl font-bold tracking-tighter text-foreground">
					Notes from building Voicebox.
				</h1>
				<p className="mt-5 max-w-2xl text-lg text-muted-foreground">
					The story behind the project, what's shipping next, and the occasional
					look under the hood.
				</p>

				{posts.length === 0 ? (
					<p className="mt-16 border-t border-border pt-10 text-muted-foreground">
						Nothing published yet.
					</p>
				) : (
					<ul className="mt-16 border-t border-border">
						{posts.map((post) => (
							<li key={post.slug}>
								<Link
									href={`/blog/${post.slug}`}
									className="group grid gap-4 border-b border-border py-10 md:grid-cols-[11rem_1fr] md:gap-10"
								>
									<div className="font-mono text-sm text-muted-foreground md:pt-1.5">
										<p>{formatDate(post.date)}</p>
										<p className="mt-1">{post.readingMinutes} min read</p>
									</div>
									<div className="max-w-2xl">
										<h2 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground transition-colors group-hover:text-accent">
											{post.title}
										</h2>
										{post.excerpt ? (
											<p className="mt-3 leading-7 text-muted-foreground">
												{post.excerpt}
											</p>
										) : null}
										{post.tags.length > 0 ? (
											<div className="mt-5 flex flex-wrap gap-2">
												{post.tags.map((tag) => (
													<span
														key={tag}
														className="rounded-full border border-border/60 bg-card/40 px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground"
													>
														{tag}
													</span>
												))}
											</div>
										) : null}
									</div>
								</Link>
							</li>
						))}
					</ul>
				)}
			</main>

			<Footer />
		</>
	);
}
