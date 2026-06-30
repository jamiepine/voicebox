import type {Metadata} from "next";
import Link from "next/link";
import {notFound} from "next/navigation";
import {Footer} from "@/components/Footer";
import {Navbar} from "@/components/Navbar";
import {formatDate, getPost, loadAllPosts} from "@/lib/blog";

export function generateStaticParams() {
	return loadAllPosts().map((post) => ({slug: post.slug}));
}

export async function generateMetadata({
	params,
}: {
	params: Promise<{slug: string}>;
}): Promise<Metadata> {
	const {slug} = await params;
	const post = getPost(slug);
	if (!post) return {title: "Post not found — Voicebox"};
	return {
		title: `${post.title} — Voicebox`,
		description: post.excerpt,
		openGraph: {
			title: post.title,
			description: post.excerpt,
			type: "article",
			url: `https://voicebox.sh/blog/${post.slug}`,
			// og:image / twitter:image come from the colocated opengraph-image.tsx
		},
		twitter: {
			card: "summary_large_image",
			title: post.title,
			description: post.excerpt,
		},
	};
}

export default async function BlogPostPage({
	params,
}: {
	params: Promise<{slug: string}>;
}) {
	const {slug} = await params;
	const post = getPost(slug);
	if (!post) notFound();

	return (
		<>
			<Navbar />

			<main className="mx-auto w-full max-w-3xl px-6 pt-32 pb-20">
				<Link
					href="/blog"
					className="font-mono text-sm text-muted-foreground underline-offset-4 transition-colors hover:text-foreground hover:underline"
				>
					← Back to blog
				</Link>

				<header className="mt-8 border-b border-border pb-10">
					{post.tags.length > 0 ? (
						<div className="mb-5 flex flex-wrap gap-2">
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
					<h1 className="text-4xl md:text-5xl font-bold tracking-tighter text-foreground">
						{post.title}
					</h1>
					<p className="mt-5 font-mono text-sm text-muted-foreground">
						by <span className="text-foreground">{post.author}</span> ·{" "}
						{formatDate(post.date)} · {post.readingMinutes} min read
					</p>
				</header>

				<article
					className="blog-prose mt-10"
					// Content is authored markdown from this repo, not user input.
					// biome-ignore lint/security/noDangerouslySetInnerHtml: trusted local markdown
					dangerouslySetInnerHTML={{__html: post.html}}
				/>
			</main>

			<Footer />
		</>
	);
}
