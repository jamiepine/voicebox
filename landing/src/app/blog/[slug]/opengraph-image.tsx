import {readFileSync} from "node:fs";
import {join} from "node:path";
import {ImageResponse} from "next/og";
import {formatDate, getPost, loadAllPosts} from "@/lib/blog";

// Per-post Open Graph image, generated with Satori at build time (static export
// of each post route) and served as PNG. Note: this runs in the Satori renderer,
// which only understands inline styles + flexbox and a subset of CSS — no
// Tailwind classes, no `filter: blur()`. Glows are done with radial gradients.

export const size = {width: 1200, height: 630};
export const contentType = "image/png";
export const alt = "Voicebox Blog";

// Pre-build an image for every post route (mirrors the page's static params).
export function generateStaticParams() {
	return loadAllPosts().map((post) => ({slug: post.slug}));
}

// 8-bit PNG decodes reliably in Satori; the 1024px logos are 16-bit and don't.
const logo = `data:image/png;base64,${readFileSync(
	join(process.cwd(), "public/apple-touch-icon.png"),
).toString("base64")}`;

function titleFontSize(title: string): number {
	if (title.length <= 38) return 76;
	if (title.length <= 64) return 60;
	return 48;
}

export default async function OgImage({
	params,
}: {
	params: Promise<{slug: string}>;
}) {
	const {slug} = await params;
	const post = getPost(slug);
	const title = post?.title ?? "Voicebox Blog";
	const meta = post
		? `${post.author} · ${formatDate(post.date)}`
		: "Open source voice cloning. Local-first.";

	return new ImageResponse(
		(
			<div
				style={{
					width: "100%",
					height: "100%",
					display: "flex",
					flexDirection: "column",
					justifyContent: "space-between",
					padding: 80,
					background:
						"radial-gradient(ellipse 80% 70% at 30% 30%, hsla(43,60%,50%,0.14) 0%, hsla(43,60%,50%,0.04) 40%, transparent 70%), linear-gradient(180deg, hsl(30,4%,6%) 0%, hsl(30,4%,4%) 100%)",
				}}
			>
				{/* Top: logo + eyebrow */}
				<div style={{display: "flex", alignItems: "center", gap: 24}}>
					{/* biome-ignore lint/performance/noImgElement: Satori only renders <img> */}
					<img src={logo} width={88} height={88} alt="" />
					<div
						style={{
							display: "flex",
							fontSize: 26,
							letterSpacing: 6,
							fontWeight: 600,
							textTransform: "uppercase",
							color: "hsl(43, 60%, 58%)",
						}}
					>
						Voicebox Blog
					</div>
				</div>

				{/* Title */}
				<div
					style={{
						display: "flex",
						fontSize: titleFontSize(title),
						lineHeight: 1.1,
						fontWeight: 700,
						letterSpacing: -1,
						color: "hsl(30, 10%, 94%)",
						maxWidth: 1000,
					}}
				>
					{title}
				</div>

				{/* Footer meta */}
				<div
					style={{
						display: "flex",
						fontSize: 28,
						color: "hsl(30, 5%, 55%)",
					}}
				>
					{meta}
				</div>
			</div>
		),
		size,
	);
}
