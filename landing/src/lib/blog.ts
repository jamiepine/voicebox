import {readdirSync, readFileSync} from "node:fs";
import {join} from "node:path";
import matter from "gray-matter";
import {marked} from "marked";

// Posts are authored as markdown under src/posts. This module uses the Node fs
// API and must only be imported from server components / server code — never
// from a "use client" file, or the markdown libraries leak into the bundle.

export type BlogPost = {
	slug: string;
	title: string;
	author: string;
	date: string;
	tags: string[];
	excerpt: string;
	readingMinutes: number;
	html: string;
};

export type BlogPostSummary = Omit<BlogPost, "html">;

const POSTS_DIR = join(process.cwd(), "src/posts");

function slugFromFile(file: string): string {
	return file.replace(/\.md$/, "");
}

function parsePost(file: string, raw: string): BlogPost {
	const {data, content} = matter(raw);
	const words = content.trim().split(/\s+/).filter(Boolean).length;
	return {
		slug: slugFromFile(file),
		title: String(data.title ?? "Untitled"),
		author: String(data.author ?? "Jamie Pine"),
		date:
			data.date instanceof Date
				? data.date.toISOString().slice(0, 10)
				: String(data.date ?? ""),
		tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
		excerpt: String(data.excerpt ?? ""),
		readingMinutes: Math.max(1, Math.ceil(words / 220)),
		html: marked.parse(content, {async: false}) as string,
	};
}

export function loadAllPosts(): BlogPost[] {
	let files: string[] = [];
	try {
		files = readdirSync(POSTS_DIR).filter((f) => f.endsWith(".md"));
	} catch {
		return []; // posts dir doesn't exist yet — treat as empty
	}
	return files
		.map((file) => parsePost(file, readFileSync(join(POSTS_DIR, file), "utf8")))
		.sort((a, b) => (a.date < b.date ? 1 : -1));
}

export function listPosts(): BlogPostSummary[] {
	return loadAllPosts().map(({html: _html, ...summary}) => summary);
}

export function getPost(slug: string): BlogPost | null {
	return loadAllPosts().find((post) => post.slug === slug) ?? null;
}

export function formatDate(date: string): string {
	if (!date) return "";
	const d = new Date(`${date}T00:00:00Z`);
	if (Number.isNaN(d.getTime())) return date;
	return d.toLocaleDateString("en-US", {
		year: "numeric",
		month: "long",
		day: "numeric",
		timeZone: "UTC",
	});
}
