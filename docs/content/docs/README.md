---
title: "Documentation README"
description: "Voicebox documentation development guide"
---

This directory contains the documentation for Voicebox, built with [Fumadocs](https://fumadocs.dev).

## Development

### Running Locally

From the `docs/` directory:

```bash
bun install
bun run dev
```

The docs will be available at `http://localhost:3000`.

### Structure

- `content/docs/overview/` — user-facing guides (installation, quick start, feature walkthroughs)
- `content/docs/developer/` — architecture, backend internals, and contributor guides
- `content/docs/api-reference/` — auto-generated from the backend's OpenAPI schema
- `content/docs/index.mdx` — landing page
- `public/` — static assets (images, screenshots, videos)

### Writing Docs

- Use `.mdx` files for all documentation pages
- Navigation is generated from `content/docs/meta.json` files
- Fumadocs components available: `Callout`, `Cards` / `Card`, `Tabs` / `Tab`, `Steps` / `Step`, `Accordion` / `AccordionGroup`, `Files` / `Folder` / `File`
- API reference pages under `api-reference/` are regenerated from the backend's OpenAPI schema — don't edit them by hand

## Deployment

Docs are automatically deployed when changes land on `main`.
