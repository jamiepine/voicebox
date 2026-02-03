---
title: "Documentation README"
description: "Voicebox documentation development guide"
---

This directory contains the documentation for Voicebox, built with [Fumadocs](https://fumadocs.dev).

## Development

### Prerequisites

Install Mintlify globally using bun:

```bash
bun add -g mintlify
```

Or use the helper script:

```bash
bun run install:mintlify
```

### Running Locally

```bash
bun run dev
```

This will start the Mintlify dev server.

The docs will be available at `http://localhost:3000`

### Structure

```
docs/
├── mint.json           # Mintlify configuration
├── custom.css          # Custom styles
├── overview/           # Getting started & feature docs
├── guides/             # User guides
├── api/                # API reference
├── development/        # Developer documentation
├── logo/               # Logo assets
└── public/             # Static assets
```

### Writing Docs

- Use `.mdx` files for all documentation pages
- Follow the existing structure in `mint.json` for navigation
- Use Mintlify components for enhanced formatting (Card, CardGroup, Accordion, etc.)
- Reference the [Mintlify documentation](https://mintlify.com/docs) for available components

## Deployment

Docs are automatically deployed when changes are pushed to the main branch.

To manually deploy:

```bash
mintlify deploy
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.
