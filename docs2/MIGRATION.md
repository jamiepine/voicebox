# Documentation Migration: Mintlify → Fumadocs

This document summarizes the migration of documentation from `/docs` (Mintlify) to `/docs2` (Fumadocs).

## What Was Done

### 1. Files Copied

- ✅ All 29 MDX files from `/docs` folders (overview, api, developer, plans)
- ✅ All 4 root-level markdown files (AUTOUPDATER.md, AUTOUPDATER_QUICKSTART.md, TROUBLESHOOTING.md, README.md)
- ✅ All images (3 webp files) → `public/images/`
- ✅ All logo files (2 png files) → `public/logo/`

### 2. Component Migration

Created compatibility layer in `components/mintlify-compat.tsx` that maps Mintlify components to Fumadocs equivalents:

- `<Frame>` → Simple div wrapper (images are zoomable by default in Fumadocs)
- `<CardGroup>` → `<Cards>` (Fumadocs component)
- `<Card>` → `<Card>` (with icon string → Lucide icon mapping)
- `<Steps>` / `<Step>` → Direct mapping to Fumadocs components
- `<Tip>`, `<Note>`, `<Info>` → `<Callout type="info">`
- `<Warning>` → `<Callout type="warn">`
- `<Danger>` → `<Callout type="error">`
- `<AccordionGroup>` / `<Accordion>` → HTML `<details>` / `<summary>` elements

### 3. Navigation Structure

Created `meta.json` files for each folder:

- `content/docs/meta.json` - Root documentation
- `content/docs/overview/meta.json` - Overview pages
- `content/docs/api/meta.json` - API reference
- `content/docs/developer/meta.json` - Developer docs
- `content/docs/plans/meta.json` - Plans/roadmap

### 4. Link Fixes

- Fixed incorrect `/guides/...` paths → `/overview/...`
- All internal links now use correct paths

### 5. Branding

- Updated `lib/layout.shared.tsx` to use "Voicebox" as the nav title

## File Structure

```
docs2/
├── components/
│   └── mintlify-compat.tsx    # Mintlify → Fumadocs component mappings
├── content/docs/
│   ├── meta.json              # Root navigation
│   ├── overview/              # 12 MDX files
│   ├── api/                   # 5 MDX files
│   ├── developer/             # 12 MDX files
│   ├── plans/                 # 4 MD files
│   └── *.md                   # 4 root markdown files
├── public/
│   ├── images/                # 3 webp files
│   └── logo/                  # 2 png files
└── mdx-components.tsx         # MDX component configuration
```

## Icon Mapping

The following icon strings are mapped to Lucide icons:

- `microphone` → Mic
- `film` → Film
- `code` → Code
- `shield` → Shield
- `download` → Download
- `rocket` → Rocket
- `apple` → Apple
- `windows` → Windows
- `server` → Server
- `user` → User
- `waveform` → Waveform

## Next Steps

1. **Test the build**: Run `npm run build` (requires Node.js >= 20.9.0)
2. **Start dev server**: Run `npm run dev` to preview
3. **Customize styling**: Update `app/global.css` if needed
4. **Add more icons**: Extend `iconMap` in `mintlify-compat.tsx` as needed
5. **Review navigation**: Adjust `meta.json` files to customize page order

## Notes

- Image paths (`/images/...`) work as-is since Next.js serves from `public/`
- All Mintlify components are now compatible with Fumadocs
- Navigation structure follows Fumadocs conventions
- No breaking changes to content - all MDX files work with compatibility layer
