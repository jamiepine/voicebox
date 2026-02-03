---
title: "Auto-Updater Documentation"
description: "How Voicebox automatic updates work for users and developers"
---

Voicebox includes automatic updates powered by Tauri's updater plugin. This document explains how it works for both users and developers.

## 1. Generate Signing Keys

Run this command to generate your signing keypair:

```bash
cd tauri && bun tauri signer generate -w ~/.tauri/voicebox.key
```

This creates:

- **Private key**: `~/.tauri/voicebox.key` (keep this secret!)
- **Public key**: `~/.tauri/voicebox.key.pub`

## 2. Update Configuration

Copy the content from `~/.tauri/voicebox.key.pub` and replace the placeholder in `tauri/src-tauri/tauri.conf.json`:

```json
{
	"plugins": {
		"updater": {
			"pubkey": "PASTE_PUBLIC_KEY_CONTENT_HERE",
			"endpoints": [
				"https://github.com/YOUR_USERNAME/voicebox/releases/latest/download/latest.json"
			]
		}
	}
}
```

Update the endpoint URL with your actual GitHub username/organization.

## 3. Building with Signatures

When building releases, set these environment variables:

**macOS/Linux:**

```bash
export TAURI_SIGNING_PRIVATE_KEY="$(cat ~/.tauri/voicebox.key)"
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD=""
bun run build
```

**Windows PowerShell:**

```powershell
$env:TAURI_SIGNING_PRIVATE_KEY = Get-Content ~/.tauri/voicebox.key -Raw
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = ""
bun run build
```

## 4. GitHub Release Setup

When you create a GitHub release, the build process will generate:

- Installers for each platform
- `.sig` signature files
- `latest.json` update manifest

### Manual Release Process

1. Build the app with signing keys set
2. Create a new GitHub release
3. Upload all files from `tauri/src-tauri/target/release/bundle/`
4. Create `latest.json` in your release assets:

```json
{
	"version": "0.2.0",
	"notes": "Bug fixes and improvements",
	"pub_date": "2026-01-25T12:00:00Z",
	"platforms": {
		"darwin-aarch64": {
			"signature": "CONTENT_FROM_.app.tar.gz.sig",
			"url": "https://github.com/YOUR_USERNAME/voicebox/releases/download/v0.2.0/voicebox_0.2.0_aarch64.dmg"
		},
		"darwin-x86_64": {
			"signature": "CONTENT_FROM_.app.tar.gz.sig",
			"url": "https://github.com/YOUR_USERNAME/voicebox/releases/download/v0.2.0/voicebox_0.2.0_x64.dmg"
		},
		"linux-x86_64": {
			"signature": "CONTENT_FROM_.AppImage.sig",
			"url": "https://github.com/YOUR_USERNAME/voicebox/releases/download/v0.2.0/voicebox_0.2.0_amd64.AppImage"
		},
		"windows-x86_64": {
			"signature": "CONTENT_FROM_.msi.sig",
			"url": "https://github.com/YOUR_USERNAME/voicebox/releases/download/v0.2.0/voicebox_0.2.0_x64_en-US.msi"
		}
	}
}
```

### Automated GitHub Actions (Recommended)

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

jobs:
  release:
    strategy:
      matrix:
        platform: [macos-latest, ubuntu-22.04, windows-latest]

    runs-on: ${{ matrix.platform }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Bun
        uses: oven-sh/setup-bun@v1

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install dependencies (Ubuntu)
        if: matrix.platform == 'ubuntu-22.04'
        run: |
          sudo apt-get update
          sudo apt-get install -y libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf

      - name: Install dependencies
        run: bun install

      - name: Build
        env:
          TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}
          TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY_PASSWORD }}
        run: bun run build

      - name: Upload Release
        uses: softprops/action-gh-release@v1
        with:
          files: tauri/src-tauri/target/release/bundle/**/*
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

Add your private key to GitHub secrets:

- Go to Settings → Secrets and variables → Actions
- Add `TAURI_SIGNING_PRIVATE_KEY` with the content of `~/.tauri/voicebox.key`
- Add `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` (empty string if no password)

## Frontend Integration

The frontend integration is complete with automatic update notifications and manual update checks:

- **Update Notification Banner** - Appears automatically when updates are available
- **Settings Panel** - Manual "Check for Updates" button in Settings tab
- **Update Hook** - React hook handles all update operations

See `docs/AUTOUPDATER_QUICKSTART.md` for a quick setup guide.

## Security Notes

- Never commit your private key to version control
- Store private keys securely (use GitHub secrets for CI/CD)
- The public key in `tauri.conf.json` is safe to commit
- Updates are cryptographically verified before installation
- HTTP endpoints are blocked by default (HTTPS only)

## Testing Updates

1. Build version 0.1.0 and install it
2. Update version in `tauri.conf.json` to 0.2.0
3. Build version 0.2.0 with signatures
4. Create a local server or GitHub release with `latest.json`
5. Run version 0.1.0 and trigger update check
6. Verify update downloads and installs correctly

## Troubleshooting

**"Invalid signature" error:**

- Verify public key matches the private key used to sign
- Ensure signature files (.sig) are uploaded correctly

**"No update available" when one exists:**

- Check endpoint URL is correct
- Verify `latest.json` format matches specification
- Ensure version in latest.json is higher than current version

**Build fails with signing:**

- Confirm environment variables are set correctly
- Check private key file exists and is readable
- Verify private key format (should start with `dW50cnVzdGVkIGNvbW1lbnQ6`)
