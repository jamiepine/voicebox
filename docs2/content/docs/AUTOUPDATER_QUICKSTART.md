---
title: "Autoupdater Quick Start"
description: "Quick guide to activate the Tauri v2 autoupdater"
---

The Tauri v2 autoupdater has been fully configured and integrated. Follow these steps to activate it.

## What's Already Done

✅ Rust plugin installed and initialized
✅ Tauri configuration set up with updater settings
✅ Permissions granted for update operations
✅ GitHub Actions workflow updated with signing support
✅ Frontend components created and integrated
✅ Update notifications on app startup
✅ Manual update check in Settings tab

## Required Steps (5 minutes)

### 1. Generate Signing Keys

```bash
bun run generate:keys
```

This creates:

- Private key: `~/.tauri/voicebox.key` (keep secret!)
- Public key: `~/.tauri/voicebox.key.pub` (safe to share)

### 2. Update Tauri Config

Open `tauri/src-tauri/tauri.conf.json` and:

1. Replace `"REPLACE_WITH_YOUR_PUBLIC_KEY"` with the content from `~/.tauri/voicebox.key.pub`
2. Update the endpoint URL with your GitHub username:
   ```json
   "endpoints": [
     "https://github.com/YOUR_USERNAME/voicebox/releases/latest/download/latest.json"
   ]
   ```

### 3. Add GitHub Secrets

Go to your repo Settings → Secrets and variables → Actions:

1. Add `TAURI_SIGNING_PRIVATE_KEY`:

   ```bash
   cat ~/.tauri/voicebox.key
   ```

   Copy the entire output and paste as the secret value

2. Add `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`:
   Leave empty (or add your password if you set one)

### 4. Test the Setup

To test locally before creating a release:

```bash
bun run build:release
```

This will verify your keys are set up correctly.

## How It Works

### For Users

1. App checks for updates on startup (only in Tauri builds)
2. If an update is available, a banner appears at the top
3. Users can click "Install Now" to download and install
4. App restarts automatically after installation

### For Developers

1. Create a new git tag: `git tag v0.2.0 && git push --tags`
2. GitHub Actions builds signed releases for all platforms
3. Uploads installers and generates `latest.json` manifest
4. Users running older versions will be notified automatically

## UI Components

### Update Notification Banner

- Shows at top of app when update is available
- Appears automatically on startup
- Displays download/install progress

### Settings Panel

- Located in Settings tab
- Shows current version
- Manual "Check for Updates" button
- Update status and progress

## Troubleshooting

**"Public key not configured"**

- Make sure you copied the entire content from `voicebox.key.pub`
- The key should start with `dW50cnVzdGVkIGNvbW1lbnQ6`

**"Failed to check for updates"**

- Endpoint URL might be incorrect
- No releases published yet (expected for first setup)

**Build fails with signing error**

- Check that GitHub secrets are set correctly
- Verify private key file exists at `~/.tauri/voicebox.key`

## Next Release Workflow

1. Update version in `tauri/src-tauri/tauri.conf.json`
2. Commit changes
3. Create and push tag: `git tag v0.2.0 && git push --tags`
4. GitHub Actions will automatically build and create a draft release
5. Review the release and publish it
6. Users will be notified of the update

## See Also

- Full documentation: `docs/AUTOUPDATER.md`
- Build script: `scripts/prepare-release.sh`
- GitHub workflow: `.github/workflows/release.yml`
