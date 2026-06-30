# macOS Notarization & Gatekeeper

**Status:** Diagnosis — Homebrew Cask CI rejects v0.4.5 on macOS 15 (Sequoia); fix pending
**Touches:** `.github/workflows/release.yml`, Tauri bundler config, sidecar signing
**Last reviewed:** 2026-04-24

## Context

Homebrew Cask PR [#260314](https://github.com/Homebrew/homebrew-cask/pull/260314) adds `brew install --cask voicebox`. CI is green on macOS 14 and macOS 26 (arm + intel) but fails on macOS 15 (arm + intel). The 0.4.3 release added DMG-level stapling to address this, and it didn't move CI — 0.4.5 still fails. A maintainer reproduced the failure in a fresh Sequoia VM.

This document is the working diagnosis plus the ordered fix plan.

## What the failing check actually does

The failing step is `brew audit --cask --online --signing --new voicebox`, not `brew install`. `brew install` succeeds end-to-end in CI (the log shows `Uninstalling Cask voicebox` after the install phase). The `--signing` audit:

1. Downloads the cask's `url`
2. Mounts the DMG
3. Runs `spctl --assess -t open --context context:primary-signature` against the `.app` inside

That policy tests the first-launch Gatekeeper path on the extracted bundle. It reads the `.app`'s own code signature and notarization ticket — the DMG wrapper is not involved. The staple added in 0.4.3 covers the DMG, so it has no effect on this audit.

## Why Sequoia and not Sonoma

`spctl -t open` on macOS 15 enforces checks that 14 tolerated:

- Secure timestamp required on hardened-runtime signatures. Untimestamped signatures pass on 14, fail on 15.
- Deep verification of nested Mach-Os. If any embedded `.dylib` or helper binary carries an ad-hoc signature (or a signature with a different Team ID), 15 rejects the whole bundle; 14 often accepted it.
- Hardened runtime must be set on every nested executable, not just the top-level app binary. Entitlements declared on the outer app do not propagate.

Local dev machines pass `spctl` because the first-party developer context and cached notarization tickets mask these failures. A fresh Sequoia VM with no prior trust state does not.

## Where the gap is likely to be

Voicebox ships PyInstaller sidecars declared in `tauri.conf.json` under `externalBin`:

- **0.4.x:** `voicebox-server` only (single `--onefile` Mach-O on macOS)
- **0.5.0+:** `voicebox-server` and `voicebox-mcp` (`voicebox-mcp` is new in 0.5.0)

Tauri's bundler signs each `externalBin` with the configured identity but does not apply `--options=runtime` or `--timestamp` automatically, and does not merge the outer app's entitlements into the sidecar signature. The outer `Voicebox` binary is correctly signed with hardened runtime + `disable-library-validation`; the sidecars likely are not.

Order of likelihood:

1. Sidecar `voicebox-server` lacks hardened runtime or a secure timestamp in its signature.
2. The sidecar inherits the identity but was signed before tauri-action's final notarization pass, so the notarization ticket doesn't actually cover it.
3. Something inside the sidecar's PyInstaller archive unpacks to a `.dylib` at runtime that Gatekeeper inspects during assessment.

The 0.5.0 fix must cover both sidecars.

## Diagnostic commands

Run against a freshly downloaded release DMG (not a dev build, and from a machine that has never opened the app before):

```
hdiutil attach Voicebox_0.4.5_aarch64.dmg
xcrun stapler validate "/Volumes/Voicebox 0.4.5/Voicebox.app"
spctl -a -vvv -t open --context context:primary-signature "/Volumes/Voicebox 0.4.5/Voicebox.app"
codesign --verify --deep --strict --verbose=2 "/Volumes/Voicebox 0.4.5/Voicebox.app"
codesign -dv --verbose=4 "/Volumes/Voicebox 0.4.5/Voicebox.app/Contents/MacOS/voicebox-server"
```

The last command is the tell — look for `flags=0x10000(runtime)` and a `Timestamp=` line. If either is missing, the sidecar is the failure.

`spctl -t install` (what 0.4.3 verified with) is a different policy and can pass while `-t open` fails — any future verification should use `-t open --context context:primary-signature` to match what Homebrew's audit runs.

## Phases

### Phase 1 — Confirm the failure mode

Pull the 0.4.5 DMG on a fresh Sequoia environment or a VM snapshot with no trust state. Run the diagnostic block above. Record the exact failing command and its CSSMERR / rejection reason. This disambiguates between the three hypotheses before we change the workflow.

### Phase 2 — Sign sidecars explicitly in the release workflow

Between tauri-action's build step and the DMG-notarization step already in `release.yml`, add a step that re-signs every `externalBin` present under `Voicebox.app/Contents/MacOS/` with:

- `--options=runtime` (hardened runtime)
- `--timestamp` (secure timestamp)
- `--entitlements` pointing at `Entitlements.plist` or a sidecar-specific subset
- The same `APPLE_SIGNING_IDENTITY` the outer app uses

Re-sign the outer `.app` afterward so its seal covers the updated nested signatures.

Covers `voicebox-server` on 0.4.x and both sidecars from 0.5.0 forward.

### Phase 3 — Re-notarize and staple the `.app`

After sidecars are re-signed the outer bundle's notarization ticket is stale. Submit the `.app` (zipped) to `notarytool`, wait, then `xcrun stapler staple Voicebox.app`. This puts the ticket directly on the `.app` so the `spctl -t open` audit passes without any online ticket lookup.

Then rebuild the DMG from the stapled `.app` and keep the existing DMG-level notarize/staple step — it still helps Finder drag-install.

### Phase 4 — CI verification gate in the release workflow

Before upload, run the same four diagnostic commands against the built artifact inside the workflow. If any fail, fail the release job rather than shipping a DMG that Homebrew (and Sequoia Finder users) will reject. This is the check that would have caught the 0.4.3 and 0.4.5 attempts before they cost PR review cycles.

### Phase 5 — Re-request Homebrew CI

Once a tagged release passes Phase 4 locally, push a cask update to #260314. Expect `test voicebox (macos-15, arm)` and `test voicebox (macos-15-intel, intel)` to go green.

## Open questions

- Does tauri-action v0.6 pass `APPLE_API_KEY_PATH` to the bundler's notarize path, or does it rely on the `~/.appstoreconnect/private_keys/AuthKey_*.p8` auto-discovery the staple step already sets up? If the former isn't working, tauri may be signing but never notarizing the `.app`, which would make the ticket absent entirely rather than stale. Worth a `grep -i notariz` on a full release job log.
- If Phase 2 resolves the macOS 15 failure, revisit whether the 0.4.3 DMG staple step is still needed. It's cheap to keep and helps the Finder-open case, so default to leaving it.
