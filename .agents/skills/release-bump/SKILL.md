---
name: release-bump
description: Use this skill to finalize a release. It stamps the [Unreleased] changelog section with a version and date, runs bumpversion to update all version files, and creates the release commit and tag. Only run this when you're ready to ship.
---

# Release Bump

## Goal

Finalize the changelog draft, bump the version across all tracked files, and create a tagged release commit. After this skill runs, the repo has a clean release commit and tag ready to push.

## Prerequisites

- `gh` CLI installed and authenticated (`gh auth status`).
- `bumpversion` installed (`pip install bumpversion` or available in the project venv).
- The `[Unreleased]` section of `CHANGELOG.md` should already contain the release narrative. If it's empty or stale, run the `draft-release-notes` skill first.

## Workflow

1. **Verify the working tree is clean** (except `CHANGELOG.md` which may have the draft).

   ```bash
   git status --porcelain
   ```

   Only `CHANGELOG.md` (and optionally `.agents/` files) should be modified. If there are other uncommitted changes, stop and ask the user to commit or stash them first.

2. **Determine the bump level.**

   Ask the user if not specified: `patch`, `minor`, or `major`. Check the current version:

   ```bash
   grep '^current_version' .bumpversion.cfg
   ```

3. **Stamp the changelog.**

   Read the current `[Unreleased]` content from `CHANGELOG.md`. Compute the new version (based on bump level and current version). Then:

   a. Replace the `## [Unreleased]` section body with an empty placeholder.
   b. Insert a new stamped section immediately after `## [Unreleased]`:

   ```markdown
   ## [Unreleased]

   ## [X.Y.Z] - YYYY-MM-DD

   <the content that was in [Unreleased]>
   ```

   c. Update the reference links at the bottom of the file:
   - Change the `[Unreleased]` link to compare against the new tag
   - Add a new link for the new version

   ```markdown
   [Unreleased]: https://github.com/jamiepine/voicebox/compare/vX.Y.Z...HEAD
   [X.Y.Z]: https://github.com/jamiepine/voicebox/compare/vPREVIOUS...vX.Y.Z
   ```

4. **Stage the changelog.**

   ```bash
   git add CHANGELOG.md
   ```

5. **Run bumpversion.**

   ```bash
   bumpversion --allow-dirty <patch|minor|major>
   ```

   The `--allow-dirty` flag is needed because `CHANGELOG.md` is already staged. bumpversion will:
   - Update version strings in all tracked files (see `.bumpversion.cfg`)
   - Create a commit with message `Bump version: X.Y.Z -> A.B.C`
   - Create a tag `vA.B.C`

   The staged `CHANGELOG.md` will be included in this commit automatically.

6. **Verify results.**

   ```bash
   git show --name-only --stat HEAD
   git tag --list "v*" --sort=-v:refname | head -n 5
   ```

   Confirm the commit contains:
   - `CHANGELOG.md`
   - `.bumpversion.cfg`
   - `tauri/src-tauri/tauri.conf.json`
   - `tauri/src-tauri/Cargo.toml`
   - `package.json`
   - `app/package.json`
   - `tauri/package.json`
   - `landing/package.json`
   - `web/package.json`
   - `backend/__init__.py`

   Confirm the new tag exists.

7. **Do NOT push** unless the user explicitly asks. Report the tag name and suggest:

   ```
   Ready to push. When you're ready:
     git push origin main --follow-tags
   ```

## Version Calculation Reference

Given current version `X.Y.Z`:
- `patch` -> `X.Y.(Z+1)`
- `minor` -> `X.(Y+1).0`
- `major` -> `(X+1).0.0`

## Error Recovery

- If bumpversion fails, the tag won't exist. Fix the issue and re-run — bumpversion is idempotent as long as the tag doesn't already exist.
- If you need to undo a release commit (before pushing): `git tag -d vX.Y.Z && git reset --soft HEAD~1`
- Never amend a release commit that has been pushed.

## Notes

- When the tag is pushed, the release CI (`.github/workflows/release.yml`) automatically extracts the matching version section from `CHANGELOG.md` and uses it as the GitHub Release body. No manual copy-paste needed.
- The release commit message is controlled by `.bumpversion.cfg` (`Bump version: X.Y.Z -> A.B.C`). Do not override it.
- If you need to manually update the GitHub Release body after the fact: `gh release edit vX.Y.Z --notes-file <(sed -n '/## \[X.Y.Z\]/,/## \[/p' CHANGELOG.md | head -n -1)`
