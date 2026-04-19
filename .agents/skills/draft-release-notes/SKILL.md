---
name: draft-release-notes
description: Use this skill to draft or update the [Unreleased] section of CHANGELOG.md from the actual changes since the last tag. Run this at any point during development to keep a working copy of the release narrative. Does NOT bump versions or create tags.
---

# Draft Release Notes

## Goal

Update the `[Unreleased]` section at the top of `CHANGELOG.md` with a narrative release story based on the real changes since the last tag. This is a **non-destructive working copy** — run it as many times as you want during development.

## Workflow

1. **Identify the last release tag and gather changes.**

   ```bash
   LAST_TAG=$(git tag --list "v*" --sort=-v:refname | head -n 1)
   echo "Last tag: $LAST_TAG"
   ```

   Then collect raw material from three sources:

   a. **Commit log since last tag:**
   ```bash
   git log --oneline "$LAST_TAG"..HEAD
   ```

   b. **GitHub-generated release notes preview** (PR titles, new contributors):
   ```bash
   gh api repos/:owner/:repo/releases/generate-notes \
     -f tag_name="vNEXT" \
     -f target_commitish="$(git rev-parse HEAD)" \
     -f previous_tag_name="$LAST_TAG" \
     --jq '.body'
   ```

   c. **Diff stat for theme analysis:**
   ```bash
   git diff --stat "$LAST_TAG"..HEAD
   ```

2. **Draft the release narrative.**

   Write markdown for the `[Unreleased]` section following the format below. Do not include the `## [Unreleased]` heading itself — just the body content.

3. **Update CHANGELOG.md.**

   Replace everything between `## [Unreleased]` and the next `## [` heading with the new draft. Preserve the HTML comment header and all existing release sections below.

   The `[Unreleased]` section must always exist and always be the first section after the header comments.

4. **Do NOT commit, tag, or bump versions.** Just leave the file modified in the working tree.

## Release Story Format

Structure the `[Unreleased]` section like this:

```markdown
## [Unreleased]

<One strong opening paragraph: what this release is about and why it matters.
Tie it to concrete shipped changes. No vague hype.>

<One paragraph on major technical shifts, if applicable.>

### <Feature/Theme Group>
- Bullet points with specifics
- Reference PRs where available: ([#123](https://github.com/jamiepine/voicebox/pull/123))

### <Another Group>
- ...

### Bug Fixes
- ...
```

### Style Guidelines

- **Factual and specific.** Every claim should trace to a real commit or PR.
- **Narrative over list.** Lead with paragraphs that tell the story, then support with bullets.
- **Group by theme, not by commit.** Cluster related changes under descriptive headings.
- **Reference PRs** where they exist, but don't fabricate them.
- **Skip trivial chores** (typo fixes, CI tweaks) unless they're the bulk of the release.
- **Match the voice of existing releases** — look at the v0.2.1 and v0.2.3 entries in CHANGELOG.md for tone reference.

## When There Are No Changes

If `git log "$LAST_TAG"..HEAD` is empty, leave the `[Unreleased]` section empty (just the heading) and tell the user there's nothing to draft.

## Notes

- This skill only touches the `[Unreleased]` section. It never modifies stamped release sections.
- The agent can be asked to run this skill at any point — mid-feature, before a PR, or right before cutting a release.
- The `release-bump` skill depends on this draft being up to date before it finalizes.
