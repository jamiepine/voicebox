---
name: triage-prs
description: Use this skill to triage the open PR queue before a release. Classifies every open PR into must-merge, candidate, superseded, or deferred; writes a working triage doc; and runs the merge loop end-to-end. Designed for the pre-release "PR speedrun" pass where a solo maintainer wants to clear the inbound backlog in a single session.
---

# Triage PRs

## Goal

Turn a backlog of open PRs into a shipped set of merges in a single focused session. Produce a tracked, resumable plan (`<VERSION>_PR_TRIAGE.md`), then work it — rebasing where needed, merging in isolation-safe batches, applying post-merge follow-ups, and closing superseded or partially-applicable PRs with credit to their authors.

This skill pairs with `draft-release-notes` and `release-bump`: triage first, then draft notes against the new main, then cut the release.

## When to use

- Before a minor or major release when 10+ open PRs have accumulated
- When you want to unblock merging without losing the narrative of what's landing
- When you know you can't personally review every PR deeply, but need to land the critical subset fast

## Prerequisites

- `gh` CLI authenticated against the repo
- A dedicated worktree for PR review (avoid contaminating `main` with checkouts of contributor branches)
- Clarity on the target version — the triage doc is named after it (e.g. `0.4.0_PR_TRIAGE.md`)

## Workflow

### 1. Set up an isolated PR-review worktree

```bash
git worktree list  # check for stale ones first
git worktree prune
git worktree add ../voicebox-pr-review -b pr-review-<VERSION> main
```

Keep the main worktree for release-prep work (changelog drafts, direct-to-main follow-ups). Keep the review worktree for `gh pr checkout` — each checkout moves HEAD to a contributor branch, which you don't want to do in the main worktree.

### 2. Gather metadata for every open PR

```bash
gh pr list --state open --limit 50 --json \
  number,title,author,isDraft,mergeable,mergeStateStatus,files,additions,deletions,reviewDecision,statusCheckRollup,maintainerCanModify \
  --jq '.[] | {num: .number, title, author: .author.login, mergeable, state: .mergeStateStatus, canModify: .maintainerCanModify, changes: "+\(.additions)/-\(.deletions)", files: [.files[].path]}'
```

You want, for each PR:
- Size (`+additions/-deletions`)
- Mergeable state (`CLEAN`, `UNSTABLE`, `DIRTY` = conflicts, `UNKNOWN` = GitHub still computing)
- Whether maintainer edits are allowed on the branch (needed later if you rebase for the author)
- File paths touched (helps spot overlaps between PRs)

`UNKNOWN` is common right after a push to main — just try the merge and see.

### 3. Classify into tiers

Sort each PR into exactly one bucket:

**Tier 1 — Merge:** small, mergeable, fixes a real bug, clean CI, low review cost. One-liners, dependency relaxations, targeted safety hardening. These are the easy wins.

**Tier 2 — Candidate, review:** medium size (50-200 lines), touches more surface area, looks sound but needs a closer read. New user-facing features that fit the product direction.

**Supersede:** the fix or feature is already covered by something merged. Close with a comment pointing to the superseding PR. Check carefully — "similar title" isn't proof; compare the actual diffs.

**Defer to next release:** big features, dirty conflicts, draft PRs, anything touching the release pipeline in ways that would introduce risk. Don't merge these in a speedrun — they need dedicated focus.

### 4. Write the triage doc

Create `<VERSION>_PR_TRIAGE.md` in the PR-review worktree root. Structure:

```markdown
# <Repo> <VERSION> — PR Triage

Working doc for tracking which open PRs land in <VERSION>. Delete after release cut.

Last updated: <DATE>

## Progress

**Tier 1: 0 / N merged**
**Tier 2: 0 / M handled**
**Supersede triage: pending**

---

## Merge for <VERSION> — critical bug fixes

| PR | Status | Size | What it fixes | Why must-have |
|---|---|---|---|---|
| [#123](url) | [ ] | +5/-0 | ... | ... |

## Strong candidate — needs a quick review

| PR | Status | Size | Summary |
|---|---|---|---|

## Close as superseded

| PR | Status | Reason |
|---|---|---|

## Defer to <NEXT_VERSION>

- [#xxx](url) ... — reason

---

## Order of attack

1. Close superseded PRs (one-liner comments)
2. Merge tier-1 in dependency-free batches — check file paths don't overlap
3. Review tier-2 individually
4. Rerun `draft-release-notes` to pick up everything
5. Run `release-bump`
```

The **Progress** header is the most important part — it's your scoreboard and lets you resume cleanly if the session gets interrupted.

### 5. Work the loop — per PR

For each PR in the tier-1 / tier-2 list:

**a. Checkout in the review worktree:**
```bash
cd ../voicebox-pr-review
git checkout pr-review-<VERSION>  # reset to neutral base
gh pr checkout <N>
```

**b. Read the *actual* commit, not `main..HEAD`:**

```bash
git show HEAD                    # the PR's actual changes
git show --stat HEAD             # files touched + line counts
```

**Do NOT review via `git diff main..HEAD`** if the PR branch is older than main. That diff includes *every commit that landed on main after the PR was forked* as `-` (deletion) lines. A 3-line PR can look like a 700-line revert. This is the single easiest way to misjudge a PR.

**c. Evaluate concerns:** correctness, scope, interaction with already-merged work, version compatibility (e.g. can't use an API that requires a dependency version we don't yet pin).

**d. Rebase if the branch is behind main:**
```bash
git fetch origin main
git rebase origin/main
```

This is **essential** before squash-merging. GitHub's squash computes `diff(PR-head, merge-base)` — on a stale branch, that diff includes reverting every in-between commit. Rebasing moves the merge-base forward so the squash is clean.

**e. If maintainer edits are allowed, push the rebase back to the contributor's fork:**
```bash
git remote add <author> https://github.com/<author>/<repo>.git
git fetch <author> <branch>              # get their ref first
git push <author> HEAD:<branch> --force-with-lease
```

This keeps GitHub's PR UI in sync with the rebased state and makes the merge clean from the GitHub side.

**f. Merge:**
```bash
gh pr merge <N> --squash
```

**g. Update the triage doc** — flip the checkbox to `✅ merged <sha>` (use the short SHA from `gh pr view <N> --json mergeCommit --jq '.mergeCommit.oid[0:7]'`). Update the Progress header.

### 6. Batch tiny fixes

PRs with ≤5 line changes, clean CI, non-overlapping file paths, and obviously-correct intent (e.g. one-line dependency relax, env var add, import path fix) can be merged in a single loop without the review-per-PR ceremony:

```bash
for pr in 425 384 416 429; do
  echo "=== Merging PR $pr ==="
  gh pr merge $pr --squash
done
```

Verify afterward that each landed cleanly:
```bash
for pr in 425 384 416 429; do
  gh pr view $pr --json state,mergeCommit --jq "{pr: $pr, state, sha: .mergeCommit.oid[0:7]}"
done
```

### 7. Post-merge follow-ups

Sometimes a PR is worth merging despite a known minor issue (e.g. incomplete dtype map, stale sentinel cleanup). Don't block the merge; apply the follow-up as a normal branch + PR right after:

```bash
cd <main-worktree>
git pull --ff-only origin main
git checkout -b fix/<short-name>
# edit...
git commit -m "fix(<area>): <one-liner>"
git push -u origin fix/<short-name>
gh pr create --title "..." --body "Follow-up to #<N>. ..."
```

Record both SHAs in the triage doc (`✅ merged <pr-sha> + follow-up <pr>`).

**Direct-to-main exception:** only under an explicit, scoped policy (e.g. "release speedrun"). Don't default to it.

### 8. Supersede: close with a credit-pointing comment

```bash
gh pr close <N> --comment "Closing — superseded by merged #<M> which landed <brief description>. Thanks!"
```

Check the diffs first — "similar title" is not enough. If the PR is *partially* superseded (the diagnosis is right but only half the changes are still needed), do a partial-apply instead.

### 9. Partial-apply pattern

When a PR has both valuable and questionable changes bundled:

```bash
cd <main-worktree>
git pull --ff-only origin main

# Cherry-pick specific files from the PR branch
git checkout <pr-commit-sha> -- <file1> <file2>

# Review the staged changes, adjust as needed
git diff --cached

# Apply any surgical edits to files you don't want to bulk-replace
# (e.g. the PR's file predates a recent main commit you need to preserve)

# Commit with a trailer crediting the original author
git commit -m "$(cat <<'EOF'
<subject>

<body explaining what was kept vs dropped>

Co-Authored-By: <author> <noreply@github.com>
EOF
)"
git push ...  # branch + PR, unless under the direct-to-main exception
```

Then close the PR with a comment explaining what was applied and what was dropped, referencing the commit SHA.

### 10. Keep the doc current

Every merge, every close, every follow-up → update `<VERSION>_PR_TRIAGE.md`. The doc is your session log. If you're interrupted and resume tomorrow, the doc is the only source of truth for "where am I."

### 11. When triage is done

- Every PR in the doc has a terminal status (✅ merged / ✅ closed / deferred)
- Progress header shows N/N for each tier
- Next skill to run is `draft-release-notes` (to regenerate `[Unreleased]` against the new main), then `release-bump`

You can delete the triage doc after the release ships, or keep it in version history as a record.

## Gotchas

- **`main..HEAD` on a stale branch lies.** It shows everything main gained since the branch split as deletions. Always review via `git show HEAD` for the PR's actual commit.
- **Squash-merging an unrebased branch reverts in-between work.** The squash computes `diff(PR-head, merge-base)`. Rebase moves the merge-base forward.
- **`mergeable=UNKNOWN`** is transient — GitHub is recomputing after a push. Just try the merge.
- **Route ordering matters (FastAPI and similar):** `DELETE /history/failed` must be registered *before* `DELETE /history/{id}`, or the parameterized path will consume `"failed"` as an ID.
- **Apple's `-weak_framework` overrides `-framework`** for the same framework, regardless of order — use it via `cargo:rustc-link-arg=-Wl,-weak_framework,Name` when a dependency hard-links something optional.
- **Dependency version floors constrain what you can apply.** Before accepting a kwarg rename like `torch_dtype=` → `dtype=`, check the min-version pin supports it. Sometimes the right move is to cherry-pick half the PR.
- **`cpal::Stream` and similar `!Send` audio types** can't cross `await` points or `spawn_blocking`. Sometimes a "not-ideal but correct" sync wait is the best available fix; flag but don't block.
- **PyTorch nightly builds are not shippable for releases** — non-deterministic, can regress between runs. If a PR suggests switching to nightly to fix a GPU issue, prefer `TORCH_CUDA_ARCH_LIST=...+PTX` or wait for stable support instead.

## Canonical commands reference

```bash
# Bulk PR metadata
gh pr list --state open --limit 50 --json number,title,author,mergeable,mergeStateStatus,additions,deletions,maintainerCanModify,files

# Detailed single-PR view
gh pr view <N> --json body,author,headRefName,baseRefName,mergeable,maintainerCanModify,files,statusCheckRollup

# The actual commit, not the branch-vs-main diff
git show HEAD
git show --stat HEAD
gh pr diff <N>

# Rebase contributor branch onto current main
git fetch origin main && git rebase origin/main

# Push rebase back to contributor fork (maintainerCanModify=true required)
git remote add <author> https://github.com/<author>/<repo>.git
git fetch <author> <branch>
git push <author> HEAD:<branch> --force-with-lease

# Merge
gh pr merge <N> --squash

# Confirm merge SHA for triage doc
gh pr view <N> --json state,mergeCommit --jq '{state, sha: .mergeCommit.oid[0:7]}'

# Close superseded
gh pr close <N> --comment "Closing — superseded by merged #<M>. Thanks!"
```

## Notes

- **Never review a stale branch via `main..HEAD`.** This is the single most important line in this skill.
- **The triage doc is the session state.** Lose the doc, lose the session. Update it after every action.
- **Credit contributors even on partial-applies.** Use `Co-Authored-By:` trailers and close comments that link to the applied commit.
- **Don't let perfect be the enemy of shipped.** A fix that goes from "broken" to "works with a minor known issue" is a strict improvement. Flag the issue, file a follow-up, merge the fix.
