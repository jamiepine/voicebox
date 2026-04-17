Improvements suggested by @sharkello

Summary
-------
Voicebox is already polished; small additions will increase try‑rate and contributions.

Top items
---------
- Add a small "Try in 5 minutes" section with `just setup` + `just dev` command near top.
- Provide a Docker / CPU fallback example for Linux users.
- Add `examples/` with 3 short workflows: clone voice, generate short sample, export audio.
- Add CONTRIBUTING.md and 3 `good first issue` tasks (docs, example, small bugfix).
- Add quick video/GIF to top of README showing a generation.

Suggested README snippet
-----------------------
```markdown
# Voicebox — local voice cloning studio
Clone voices and generate speech locally. Try in 3 commands:
1. `just setup`
2. `just dev`
3. Open http://localhost:17493 and click "Generate"
```

Good first issues to add
-----------------------
- "Add examples/simple-generate.md" — `good first issue`, `examples`.
- "Add Docker instructions for Linux" — `good first issue`, `docs`.
- "Add CI badge for build/tests" — `good first issue`, `ci`.

Notes
-----
Committed as a branch locally. I can prepare PRs or patches; pushing would require creating forks/PRs on GitHub.
