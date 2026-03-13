# Voicebox Issue Pain Points (Snapshot)

## Scope

- Dataset: **128 total issues** (**107 open**, **21 closed**)
- Source: GitHub issues in `jamiepine/voicebox`
- Classification: keyword/theme clustering
- Note: counts below are **non-exclusive** (one issue can belong to multiple pain points)

## Most Common Pain Points (Open Issues)

| Rank | Pain Point | Open Issues | What users are reporting |
|---|---|---:|---|
| 1 | Model download & offline reliability | **32** | Downloads failing/stalling, cache/offline behavior inconsistent, wrong model size selected, Errno issues |
| 2 | GPU/backend compatibility | **22** | GPU not detected, backend fallback surprises, platform-specific runtime failures (Windows/Mac) |
| 3 | Export/save/file persistence | **15** | Export fails, "failed to fetch/download audio", samples/profiles not saving |
| 4 | Language/accent quality & coverage | **14** | Missing language support, accent mismatch, robotic outputs |
| 5 | Update/restart safety + long-op controls | **4** | Auto-restart without warning, update confusion, lack of cancel/pause controls |

## Representative Issues by Pain Point

### 1) Model download & offline reliability (32)

- [#159](https://github.com/jamiepine/voicebox/issues/159) - Qwen download fails with Errno 22
- [#151](https://github.com/jamiepine/voicebox/issues/151) - Model loading hangs / server crashes
- [#150](https://github.com/jamiepine/voicebox/issues/150) - Internet required despite downloaded models
- [#149](https://github.com/jamiepine/voicebox/issues/149) - Cancel/pause controls for large downloads
- [#96](https://github.com/jamiepine/voicebox/issues/96) - 0.6B selection still uses/downloads 1.7B

### 2) GPU/backend compatibility (22)

- [#164](https://github.com/jamiepine/voicebox/issues/164) - Windows: no GPU usage + multiple breakages
- [#141](https://github.com/jamiepine/voicebox/issues/141) - Using CPU only, GPU not used
- [#131](https://github.com/jamiepine/voicebox/issues/131) - Numpy ABI mismatch in bundled app
- [#130](https://github.com/jamiepine/voicebox/issues/130) - Intel Mac tensor/padding generation error
- [#127](https://github.com/jamiepine/voicebox/issues/127) - GPU not found

### 3) Export/save/file persistence (15)

- [#148](https://github.com/jamiepine/voicebox/issues/148) - Japanese export fails on 0.1.12
- [#143](https://github.com/jamiepine/voicebox/issues/143) - Samples not saving
- [#134](https://github.com/jamiepine/voicebox/issues/134) - Can't save profile
- [#105](https://github.com/jamiepine/voicebox/issues/105) - Export audio fails (failed to fetch)
- [#49](https://github.com/jamiepine/voicebox/issues/49) - Export filename/location ignored on Windows

### 4) Language/accent quality & coverage (14)

- [#162](https://github.com/jamiepine/voicebox/issues/162) - Persian audio request/problem
- [#117](https://github.com/jamiepine/voicebox/issues/117) - Arabic language support
- [#113](https://github.com/jamiepine/voicebox/issues/113) - Polish language support
- [#109](https://github.com/jamiepine/voicebox/issues/109) - Ukrainian support
- [#100](https://github.com/jamiepine/voicebox/issues/100) - Non-US accent quality issues

### 5) Update/restart safety + controls (4)

- [#164](https://github.com/jamiepine/voicebox/issues/164) - Update behavior + usability failures
- [#136](https://github.com/jamiepine/voicebox/issues/136) - Auto-restart without warning
- [#86](https://github.com/jamiepine/voicebox/issues/86) - Unexpected restart with no confirmation
- [#149](https://github.com/jamiepine/voicebox/issues/149) - Need pause/cancel and pre-download confirmation

## Additional Signal

- There is also a large **feature-request/misc** bucket (**36 open**) that is competing with stability triage (audiobook, Linux build, additional ASR/TTS models, integrations).

## Takeaway

Most user pain is concentrated in four stability areas: **download/offline path**, **GPU/backend detection**, **save/export reliability**, and **language/accent correctness**. Addressing those first should reduce the majority of current support friction.
