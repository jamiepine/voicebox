# Voicebox Cloud Roadmap

The post-mobile commercial trajectory. Captures the strategic arc beyond `mobile/PLAN.md` — what Voicebox becomes once the mobile companion ships and we start layering optional cloud services on top of the local-first base.

The desktop app stays free. Paid surface is the cloud layer, gated behind a Voicebox account, designed so the server sees as little as possible.

---

## Phases

### Phase 0 — Mobile companion (in progress)

See [`mobile/PLAN.md`](../../mobile/PLAN.md). Entirely local: paired-device keys live on the iPhone, traffic goes over Tailscale or LAN, no cloud account required. This is the wedge — it establishes the device-key primitive that every later phase reuses.

### Phase 1 — Backup & Sync (next big feature)

First introduction of a Voicebox cloud account. Server stores **only encrypted blobs**.

- **E2E encryption keyed off the device key** from the mobile pairing flow. Audio + transcript blobs are encrypted client-side before upload; the server never has the plaintext or the key.
- **Quota by number of generations**, not by storage GB. Avoids "how many GB do you offer" framing and keeps tiering legible. (Word-count quotas are an alternative — closer to the ElevenLabs model — but generations are simpler to communicate.)
- **What's synced:** captures (audio + transcripts), generations, voice profiles **as ciphertext**, settings.
- **What's NOT synced:** voice profile audio in plaintext, refinement LLM context, anything that would let us reconstruct what a user said or who they sound like.
- **Multi-device read:** the same paired-device key on a second device decrypts the backup. Recovery via printable key on first pairing.

The privacy framing is load-bearing. "We see encrypted blobs and that's it" is the commitment the rest of the cloud story rests on.

### Phase 2 — Private Voice Inference ("the OpenRouter for voice")

The big bet. Today there is no major neutral voice-inference provider — every cloud TTS service ships its own proprietary models. Open-source TTS models exist and keep getting better, but nobody runs them as a paid hosted catalog  at scale.

Voicebox already has the distribution. The thesis is: the same users who chose local-first specifically to avoid sending voice data to ElevenLabs will pay a fair markup to run open-source voices on hosted GPUs **when they don't have local hardware** (mobile-only users, low-end laptops, "I just don't want to manage CUDA"), provided the privacy story stays consistent.

- **Catalog-first positioning.** Cloud can offer more voices than the desktop binary bundles (the bundle is already 500MB without CUDA, ~3GB with — there's a hard ceiling on what we can ship locally). Catalog grows over time.
- **Pricing tiers (rough first cut):** $5 / $15 / $25 / month, plus Enterprise. Final numbers depend on benchmarking — see below.
- **Unit economics work to do:** benchmark every open-source TTS engine in the lineup (Qwen3-TTS, Chatterbox Multilingual + Turbo, TADA, Kokoro, LuxTTS, plus future additions) for cost-per-generation on candidate hardware. Find the engines where our markup is comfortably below ElevenLabs's per-character cost.
- **Privacy ceiling:** server-side inference cannot be cryptographically verified the way E2E backup can. The honest framing is "we don't log inputs, we don't train on your data, audited" — not "we mathematically can't see it." That's a real step down from Phase 1's guarantee, and the product has to be clear about it.
- **Mobile + OS integrations.** Once cloud inference exists, the mobile app unlocks the same OS-level surfaces ElevenLabs has (keyboard-tied dictation, share-sheet TTS, Siri-equivalent). Local-first users still get them via paired desktop; cloud users get them without needing a desktop at all.

#### Inference architecture

**Compute layer.** Modal as the v1 platform — per-second billing, scale-to-zero, volume mounts for model weights, runs our existing Python code with a thin decorator. The ~30-50% premium over raw GPU cost is irrelevant at launch scale and small (less than one DevOps hire) at $1M ARR. Migrate engine-by-engine to bare-metal (Lambda Labs, Crusoe, CoreWeave) once any single engine has predictable demand. Hyperscalers (AWS / GCP) only for enterprise contracts that require it.

**Topology.**

```
Client (desktop / mobile / API user)
   │
   ▼  HTTPS, bearer auth
Gateway  ←  R2: encrypted profile blobs
   │       ←  D1 / Postgres: users, billing, quotas, profile metadata
   ▼  internal RPC
Per-engine Modal apps (Kokoro / Chatterbox / TADA / Whisper / …)
```

**Gateway.** Cloudflare Workers + R2 + D1 for v1 — Workers handle auth and routing, R2 has no egress fees which matters when the payload is audio, D1 handles small relational state (users, quotas, profile metadata). Auth, billing, rate limits, profile resolution, engine routing, and log redaction all live in the gateway. Workers stay dumb: they receive a request with the profile envelope already in hand, run inference, stream audio back. The gateway is what makes engine migration painless — moving TADA to bare-metal later is a routing config change, not a client change.

**Model packaging.** Each `backend/backends/<engine>.py` class becomes a Modal `@app.cls` wrapper. Same inference code as desktop. Weights download on container build, live on a Modal Volume, get reused by warm containers. The PyInstaller-specific runtime hooks from 0.4.x (scipy / transformers / `torch._dynamo` workarounds for the frozen binary) factor into a `frozen.py` runtime hook the desktop build imports — cloud doesn't. Single source of truth for inference logic; two entry points for two runtimes.

**Streaming.** SSE over HTTPS, base64-encoded audio frames, interleaved status events (`queued` / `generating` / `done`), usage event at the end with `characters_consumed` and `seconds_generated`. Wire format identical to the desktop SSE pattern from 0.2.x — cloud is the same shape at a different URL.

**Latency budgets** (first audio chunk, warm / cold):

| Engine                             | Hardware  | Warm | Cold | Pool strategy                  |
| ---------------------------------- | --------- | ---- | ---- | ------------------------------ |
| Kokoro, LuxTTS                     | CPU       | <1s  | ~5s  | Scale-to-zero                  |
| Chatterbox Turbo, Whisper Turbo    | A10g / L4 | 1-3s | ~15s | Small warm pool, p95 sizing    |
| Qwen3-TTS, Chatterbox Multilingual | A10g / L4 | 2-5s | ~30s | Larger warm pool               |
| TADA-3B                            | A100      | ~5s  | ~60s | Premium tier only, capped pool |

Scale-to-zero where cold start fits the budget. Hot engines need warm pools sized to p95 demand — that's where unit economics get sensitive. Reserved capacity only after a quarterly demand baseline.

**Profile pipeline.** Cloned voice → encrypted blob with user-account-key → uploaded to R2 cold storage → fetched into worker memory at job start → decrypted in memory only, never written to worker disk → discarded on worker idle. TTL applies at the R2 layer (cold storage retention); worker hot-path retention is bounded by warmup window. Embedding-only caching where the engine exposes a stable embedding interface; raw-audio caching is the fallback. Per-engine audit needed before launch — see open questions.

**Hybrid routing on the client.** Desktop, mobile, and MCP clients already speak `127.0.0.1:17493`. Add `VOICEBOX_API_URL` + `VOICEBOX_API_KEY` plus a routing function:

```
if local_backend_reachable() and engine in local_engines:
   → 127.0.0.1:17493
else:
   → api.voicebox.sh/v1
```

Mobile-without-paired-desktop falls through to cloud automatically. Desktop without a usable GPU falls through for big engines, stays local for Kokoro. Same `voicebox.speak()` MCP call works either way. This is the differentiator versus ElevenLabs (cloud-only) and pure local-first competitors (no fallback).

#### Cloud-cached voice profiles

Inference latency makes it untenable to re-upload reference samples per call. Cloud caches the user's own voice profiles for the user's own inference, under tight guardrails:

- **Per-profile opt-in.** Profiles are local-only by default. A "Cloud-enabled" toggle (per-profile, never global, never automatic) is what triggers upload on first cloud generation. Mobile-without-paired-desktop is the main upgrade path here — without cached profiles, mobile cloud is preset-voices only.
- **User-controlled TTL.** `Session only` / `24h` / `7d` / `30d` / `Never expire`. Conservative default (24h). Auto-purge on inactivity regardless of ceiling.
- **Encrypted at rest under a user-account-key envelope.** Inference workers decrypt in memory only. Keys derived from the same identity primitive that backs Phase 1.
- **Cache embeddings, not raw audio, where the engine supports it.** Speaker embeddings (Chatterbox-style) are derived vectors — cache *those* instead of the .wav. Smaller blast radius, not reconstructible to original speech. Per-engine audit needed before launch (Qwen3-TTS, Chatterbox Multilingual + Turbo, TADA all do speaker conditioning differently); raw-audio caching is the fallback when the engine doesn't expose a stable embedding interface.
- **Consent attestation logged at upload.** "I have rights to this voice." Timestamped, retained. Doesn't shield from claims but it's the legal posture.
- **Verifiable deletion.** `DELETE /v2/profiles/{id}/cloud-cache` from day one, enforced across replicas, surfaced in-app as a one-click action.
- **Trust tier:** audited-no-log, encrypted at rest, user-controlled TTL — *not* the cryptographic guarantee Phase 1 backup carries. The product has to communicate this difference clearly so cached profiles don't bleed into the Phase 1 framing. The TTL control is the marketable differentiator versus ElevenLabs, which doesn't expose retention as a user lever at all.
- **Legal line items.** GDPR Article 9 (biometrics are special category), BIPA ($1k–5k statutory damages per violation), Texas CUBI, Washington MHMD. Real consent flow, retention controls, deletion rights, breach notification, signed DPAs for enterprise. SOC 2 + pen test before this surface goes public — not optional.

### Phase 3 — Voice Marketplace (much later)

A marketplace where voice owners license their cloned voices for others to use, with revenue sharing. Possibly: "rent out your AI voice."

This is the only phase that requires hosting voice profiles, and it requires real licensing infrastructure first — consent verification, takedown flow, identity claims, revenue accounting. Until that exists, **Voicebox does not host voice profiles in cloud at all** (see constraint below). Marketplace is the long-term endgame, not the next quarter.

---

## Cross-cutting constraints

### Voice profiles in cloud: owner-only, opt-in, time-bound

Three rules, in increasing strictness depending on phase:

- **Phase 1 (backup & sync):** profiles travel as ciphertext the server cannot decrypt. The server has no path to plaintext for any reason.
- **Phase 2 (inference):** the user's own profiles can be cached for the user's own inference, but only with per-profile opt-in, user-controlled TTL, encryption at rest, and verifiable deletion. The server holds plaintext (or derived embeddings) under audited-no-log terms — a real downshift from Phase 1's cryptographic guarantee, and one the product has to communicate honestly.
- **Phase 3 (marketplace):** hosting other users' voices for non-owners is gated on consent verification, licensing, takedown, and revenue accounting infrastructure. Until those exist, no profile is served to anyone but its owner. No shortcuts.

This protects two things at once:
- **Legal posture.** Biometric voice data triggers GDPR Article 9, BIPA, Texas CUBI, Washington MHMD. The trust hierarchy above maps to the consent and retention story we can defend at each phase.
- **Privacy positioning.** Phase 1 is "cryptographically can't see." Phase 2 is "audited won't see, with a timer you control." Both are honest, both sit above ElevenLabs's posture, and both have to be communicated as distinct trust tiers — not blurred together.

### Privacy is the moat, not a feature

The "private LLM users → ElevenLabs voice" workflow is incoherent: people pay to keep their text private and then hand their speech to a cloud vendor that trains on it. Voicebox is the consistent answer for that audience. Every cloud feature should be designed so a privacy-conscious user can adopt it without breaking that internal consistency — which is why Phase 1 is fully E2E and Phase 2 is "audited no-log" rather than "we have your audio but trust us."

### Revenue stack is multi-source

Subscriptions are not the only line. The full picture:

- **Subscriptions** — Phase 1 quotas + Phase 2 inference
- **Corporate sponsorship** — `landing/src/app/sponsors/page.tsx`, $500/mo tier live in 0.5
- **Individual donations** — Buy Me a Coffee
- **Marketplace revenue share** — Phase 3, far off

Diversification matters because the desktop app stays free forever. Subscriptions never have to carry the whole product.

---

## Sequencing & "ease it onto them"

The deliberate ordering is privacy-additive: each phase introduces the next layer of cloud only after the user has had time to trust the previous one.

1. **Mobile (entirely local)** — no account, no cloud, just a companion to the desktop you already trust.
2. **Backup & sync (cloud, fully E2E)** — first cloud account. Server sees nothing. Trust is bootstrapped on "we built the math so we can't see your data even if we wanted to."
3. **Private inference (cloud, audited no-log)** — second cloud surface. Honest about the ceiling: server-side inference can't carry the same cryptographic guarantee, but the operational commitment is no logs, no training, audited.
4. **Marketplace (cloud, profiles hosted with consent)** — only after licensing infra. The most invasive surface, gated behind real verification.

Skipping ahead breaks the trust ladder. Don't ship marketplace before backup & sync is mature; don't ship hosted inference before users are comfortable holding accounts at all.

---

## Open questions

1. **Quota unit.** Generations vs. words vs. characters. Generations is the cleanest to communicate; words/characters maps onto how ElevenLabs prices and might be required for inference billing. Could be different units per phase (generations for backup, characters for inference).
2. **Recovery key UX.** First pairing in Phase 1 needs to print a recovery key. How prominent? Force-display vs. hide-behind-link?
3. **Inference billing model.** Per-character (ElevenLabs-style), per-generation (simpler), per-second-of-output (closest to GPU cost). Pick before pricing tiers are finalized.
4. **Bring-your-own-key for inference?** Some privacy-conscious users may prefer to provide their own GPU credits / API keys to a third-party host through us. Worth considering for Enterprise.
5. **Marketplace consent verification.** What's the bar? Notarized release? Real-time liveness check? Out of scope for Phase 1-2 but informs how the device key is structured today.
6. **Default cloud-cache TTL.** 24h is the proposed conservative default. Worth A/B testing against `Session only` for first-time users — the "auto-purge after this session" framing might be a stronger trust signal than any number.
7. **Embedding vs. raw-audio caching, per engine.** Chatterbox produces stable speaker embeddings; Qwen3-TTS, TADA, and others use different conditioning strategies. Audit needed before launch — embedding-only caching shrinks the legal/privacy surface meaningfully, but only where the engine exposes a clean embedding interface.
8. **Single gateway region or multi-region?** Cloudflare is global by default, but Modal apps are primarily us-east / us-west. EU users hitting US compute = +100ms first-token latency, and GDPR pushes toward EU compute regardless. v1 single-region or hold launch for EU?
9. **SSE vs WebSocket for streaming.** SSE works through any proxy and is what desktop already uses, so the wire format is shared for free. WebSocket is bidirectional and unlocks "interrupt mid-generation" and live duplex features later. Default: SSE for v1, WS as a follow-on.
10. **Cloud Whisper in the v1 bundle?** Phase 2 was framed as TTS-only ("OpenRouter for voice"), but mobile dictation hitting cloud Whisper instead of a paired desktop is the obvious mobile-only feature. Same launch bundle, or hold for Phase 2.5?
11. **Billing integration.** Stripe Metered + customer portal (~2 weeks of work, 2.9% fee) vs self-hosted (saves the fee, adds significant ongoing work). Default: Stripe.

---

## How this connects to mobile V1

The encryption story starts with the device key minted during mobile pairing (`mobile/PLAN.md` → "Pairing & transport"). That same key — or a key derived from it — is what encrypts cloud blobs in Phase 1. Don't treat the mobile pairing key as a one-off; design it as the root of the user's lifetime encryption identity, with rotation + multi-device-add flows in mind even if those don't ship until Phase 1.
