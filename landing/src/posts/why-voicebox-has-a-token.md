---
title: "Why Voicebox has a token"
author: Jamie Pine
date: 2026-06-27
tags: [Token, Transparency]
excerpt: "Voicebox grew to over a million downloads with zero marketing — but donations never made full-time work sustainable. Here's the honest reasoning behind $VOICEBOX, and the commitments that come with it."
---

Voicebox started as a one-day experiment. The Qwen3-TTS model dropped, I wanted to try it, so I built a small CLI to load voice profiles and generate speech. As a designer I already had the interface in my head — profiles as cards, a floating generation box, a player spanning the bottom of the app, a gold accent and a microphone logo to make it feel like a real studio. First working version in a day. Open sourced in three.

I did no marketing. None, to this day. But Reddit found it, creators started making tutorials, and the "ElevenLabs just lost its moat" posts began. It crossed a million downloads and is closing in on Spacedrive's GitHub star count — entirely organically. Along the way it became two things at once: a free alternative to ElevenLabs for voice cloning, and a free alternative to Wispr Flow for dictation.

So a fair question keeps coming up: if it's this successful, why a token? Why not just put it behind a subscription, or turn on GitHub Sponsors and call it a day?

## The honest version

I could have built this as a cloud app with a monthly subscription and probably done well. I chose open source instead, and I'd make the same call again. I don't think it would have grown like this behind a paywall — people trust it more when they can read the code and run it entirely on their own machine, it earns a ton of organic exposure for free, and I get the community's help making it work across the endless combinations of GPUs and operating systems I could never test alone.

But open source doesn't pay rent. The donation button on the site brings in roughly $200–300 a month. I didn't expect GitHub Sponsors to move that number much. The old sponsor program made almost nothing. I love this project and I want to work on it full-time — and for a while I couldn't justify it.

The token changed that overnight. It's the closest thing I've had to a salary in a long time, and it let me go full-time on Voicebox immediately. That's not a hypothetical — it's already happening, and you'll see it in the commits, the releases, and the posts on [@VoiceboxAI](https://x.com/VoiceboxAI).

## What the token is — and what it isn't

**$VOICEBOX is entirely optional.** Voicebox is, and always will be, free, open source, and local-first. Every feature works without ever touching the token. It exists for supporters who want to back the project and have some fun — nothing here is gated behind it, and nothing here is financial advice.

It's also the **only** token I will ever make. My other projects, including Spacedrive, will never have an official token. I thought carefully about this: the community was clearly most interested in Voicebox because of its existing traction, and if I'm going to have a token at all, I'd rather deploy it myself so I can lock liquidity and be accountable for its trajectory — rather than have an anonymous community coin I can't control. As of now I no longer claim fees on any other community tokens, either.

## Don't trust — verify

A token only earns trust through actions you can check on-chain. So:

- Liquidity and a portion of dev holdings were **locked at launch**, visible from day one.
- I've done **buyback and burns** — buying $VOICEBOX back from the market and burning it to a dead address, permanently. I've done this more than once, and I'll keep doing it.
- The plan is a balanced mix: lock more for trust, burn periodically, but keep enough flexibility to fund real expenses and add liquidity when it helps.

Every one of these is linked on the [token page](/token), so you never have to take my word for it. Verify the contract address there before you do anything — impersonators are common.

## What the money actually builds

Going full-time means the roadmap moves faster. The near-term priorities:

- **The mobile app.** I use the prototype every day for dictation on the go. It does both cloning and dictation, and it's nearly ready to ship.
- **Encrypted cloud backup & sync.** Generate on the go, keep your captures and generations safe, pick up on any device. This will be a paid service — around $12/year — and **free for token holders**.
- **More engines and broader hardware support.** More TTS engines, better GPU and OS coverage, so Voicebox runs great on whatever you've got.

That's the whole pitch. The app stays free forever, the token is an optional way to support it and unlock the cloud service, and the proof of good faith is on-chain and in the changelog. If you want to back the project, the [token page](/token) is here — but starring the repo and telling a friend helps just as much.

Thanks for being here. Now back to shipping.
