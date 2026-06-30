'use client';

import { Github } from 'lucide-react';
import { GITHUB_REPO } from '@/lib/constants';
import { DictationHero } from './CaptureSection';

export function CaptureHero({
  version,
  totalDownloads,
}: {
  version: string | null;
  totalDownloads: number | null;
}) {
  return (
    <section className="relative pt-32 pb-16">
      {/* Background glow */}
      <div className="hero-glow hero-glow-fade pointer-events-none absolute inset-0 -top-32">
        <div className="absolute left-1/2 top-0 -translate-x-1/2 w-[900px] h-[500px] rounded-full bg-accent/12 blur-[140px]" />
        <div className="absolute left-1/2 top-16 -translate-x-1/2 w-[520px] h-[360px] rounded-full bg-accent/8 blur-[80px]" />
      </div>

      <div className="relative mx-auto max-w-5xl px-6 text-center">
        {/* Kicker */}
        <div
          className="fade-in mb-6 text-[11px] font-semibold uppercase tracking-[0.22em] text-accent"
          style={{ animationDelay: '50ms' }}
        >
          Voice dictation · for humans and AI agents
        </div>

        {/* Headline */}
        <div className="fade-in relative" style={{ animationDelay: '100ms' }}>
          <h1 className="text-5xl font-bold tracking-tighter leading-[0.9] text-foreground md:text-7xl lg:text-[96px]">
            Just talk to your computer.
          </h1>
        </div>

        {/* Subtitle */}
        <p
          className="fade-in mx-auto mt-6 max-w-2xl text-lg text-muted-foreground md:text-xl"
          style={{ animationDelay: '200ms' }}
        >
          Hold a key anywhere on your machine, speak, release — your words land in the focused
          text field. A free, open-source, entirely-local alternative to{' '}
          <b className="text-white">WisprFlow</b>. And because Voicebox clones voices too, any
          AI agent can speak back in a voice you own.
        </p>

        {/* CTAs */}
        <div
          className="fade-in mt-10 flex flex-row items-center justify-center gap-3 sm:gap-4"
          style={{ animationDelay: '300ms' }}
        >
          <a
            href="/download"
            className="rounded-full bg-accent px-8 py-3.5 text-sm font-semibold uppercase tracking-wider text-white shadow-[0_4px_20px_hsl(43_60%_50%/0.3),inset_0_2px_0_rgba(255,255,255,0.2),inset_0_-2px_0_rgba(0,0,0,0.1)] transition-all hover:bg-accent-faint active:shadow-[0_2px_10px_hsl(43_60%_50%/0.3),inset_0_4px_8px_rgba(0,0,0,0.3)]"
          >
            Download
          </a>
          <a
            href={GITHUB_REPO}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-full border border-border/60 bg-card/40 backdrop-blur-sm px-6 py-3 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground hover:border-border"
          >
            <Github className="h-4 w-4" />
            View on GitHub
          </a>
        </div>

        {/* Version + downloads */}
        <p
          className="fade-in mt-4 text-xs text-muted-foreground/50"
          style={{ animationDelay: '400ms' }}
        >
          {version ?? ''}
          {version && totalDownloads != null ? ' · ' : ''}
          {totalDownloads != null ? `${totalDownloads.toLocaleString()} downloads` : ''}
          {version || totalDownloads != null ? ' · ' : ''}
          macOS, Windows, Linux
        </p>
      </div>

      {/* Hero visual — the pill itself */}
      <div className="mt-20 px-6">
        <DictationHero />
      </div>
    </section>
  );
}
