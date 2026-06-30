'use client';

import { Coffee, Coins, Github } from 'lucide-react';
import Image from 'next/image';
import { useEffect, useState } from 'react';
import { DONATE_URL, GITHUB_REPO, TOKEN_TICKER } from '@/lib/constants';

function formatStarCount(count: number): string {
  if (count >= 1000) {
    const k = count / 1000;
    return k % 1 === 0 ? `${k}k` : `${k.toFixed(1)}k`;
  }
  return count.toString();
}

export function Navbar() {
  const [starCount, setStarCount] = useState<number | null>(null);

  useEffect(() => {
    fetch('/api/stars')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch stars');
        return res.json();
      })
      .then((data) => {
        if (typeof data.count === 'number') setStarCount(data.count);
      })
      .catch((error) => {
        console.error('Failed to fetch star count:', error);
      });
  }, []);

  return (
    <nav className="fixed inset-x-0 top-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3 sm:grid sm:grid-cols-[1fr_auto_1fr] sm:gap-x-6">
        {/* Logo + wordmark */}
        <a href="/" className="flex items-center gap-2.5 justify-self-start">
          <Image
            src="/voicebox-logo-app.webp"
            alt="Voicebox"
            width={28}
            height={28}
            className="h-7 w-7"
          />
          <span className="text-[15px] font-semibold text-foreground">Voicebox</span>
        </a>

        {/* Nav links - centered */}
        <div className="hidden sm:flex items-center gap-1 justify-self-center">
          <a
            href="/#features"
            className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            Clone
          </a>
          <a
            href="/capture"
            className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            Capture
            <span className="rounded-full bg-accent/15 px-1.5 text-[9px] font-semibold uppercase tracking-wider text-accent">
              New
            </span>
          </a>
          <a
            href="/#mcp"
            className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            MCP
          </a>
          <a
            href="/#about"
            className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            Models
          </a>
          <a
            href="/pricing"
            className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            Pricing
          </a>
          <a
            href="/blog"
            className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            Blog
          </a>
          <a
            href="https://docs.voicebox.sh"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            Docs
          </a>
        </div>

        {/* Token + Donate + GitHub star buttons */}
        <div className="flex items-center gap-2 justify-self-end">
          <a
            href="/token"
            className="hidden sm:flex items-center gap-2 rounded-lg border border-border/60 bg-card/60 px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground hover:border-accent/40"
            aria-label={`${TOKEN_TICKER} token`}
          >
            <Coins className="h-4 w-4 text-accent" />
            <span className="text-[13px] font-semibold tracking-wide text-foreground">
              {TOKEN_TICKER}
            </span>
          </a>
          <a
            href={DONATE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-lg border border-border/60 bg-card/60 px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground hover:border-[#FFDD00]/40"
            aria-label="Donate via Buy Me a Coffee"
          >
            <Coffee className="h-4 w-4 text-[#FFDD00]" />
            <span className="text-[13px] font-medium">Donate</span>
          </a>
          <a
            href={GITHUB_REPO}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-lg border border-border/60 bg-card/60 px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground hover:border-border"
          >
            <Github className="h-4 w-4" />
            <span className="text-[13px] font-medium">Star</span>
            {starCount !== null && (
              <span className="border-l border-border/60 pl-2 text-[13px] font-semibold text-foreground">
                {formatStarCount(starCount)}
              </span>
            )}
          </a>
        </div>
      </div>
    </nav>
  );
}
