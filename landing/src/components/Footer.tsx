import { ArrowUpRight, Coffee, Coins } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { CopyAddress } from '@/components/CopyAddress';
import {
  DONATE_URL,
  GITHUB_REPO,
  TOKEN_CONTRACT_ADDRESS,
  TOKEN_TICKER,
} from '@/lib/constants';

export function Footer() {
  return (
    <footer className="border-t border-border py-12">
      <div className="mx-auto max-w-7xl px-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-8 mb-10">
          {/* Brand */}
          <div className="md:col-span-1">
            <div className="flex items-center gap-2.5 mb-4">
              <Image
                src="/voicebox-logo-app.webp"
                alt="Voicebox"
                width={24}
                height={24}
                className="h-6 w-6"
              />
              <span className="text-sm font-semibold">Voicebox</span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed mb-4">
              Open source voice cloning studio. Local-first, free forever.
            </p>
            <a
              href={DONATE_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-lg border border-border/60 bg-card/60 px-3 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground hover:border-[#FFDD00]/40"
              aria-label="Donate via Buy Me a Coffee"
            >
              <Coffee className="h-4 w-4 text-[#FFDD00]" />
              <span className="text-[13px] font-medium">Donate</span>
            </a>
          </div>

          {/* Product */}
          <div>
            <h4 className="text-sm font-semibold mb-3">Product</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>
                <a href="/#features" className="hover:text-foreground transition-colors">
                  Clone
                </a>
              </li>
              <li>
                <a href="/capture" className="hover:text-foreground transition-colors">
                  Capture
                </a>
              </li>
              <li>
                <a href="/#mcp" className="hover:text-foreground transition-colors">
                  MCP
                </a>
              </li>
              <li>
                <a href="/#about" className="hover:text-foreground transition-colors">
                  Models
                </a>
              </li>
              <li>
                <a href="/#api" className="hover:text-foreground transition-colors">
                  API
                </a>
              </li>
              <li>
                <a href="/cloud" className="hover:text-foreground transition-colors">
                  Cloud
                </a>
              </li>
              <li>
                <a href="/pricing" className="hover:text-foreground transition-colors">
                  Pricing
                </a>
              </li>
              <li>
                <a href="/download" className="hover:text-foreground transition-colors">
                  Download
                </a>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="text-sm font-semibold mb-3">Resources</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>
                <a href="/blog" className="hover:text-foreground transition-colors">
                  Blog
                </a>
              </li>
              <li>
                <Link
                  href="https://docs.voicebox.sh"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors"
                >
                  Documentation
                </Link>
              </li>
              <li>
                <Link
                  href={GITHUB_REPO}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors"
                >
                  Source Code
                </Link>
              </li>
              <li>
                <Link
                  href={`${GITHUB_REPO}/releases`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors"
                >
                  Releases
                </Link>
              </li>
              <li>
                <Link
                  href={`${GITHUB_REPO}/issues`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors"
                >
                  Issues
                </Link>
              </li>
              <li>
                <a href="/sponsors" className="hover:text-foreground transition-colors">
                  VIP Sponsor
                </a>
              </li>
            </ul>
          </div>

          {/* Also by */}
          <div>
            <h4 className="text-sm font-semibold mb-3">Also By</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>
                <a
                  href="https://spacebot.sh"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors"
                >
                  Spacebot
                </a>
              </li>
              <li>
                <a
                  href="https://spacedrive.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors"
                >
                  Spacedrive
                </a>
              </li>
            </ul>
          </div>

          {/* Token */}
          <div>
            <h4 className="text-sm font-semibold mb-3">Token</h4>
            <div className="space-y-3 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <Coins className="h-4 w-4 text-accent" />
                <span className="font-semibold text-foreground">{TOKEN_TICKER}</span>
                <span className="text-xs text-muted-foreground/60">Solana</span>
              </div>
              <CopyAddress address={TOKEN_CONTRACT_ADDRESS} />
              <Link
                href="/token"
                className="inline-flex items-center gap-1.5 hover:text-foreground transition-colors"
              >
                Token details
                <ArrowUpRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>
        </div>

        <div className="border-t border-border pt-6">
          <p className="text-center text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} Voicebox. Open source under MIT license.
          </p>
        </div>
      </div>
    </footer>
  );
}
