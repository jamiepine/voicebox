import { Coffee } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { DONATE_URL, GITHUB_REPO } from '@/lib/constants';

export function Footer() {
  return (
    <footer className="border-t border-border py-12">
      <div className="mx-auto max-w-7xl px-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-8 mb-10">
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
                <a href="#features" className="hover:text-foreground transition-colors">
                  Features
                </a>
              </li>
              <li>
                <a href="/download" className="hover:text-foreground transition-colors">
                  Download
                </a>
              </li>
              <li>
                <a href="#about" className="hover:text-foreground transition-colors">
                  About
                </a>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="text-sm font-semibold mb-3">Resources</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
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
