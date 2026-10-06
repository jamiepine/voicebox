import { ArrowUpRight } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { capyUrl } from '@/lib/constants';

/**
 * Small "Funded by Capy" strip under the hero. Capy funds Voicebox and
 * Voicebox is built with Capy; the full story lives on /built-with-capy.
 */
export function FundedByCapy() {
  return (
    <section className="border-t border-border py-10">
      <div className="mx-auto max-w-4xl px-6">
        <div className="flex flex-col items-center justify-between gap-5 rounded-2xl border border-border bg-card/40 px-6 py-5 backdrop-blur-sm sm:flex-row">
          <a
            href={capyUrl('site-strip')}
            target="_blank"
            rel="noopener noreferrer"
            className="group flex items-center gap-4"
          >
            <Image
              src="/capy/capy-mark-dark.svg"
              alt="Capy"
              width={40}
              height={40}
              className="h-10 w-10 opacity-90 transition-opacity group-hover:opacity-100"
            />
            <div className="text-left">
              <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent">
                Funded by Capy
              </div>
              <p className="mt-1 text-sm text-muted-foreground">
                Capy funds Voicebox and builds it with us. Voicebox stays{' '}
                <b className="text-foreground">free and open source</b>.
              </p>
            </div>
          </a>
          <Link
            href="/built-with-capy"
            className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-border/60 bg-card/60 px-4 py-2 text-sm font-medium text-muted-foreground transition-colors hover:border-border hover:text-foreground"
          >
            How Voicebox is built
            <ArrowUpRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </div>
    </section>
  );
}
