import { ArrowRight, Heart } from 'lucide-react';
import { SPONSORS, type Sponsor } from '@/lib/sponsors';

export function SponsorPromo() {
  if (SPONSORS.length === 0) {
    return <SponsorPromoEmpty />;
  }
  return <SponsorStrip sponsors={SPONSORS} />;
}

function SponsorPromoEmpty() {
  return (
    <section className="border-t border-border py-16">
      <div className="mx-auto max-w-5xl px-6">
        <div className="rounded-2xl border-2 border-accent/40 bg-gradient-to-br from-card/80 to-card/40 backdrop-blur-sm p-8 md:p-10 shadow-[0_8px_40px_hsl(43_60%_50%/0.08)]">
          <div className="grid md:grid-cols-[1fr_auto] items-center gap-8">
            <div>
              <div className="inline-flex items-center gap-2 mb-4 text-[11px] font-semibold uppercase tracking-[0.22em] text-accent">
                <Heart className="h-3.5 w-3.5" />
                Sponsor Voicebox
              </div>
              <h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground mb-3">
                Get your logo in front of 170k+ monthly visitors.
              </h3>
              <p className="text-sm md:text-base text-muted-foreground leading-relaxed max-w-2xl">
                Voicebox is open-source and used by creators, voice artists, podcasters, writers,
                developers, accessibility users, and curious humans all over the world. Sponsor the
                project and your logo lands on the homepage, in the app, in the README, and on the
                sponsors page — in front of every one of them.
              </p>
            </div>

            <div className="flex flex-col items-start md:items-end gap-2">
              <a
                href="/sponsors"
                className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 text-sm font-semibold uppercase tracking-wider text-white shadow-[0_4px_20px_hsl(43_60%_50%/0.3),inset_0_2px_0_rgba(255,255,255,0.2),inset_0_-2px_0_rgba(0,0,0,0.1)] transition-all hover:bg-accent-faint whitespace-nowrap"
              >
                Become a sponsor
                <ArrowRight className="h-4 w-4" />
              </a>
              <span className="text-xs text-muted-foreground/70">From $500 / month</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function SponsorStrip({ sponsors }: { sponsors: Sponsor[] }) {
  return (
    <section className="border-t border-border py-14">
      <div className="mx-auto max-w-6xl px-6">
        <div className="text-center mb-8">
          <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-muted-foreground/80">
            Sponsored by
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-5 md:gap-6">
          {sponsors.map((sponsor) => (
            <a
              key={sponsor.name}
              href={sponsor.url}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={sponsor.name}
              className="group flex h-32 min-w-[260px] items-center justify-center rounded-2xl border border-border bg-card/40 backdrop-blur-sm px-10 transition-all hover:border-accent/40 hover:bg-card"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={sponsor.logoSrc}
                alt={sponsor.logoAlt ?? sponsor.name}
                className={`h-14 w-auto max-w-[220px] object-contain opacity-80 transition-opacity group-hover:opacity-100 ${
                  sponsor.invert ? 'brightness-0 invert' : ''
                }`}
              />
            </a>
          ))}
        </div>

        <div className="mt-8 text-center">
          <a
            href="/sponsors"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            Become a sponsor
            <ArrowRight className="h-3.5 w-3.5" />
          </a>
        </div>
      </div>
    </section>
  );
}
