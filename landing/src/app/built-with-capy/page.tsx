import { ArrowUpRight } from 'lucide-react';
import type { Metadata } from 'next';
import Image from 'next/image';
import { Footer } from '@/components/Footer';
import { Navbar } from '@/components/Navbar';
import { capyUrl, GITHUB_REPO } from '@/lib/constants';

export const metadata: Metadata = {
  title: 'How Voicebox is built - Voicebox',
  description:
    'Voicebox is funded by Capy and built with Capy. How a Captain thread and its crew test, rebase, and review every contributor pull request before Jamie merges it.',
};

const STEPS = [
  {
    title: 'A Captain directs the work',
    body: 'One long-running Capy thread, the Captain, owns the Voicebox repository day to day. It reads new issues and pull requests, decides what to do next, and hands each job to a crew thread with a written brief.',
  },
  {
    title: 'Crew threads take one job each',
    body: 'Each crew thread runs on its own cloud machine with a fresh checkout. It does one job, such as preparing a contributor pull request or drafting release notes, then writes a report for the Captain and stops.',
  },
  {
    title: 'Every contributor PR is tested on Linux and on a Mac Studio',
    body: 'A crew thread rebases the contributor branch on main, runs the typecheck, the web build, and the backend test suite on Linux, and runs the macOS and MLX checks on a Mac Studio. If something breaks, it fixes it in a follow-up commit and keeps the contributor as the author.',
  },
  {
    title: 'Reviewed, then merged by Jamie',
    body: 'The prepared branch is opened as a pull request in the Voicebox repository, where it gets a Capy review and CI. Nothing merges on its own. Jamie reads the Captain\u2019s summary and says the word.',
  },
  {
    title: 'A daily look at the field',
    body: 'A scheduled Capy automation checks comparable open-source voice tools once a day and writes a short note for the Captain, so feature parity is a plan, not a surprise.',
  },
  {
    title: 'Releases branch off main',
    body: 'When a release is ready it is cut on its own release branch and built by the GitHub release workflow. Main keeps moving toward the next version while the release waits on signing and notarization.',
  },
];

export default function BuiltWithCapy() {
  return (
    <>
      <Navbar />

      <section className="relative pt-32 pb-24">
        <div className="mx-auto max-w-2xl px-6">
          <a
            href={capyUrl('site-page')}
            target="_blank"
            rel="noopener noreferrer"
            className="mb-8 inline-flex items-center gap-3"
          >
            <Image
              src="/capy/capy-wordmark-dark.svg"
              alt="Capy"
              width={106}
              height={30}
              className="h-[30px] w-auto"
            />
          </a>

          <div className="mb-4 text-[11px] font-semibold uppercase tracking-[0.22em] text-accent">
            Funded by Capy
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground md:text-4xl">
            How Voicebox is built
          </h1>

          <p className="mt-4 text-muted-foreground">
            <a
              href={capyUrl('site-page')}
              target="_blank"
              rel="noopener noreferrer"
              className="text-foreground hover:underline"
            >
              Capy
            </a>{' '}
            funds Voicebox. Capy is also the tool Voicebox is built and maintained with. Voicebox
            stays <b className="text-foreground">free and open source</b> under the MIT license,
            with no splash screens, banners, or sponsored interruptions in the app. This page
            describes how the work gets done.
          </p>

          <ol className="mt-10 space-y-6">
            {STEPS.map((step, index) => (
              <li
                key={step.title}
                className="rounded-xl border border-border bg-card/40 px-5 py-4 backdrop-blur-sm"
              >
                <div className="flex items-baseline gap-3">
                  <span className="text-xs font-semibold tabular-nums text-accent">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <h2 className="text-base font-semibold text-foreground">{step.title}</h2>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{step.body}</p>
              </li>
            ))}
          </ol>

          <div className="mt-10 space-y-4 text-sm text-muted-foreground">
            <p>
              Contributor commits keep their authors. Fixes a crew thread adds on top are separate
              commits, so the history shows who wrote what. Every prepared pull request is public in
              the{' '}
              <a
                href={`${GITHUB_REPO}/pulls`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground hover:underline"
              >
                Voicebox repository
              </a>
              .
            </p>
          </div>

          <div className="mt-10 flex flex-wrap gap-3">
            <a
              href={capyUrl('site-page')}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 text-sm font-semibold text-white shadow-[0_4px_20px_hsl(43_60%_50%/0.3),inset_0_2px_0_rgba(255,255,255,0.2),inset_0_-2px_0_rgba(0,0,0,0.1)] transition-all hover:bg-accent-faint"
            >
              Try Capy
              <ArrowUpRight className="h-4 w-4" />
            </a>
            <a
              href={GITHUB_REPO}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-full border border-border/60 bg-card/40 px-6 py-3 text-sm font-medium text-muted-foreground transition-colors hover:border-border hover:text-foreground"
            >
              Voicebox on GitHub
            </a>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
