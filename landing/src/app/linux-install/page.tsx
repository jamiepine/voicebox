import type { Metadata } from 'next';
import { Footer } from '@/components/Footer';
import { Navbar } from '@/components/Navbar';
import { GITHUB_REPO } from '@/lib/constants';

export const metadata: Metadata = {
  title: 'Linux Install - Voicebox',
  description: 'Build Voicebox from source on Linux. Clone, setup, and build in three commands.',
};

export default function LinuxInstall() {
  return (
    <>
      <Navbar />

      <section className="relative pt-32 pb-24">
        <div className="mx-auto max-w-2xl px-6">
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Install on Linux</h1>

          <p className="mt-4 text-muted-foreground">
            We&apos;re currently working through CI issues that prevent us from shipping a reliable
            pre-built binary for Linux. In the meantime, building from source is straightforward and
            takes just a few minutes.
          </p>

          <div className="mt-10 space-y-6">
            {/* Prerequisites */}
            <div>
              <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-3">
                Prerequisites
              </h2>
              <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                <li>
                  <a
                    href="https://git-scm.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-foreground hover:underline"
                  >
                    Git
                  </a>
                </li>
                <li>
                  <a
                    href="https://www.rust-lang.org/tools/install"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-foreground hover:underline"
                  >
                    Rust
                  </a>
                </li>
                <li>
                  <a
                    href="https://github.com/casey/just#installation"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-foreground hover:underline"
                  >
                    just
                  </a>{' '}
                  — install via{' '}
                  <code className="text-xs bg-muted px-1.5 py-0.5 rounded">cargo install just</code>
                </li>
                <li>
                  <a
                    href="https://bun.sh"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-foreground hover:underline"
                  >
                    Bun
                  </a>
                </li>
                <li>
                  Tauri system deps —{' '}
                  <a
                    href="https://v2.tauri.app/start/prerequisites/#linux"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-foreground hover:underline"
                  >
                    see Tauri docs
                  </a>
                </li>
              </ul>
            </div>

            {/* Steps */}
            <div>
              <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-3">
                Build from source
              </h2>
              <div className="space-y-3">
                <div className="rounded-lg border border-border bg-card/60 p-4 font-mono text-sm">
                  <div className="text-muted-foreground select-none"># Clone the repo</div>
                  <div>git clone https://github.com/jamiepine/voicebox.git</div>
                  <div>cd voicebox</div>
                </div>

                <div className="rounded-lg border border-border bg-card/60 p-4 font-mono text-sm">
                  <div className="text-muted-foreground select-none">
                    # Install all dependencies (Python venv, JS deps, etc.)
                  </div>
                  <div>just setup</div>
                </div>

                <div className="rounded-lg border border-border bg-card/60 p-4 font-mono text-sm">
                  <div className="text-muted-foreground select-none"># Build the app</div>
                  <div>just build</div>
                </div>
              </div>

              <p className="mt-4 text-sm text-muted-foreground">
                The built app will be in{' '}
                <code className="text-xs bg-muted px-1.5 py-0.5 rounded">
                  tauri/src-tauri/target/release/bundle/
                </code>
              </p>
            </div>

            {/* Dev mode */}
            <div>
              <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-3">
                Or run in dev mode
              </h2>
              <div className="rounded-lg border border-border bg-card/60 p-4 font-mono text-sm">
                <div className="text-muted-foreground select-none">
                  # Start the dev server with hot reload
                </div>
                <div>just dev</div>
              </div>
            </div>
          </div>

          {/* Links */}
          <div className="mt-12 pt-8 border-t border-border flex flex-wrap gap-4 text-sm">
            <a
              href={GITHUB_REPO}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              GitHub Repo
            </a>
            <a
              href={`${GITHUB_REPO}/issues`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              Report an issue
            </a>
            <a
              href={`${GITHUB_REPO}/blob/main/CONTRIBUTING.md`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              Contributing guide
            </a>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
