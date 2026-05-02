import type { Metadata } from 'next';
import { Footer } from '@/components/Footer';
import { Navbar } from '@/components/Navbar';
import { GITHUB_REPO } from '@/lib/constants';

export const metadata: Metadata = {
  title: 'Установка Voicebox в Linux',
  description:
    'Соберите Voicebox из исходников в Linux: клонирование, настройка и сборка в несколько команд.',
};

export default function LinuxInstallRu() {
  return (
    <>
      <Navbar />

      <section className="relative pt-32 pb-24">
        <div className="mx-auto max-w-2xl px-6">
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Установка в Linux</h1>

          <p className="mt-4 text-muted-foreground">
            Сейчас мы ещё разбираемся с CI-проблемами, из-за которых пока не можем стабильно
            выпускать готовую Linux-сборку. Пока что самый надёжный путь — собрать Voicebox из
            исходников. Обычно это занимает всего несколько минут.
          </p>

          <div className="mt-10 space-y-6">
            <div>
              <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-3">
                Что понадобится
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
                  — установить можно через{' '}
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
                  Системные зависимости Tauri —{' '}
                  <a
                    href="https://v2.tauri.app/start/prerequisites/#linux"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-foreground hover:underline"
                  >
                    см. документацию Tauri
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-3">
                Сборка из исходников
              </h2>
              <div className="space-y-3">
                <div className="rounded-lg border border-border bg-card/60 p-4 font-mono text-sm">
                  <div className="text-muted-foreground select-none"># Клонируем репозиторий</div>
                  <div>git clone https://github.com/jamiepine/voicebox.git</div>
                  <div>cd voicebox</div>
                </div>

                <div className="rounded-lg border border-border bg-card/60 p-4 font-mono text-sm">
                  <div className="text-muted-foreground select-none">
                    # Ставим все зависимости (Python venv, JS deps и т.д.)
                  </div>
                  <div>just setup</div>
                </div>

                <div className="rounded-lg border border-border bg-card/60 p-4 font-mono text-sm">
                  <div className="text-muted-foreground select-none"># Собираем приложение</div>
                  <div>just build</div>
                </div>
              </div>

              <p className="mt-4 text-sm text-muted-foreground">
                Готовое приложение будет лежать в{' '}
                <code className="text-xs bg-muted px-1.5 py-0.5 rounded">
                  tauri/src-tauri/target/release/bundle/
                </code>
              </p>
            </div>

            <div>
              <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-3">
                Или запуск в dev-режиме
              </h2>
              <div className="rounded-lg border border-border bg-card/60 p-4 font-mono text-sm">
                <div className="text-muted-foreground select-none">
                  # Запускаем dev-сервер с hot reload
                </div>
                <div>just dev</div>
              </div>
            </div>
          </div>

          <div className="mt-12 pt-8 border-t border-border flex flex-wrap gap-4 text-sm">
            <a
              href={GITHUB_REPO}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              Репозиторий GitHub
            </a>
            <a
              href={`${GITHUB_REPO}/issues`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              Сообщить о проблеме
            </a>
            <a
              href={`${GITHUB_REPO}/blob/main/CONTRIBUTING.md`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              Руководство по contribution
            </a>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
