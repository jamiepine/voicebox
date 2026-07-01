'use client';

import { Github } from 'lucide-react';
import { useEffect, useState } from 'react';
import { AgentIntegration } from '@/components/AgentIntegration';
import { CaptureHero } from '@/components/CaptureHero';
import { CapturesMockup } from '@/components/CapturesMockup';
import { Footer } from '@/components/Footer';
import { Navbar } from '@/components/Navbar';
import { AppleIcon, LinuxIcon, WindowsIcon } from '@/components/PlatformIcons';
import { useLocale } from '@/components/LocaleProvider';
import { GITHUB_REPO } from '@/lib/constants';
import { getLocalizedPath, getLocaleTag } from '@/lib/i18n';

export default function CapturePage() {
  const [version, setVersion] = useState<string | null>(null);
  const [totalDownloads, setTotalDownloads] = useState<number | null>(null);
  const locale = useLocale();
  const isRussian = locale === 'ru';

  useEffect(() => {
    fetch('/api/releases')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch releases');
        return res.json();
      })
      .then((data) => {
        if (data.version) setVersion(data.version);
        if (data.totalDownloads != null) setTotalDownloads(data.totalDownloads);
      })
      .catch((error) => {
        console.error('Failed to fetch release info:', error);
      });
  }, []);

  return (
    <>
      <Navbar />

      {/* ── Hero ─────────────────────────────────────────────────── */}
      <CaptureHero version={version} totalDownloads={totalDownloads} />

      {/* ── Captures mockup ─────────────────────────────────────── */}
      <section className="relative border-t border-border py-24">
        <div className="mx-auto max-w-5xl px-6 text-center mb-14">
          <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-4">
            {isRussian ? 'Вкладка «Записи»' : 'The Captures tab'}
          </div>
          <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground mb-4">
            {isRussian
              ? 'Каждый захват хранит и аудио, и расшифровку.'
              : 'Every capture, paired with audio and transcript.'}
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            {isRussian
              ? 'Зажмите хоткей, скажите фразу и отпустите: запись сразу попадает во вкладку «Записи». Можно переслушать исходное аудио, заново расшифровать его другой моделью, доработать локальной LLM, скопировать в буфер обмена или сразу отправить MCP-агенту. Ничего не уходит в облако.'
              : 'Hold the shortcut, speak, release — a capture lands in the Captures tab. Replay the original audio, re-transcribe with a different model, refine with a local LLM, copy to clipboard, or send it straight to any MCP-aware agent. Nothing leaves your machine.'}
          </p>
        </div>
        <CapturesMockup />
      </section>

      {/* ── Feature bullets ─────────────────────────────────────── */}
      <section className="border-t border-border py-24">
        <div className="mx-auto max-w-6xl px-6">
          <div className="grid md:grid-cols-3 gap-6">
            <div className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6">
              <h3 className="text-[15px] font-semibold text-foreground mb-2">
                {isRussian ? 'Whisper под любую машину' : 'Whisper, sized for every machine'}
              </h3>
              <p className="text-sm leading-relaxed text-muted-foreground">
                {isRussian
                  ? 'Base, Small, Medium, Large и Turbo. Выбирайте модель для каждой записи отдельно: 99 языков на каждом уровне, всё локально, всё скачивается прямо из приложения.'
                  : 'Base, Small, Medium, Large, and Turbo. Pick per-capture — 99 languages at every tier, all local, all downloadable from inside the app.'}
              </p>
            </div>
            <div className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6">
              <h3 className="text-[15px] font-semibold text-foreground mb-2">
                {isRussian
                  ? 'LLM-очистка, которая не переписывает вас'
                  : 'LLM refinement that respects your words'}
              </h3>
              <p className="text-sm leading-relaxed text-muted-foreground">
                {isRussian
                  ? 'Локальная модель Qwen убирает междометия, самоисправления и правит пунктуацию, не меняя смысл формулировок. Исходный и улучшенный текст хранятся рядом, оригинальное аудио тоже сохраняется.'
                  : 'A local Qwen model cleans ums, self-corrections, and punctuation — without rephrasing. Keep raw and refined side-by-side; the original audio is always kept.'}
              </p>
            </div>
            <div className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6">
              <h3 className="text-[15px] font-semibold text-foreground mb-2">
                {isRussian ? 'Архив по умолчанию' : 'Archived by default'}
              </h3>
              <p className="text-sm leading-relaxed text-muted-foreground">
                {isRussian
                  ? 'Каждая диктовка сохраняет и аудио, и текст. Ищите, переобрабатывайте или превращайте любую запись в голосовой сэмпл для клонирования прямо из вкладки «Записи».'
                  : 'Every dictation keeps both the audio and the transcript. Search, re-run, or turn any capture into a voice sample for cloning from the Captures tab.'}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Agent voice output ──────────────────────────────────── */}
      <AgentIntegration />

      {/* ── Bottom CTA ──────────────────────────────────────────── */}
      <section id="download" className="border-t border-border py-24">
        <div className="mx-auto max-w-4xl px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-semibold tracking-tight text-foreground md:text-4xl mb-4">
              {isRussian ? 'Установите Voicebox и начинайте диктовать.' : 'Install Voicebox, start dictating.'}
            </h2>
            <p className="text-muted-foreground">
              {isRussian
                ? 'Бесплатно, open-source, локально. Без аккаунта, без API-ключей и без тарифа за символы.'
                : 'Free, open-source, local. No account, no API keys, no per-character fees.'}
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-2xl mx-auto">
            <a
              href={getLocalizedPath(locale, '/download?platform=macArm')}
              className="flex items-center rounded-xl border border-border bg-card/60 backdrop-blur-sm px-5 py-4 transition-all hover:border-accent/30 hover:bg-card group"
            >
              <AppleIcon className="h-6 w-6 shrink-0 text-muted-foreground group-hover:text-foreground transition-colors" />
              <div className="ml-4">
                <div className="text-sm font-medium">macOS</div>
                <div className="text-xs text-muted-foreground">Apple Silicon (ARM)</div>
              </div>
            </a>
            <a
              href={getLocalizedPath(locale, '/download?platform=macIntel')}
              className="flex items-center rounded-xl border border-border bg-card/60 backdrop-blur-sm px-5 py-4 transition-all hover:border-accent/30 hover:bg-card group"
            >
              <AppleIcon className="h-6 w-6 shrink-0 text-muted-foreground group-hover:text-foreground transition-colors" />
              <div className="ml-4">
                <div className="text-sm font-medium">macOS</div>
                <div className="text-xs text-muted-foreground">Intel (x64)</div>
              </div>
            </a>
            <a
              href={getLocalizedPath(locale, '/download?platform=windows')}
              className="flex items-center rounded-xl border border-border bg-card/60 backdrop-blur-sm px-5 py-4 transition-all hover:border-accent/30 hover:bg-card group"
            >
              <WindowsIcon className="h-6 w-6 shrink-0 text-muted-foreground group-hover:text-foreground transition-colors" />
              <div className="ml-4">
                <div className="text-sm font-medium">Windows</div>
                <div className="text-xs text-muted-foreground">{isRussian ? '64-битный MSI' : '64-bit (MSI)'}</div>
              </div>
            </a>
            <a
              href={getLocalizedPath(locale, '/linux-install')}
              className="flex items-center rounded-xl border border-border bg-card/60 backdrop-blur-sm px-5 py-4 transition-all hover:border-accent/30 hover:bg-card group"
            >
              <LinuxIcon className="h-6 w-6 shrink-0 text-muted-foreground group-hover:text-foreground transition-colors" />
              <div className="ml-4">
                <div className="text-sm font-medium">Linux</div>
                <div className="text-xs text-muted-foreground">{isRussian ? 'Сборка из исходников' : 'Build from source'}</div>
              </div>
            </a>
          </div>

          <div className="mt-6 text-center">
            <a
              href={`${GITHUB_REPO}/releases`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <Github className="h-4 w-4" />
              {isRussian ? 'Все релизы на GitHub' : 'View all releases on GitHub'}
            </a>
          </div>

          <div className="mt-10 text-center">
            <a
              href={getLocalizedPath(locale, '/')}
              className="text-sm text-muted-foreground/70 hover:text-foreground transition-colors"
            >
              {isRussian ? '← Посмотреть всё, что умеет Voicebox' : '← See everything Voicebox can do'}
            </a>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
