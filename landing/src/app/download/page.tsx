'use client';

import {
  ArrowLeft,
  Bot,
  Coffee,
  Download as DownloadIcon,
  FileText,
  Github,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { AppleIcon, LinuxIcon, WindowsIcon } from '@/components/PlatformIcons';
import { useLocale } from '@/components/LocaleProvider';
import { Button } from '@/components/ui/button';
import { DONATE_URL, GITHUB_RELEASES_PAGE, GITHUB_REPO } from '@/lib/constants';
import { getLocalizedPath } from '@/lib/i18n';
import type { DownloadLinks } from '@/lib/releases';

type Platform = keyof DownloadLinks;

type PlatformMeta = {
  key: Platform;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
};

const PLATFORMS_EN: PlatformMeta[] = [
  { key: 'macArm', label: 'macOS', description: 'Apple Silicon', icon: AppleIcon },
  { key: 'macIntel', label: 'macOS', description: 'Intel (x64)', icon: AppleIcon },
  { key: 'windows', label: 'Windows', description: '64-bit (MSI)', icon: WindowsIcon },
  { key: 'linux', label: 'Linux', description: 'Build from source', icon: LinuxIcon },
];

const PLATFORMS_RU: PlatformMeta[] = [
  { key: 'macArm', label: 'macOS', description: 'Чипы Apple Silicon', icon: AppleIcon },
  { key: 'macIntel', label: 'macOS', description: 'Процессоры Intel (x64)', icon: AppleIcon },
  { key: 'windows', label: 'Windows', description: '64-битный MSI', icon: WindowsIcon },
  { key: 'linux', label: 'Linux', description: 'Сборка из исходников', icon: LinuxIcon },
];

function getPlatforms(locale: 'en' | 'ru'): PlatformMeta[] {
  return locale === 'ru' ? PLATFORMS_RU : PLATFORMS_EN;
}

function detectPlatform(): Platform | null {
  if (typeof navigator === 'undefined') return null;
  const ua = navigator.userAgent;
  if (/Windows/i.test(ua)) return 'windows';
  if (/Linux/i.test(ua) && !/Android/i.test(ua)) return 'linux';
  if (/Mac/i.test(ua)) {
    // Apple Silicon Safari reports "Intel" for compat; default to ARM since
    // M-series is the majority. Users can click the Intel button if needed.
    return 'macArm';
  }
  return null;
}

function parseQueryPlatform(search: string): Platform | null {
  const params = new URLSearchParams(search);
  const raw = params.get('platform');
  if (!raw) return null;
  // Accept both camelCase and hyphenated forms (/download/mac-arm → ?platform=mac-arm).
  const normalized = raw
    .toLowerCase()
    .replace(/[-_\s]/g, '')
    .replace('macarm', 'macArm')
    .replace('macintel', 'macIntel');
  const valid: Platform[] = ['macArm', 'macIntel', 'windows', 'linux'];
  return (valid as string[]).includes(normalized) ? (normalized as Platform) : null;
}

export default function DownloadPage() {
  const [links, setLinks] = useState<DownloadLinks | null>(null);
  const [linksError, setLinksError] = useState(false);
  const [platform, setPlatform] = useState<Platform | null>(null);
  const [triggered, setTriggered] = useState(false);
  const locale = useLocale();
  const isRussian = locale === 'ru';
  const platforms = useMemo(() => getPlatforms(locale), [locale]);

  useEffect(() => {
    const fromQuery = parseQueryPlatform(window.location.search);
    const resolved = fromQuery ?? detectPlatform();
    // No prebuilt Linux binary yet — send Linux users to the build-from-source
    // instructions instead of sitting on /download trying to trigger a
    // download that doesn't exist.
    if (resolved === 'linux') {
      window.location.replace(getLocalizedPath(locale, '/linux-install'));
      return;
    }
    setPlatform(resolved);
  }, [locale]);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/releases')
      .then((r) => {
        if (!r.ok) throw new Error(`releases ${r.status}`);
        return r.json();
      })
      .then((data) => {
        if (cancelled) return;
        if (data.downloadLinks) setLinks(data.downloadLinks as DownloadLinks);
      })
      .catch(() => {
        if (!cancelled) setLinksError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (triggered || !links || !platform) return;
    const url = links[platform];
    if (!url) return;

    const a = document.createElement('a');
    a.href = url;
    a.rel = 'noopener';
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTriggered(true);
  }, [triggered, links, platform]);

  const activeMeta = useMemo(
    () => platforms.find((p) => p.key === platform) ?? null,
    [platform, platforms],
  );

  return (
    <div className="min-h-screen bg-background">
      {/* Minimal branded header */}
      <header className="border-b border-border/50">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <Link href={getLocalizedPath(locale, '/')} className="flex items-center gap-2.5">
            <Image
              src="/voicebox-logo-app.webp"
              alt="Voicebox"
              width={28}
              height={28}
              className="h-7 w-7"
            />
            <span className="text-[15px] font-semibold text-foreground">Voicebox</span>
          </Link>
          <Link
            href={getLocalizedPath(locale, '/')}
            className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            {isRussian ? 'Назад на voicebox.sh' : 'Back to voicebox.sh'}
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-16 md:py-24">
        {/* Hero */}
        <div className="flex flex-col md:flex-row md:items-center gap-10 md:gap-14">
          <Image
            src="/voicebox-logo-app.webp"
            alt="Voicebox"
            width={200}
            height={200}
            priority
            className="h-32 w-32 md:h-44 md:w-44 shrink-0 drop-shadow-2xl"
          />
          <div className="flex-1 min-w-0 text-center md:text-left">
            {triggered ? (
              <>
                <h1 className="text-4xl md:text-5xl font-semibold tracking-tight text-foreground mb-4">
                  {isRussian ? 'Загрузка началась.' : 'Your download has started.'}
                </h1>
                <p className="text-lg text-muted-foreground">
                  {activeMeta
                    ? isRussian
                      ? `Скачиваем Voicebox для ${activeMeta.label} (${activeMeta.description}). Проверьте папку загрузок.`
                      : `Downloading Voicebox for ${activeMeta.label} (${activeMeta.description}). Check your downloads folder.`
                    : isRussian
                      ? 'Проверьте папку загрузок — файл Voicebox уже должен быть там.'
                      : 'Check your downloads folder for Voicebox.'}
                </p>
              </>
            ) : (
              <>
                <h1 className="text-4xl md:text-5xl font-semibold tracking-tight text-foreground mb-4">
                  {linksError
                    ? isRussian
                      ? 'Не удалось загрузить данные о последнем релизе.'
                      : "We couldn't load the latest release."
                    : isRussian
                      ? 'Скачать Voicebox'
                      : 'Download Voicebox'}
                </h1>
                <p className="text-lg text-muted-foreground">
                  {linksError
                    ? isRussian
                      ? 'Сервер релизов временно недоступен. Попробуйте ещё раз через минуту.'
                      : 'Our release server is temporarily unreachable. Please try again in a moment.'
                    : isRussian
                      ? 'Выберите свою платформу, чтобы начать.'
                      : 'Pick your platform to get started.'}
                </p>
              </>
            )}
          </div>
        </div>

        {/* Platform buttons — always visible as a fallback */}
        {linksError ? (
          <div className="mt-12 rounded-xl border border-border bg-card/60 backdrop-blur-sm p-6 text-center">
            <p className="text-sm text-muted-foreground mb-4">
              {isRussian ? 'Если проблема повторяется, вы можете ' : 'If this keeps happening, you can '}{' '}
              <a
                href={`${GITHUB_RELEASES_PAGE}/latest`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-accent underline underline-offset-2 hover:text-accent/80"
              >
                {isRussian ? 'открыть релизы на GitHub' : 'browse releases on GitHub'}
              </a>
              {isRussian ? ' и скачать нужную сборку вручную.' : ' and grab the build for your platform manually.'}
            </p>
          </div>
        ) : (
          <div className="mt-12 rounded-xl border border-border bg-card/60 backdrop-blur-sm p-6">
            <h2 className="text-sm font-medium text-foreground mb-4">
              {triggered
                ? isRussian
                  ? 'Загрузка не сработала?'
                  : 'Download not working?'
                : isRussian
                  ? 'Выберите платформу'
                  : 'Choose your platform'}
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {platforms.map((meta) => {
                const isLinux = meta.key === 'linux';
                const url = isLinux ? getLocalizedPath(locale, '/linux-install') : links?.[meta.key];
                const isActive = meta.key === platform;
                const disabled = !isLinux && !url;
                return (
                  <a
                    key={meta.key}
                    href={url ?? '#'}
                    {...(isLinux ? {} : { download: true })}
                    aria-disabled={disabled}
                    onClick={(e) => {
                      if (disabled) e.preventDefault();
                    }}
                    className={`flex items-center rounded-xl border px-5 py-4 transition-all group ${
                      isActive
                        ? 'border-accent/40 bg-accent/5 hover:border-accent/60'
                        : 'border-border bg-card/40 hover:border-accent/30 hover:bg-card'
                    } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    <meta.icon className="h-6 w-6 shrink-0 text-muted-foreground group-hover:text-foreground transition-colors" />
                    <div className="ml-4 flex-1">
                      <div className="text-sm font-medium text-foreground">{meta.label}</div>
                      <div className="text-xs text-muted-foreground">{meta.description}</div>
                    </div>
                    <DownloadIcon className="h-4 w-4 text-muted-foreground/60 group-hover:text-accent transition-colors" />
                  </a>
                );
              })}
            </div>
          </div>
        )}

        {/* Donate — prominent, heartfelt, post-click context */}
        <div className="mt-16 rounded-2xl border border-border bg-gradient-to-br from-card via-card/80 to-background backdrop-blur-sm p-8 md:p-10 overflow-hidden relative">
          <div className="absolute top-0 right-0 w-64 h-64 bg-[#FFDD00]/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />
          <div className="relative">
            <div className="inline-flex items-center gap-2 mb-4">
              <span className="text-[11px] font-medium uppercase tracking-wider text-[#FFDD00]">
                {isRussian ? 'Письмо от мейнтейнера' : 'Hi from the maintainer'}
              </span>
            </div>
            <h2 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground mb-4">
              {isRussian ? 'На связи Jamie — Voicebox это мой сайд-проект.' : 'Jamie here — Voicebox is a side project.'}
            </h2>
            <p className="text-muted-foreground leading-relaxed mb-6 max-w-2xl">
              {isRussian
                ? 'Я делаю и поддерживаю Voicebox в свободное время. Проект полностью бесплатный, open-source и работает целиком на вашей машине: без аккаунтов, без облака, без подписок и без допродаж. Если Voicebox уже сэкономил вам счёт за ElevenLabs или просто порадовал, чашка кофе действительно помогает мне выпускать обновления, добавлять новые модели и чинить баги.'
                : "I build and maintain Voicebox in my spare time. It's completely free, open source, runs entirely on your machine — no accounts, no cloud, no subscriptions, no upsells. If it saves you an ElevenLabs bill or just made your day, a coffee genuinely helps me keep shipping updates, adding new models, and fixing bugs. Every little bit keeps the lights on."}
            </p>
            <Button asChild size="lg" className="bg-[#FFDD00]/10 border-[#FFDD00]/30 text-[#FFDD00] hover:bg-[#FFDD00]/20 hover:border-[#FFDD00]/50">
              <a href={DONATE_URL} target="_blank" rel="noopener noreferrer">
                <Coffee className="h-4 w-4 mr-2" />
                {isRussian ? 'Угостить кофе' : 'Buy me a coffee'}
              </a>
            </Button>
          </div>
        </div>

        {/* Resources */}
        <div className="mt-10">
          <h2 className="text-sm font-medium text-foreground mb-4">
            {isRussian ? 'Пока идёт загрузка' : 'While you wait'}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <a
              href="https://docs.voicebox.sh"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-5 hover:border-accent/30 hover:bg-card transition-all group"
            >
              <FileText className="h-5 w-5 text-accent mb-3" />
              <h3 className="text-sm font-medium text-foreground mb-1">
                {isRussian ? 'Почитать документацию' : 'Read the docs'}
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {isRussian
                  ? 'Разобраться с Voicebox: установка, клонирование голоса, REST API.'
                  : 'Get familiar with Voicebox — setup, voice cloning, the REST API.'}
              </p>
            </a>
            <a
              href="https://deepwiki.com/jamiepine/voicebox"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-5 hover:border-accent/30 hover:bg-card transition-all group"
            >
              <Bot className="h-5 w-5 text-accent mb-3" />
              <h3 className="text-sm font-medium text-foreground mb-1">
                {isRussian ? 'Есть вопросы? Спросите у AI.' : 'Got questions? Ask AI.'}
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {isRussian
                  ? 'DeepWiki знает Voicebox вдоль и поперёк. Спрашивайте что угодно.'
                  : 'DeepWiki is an AI that knows Voicebox inside-out. Ask anything.'}
              </p>
            </a>
            <a
              href={GITHUB_REPO}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-xl border border-border bg-card/60 backdrop-blur-sm p-5 hover:border-accent/30 hover:bg-card transition-all group"
            >
              <Github className="h-5 w-5 text-accent mb-3" />
              <h3 className="text-sm font-medium text-foreground mb-1">
                {isRussian ? 'Исходники на GitHub' : 'Source on GitHub'}
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {isRussian
                  ? 'Поставьте звезду, заведите issue или отправьте pull request.'
                  : 'Star the repo, file issues, or contribute a PR.'}
              </p>
            </a>
          </div>
        </div>
      </main>
    </div>
  );
}
