'use client';

import { ArrowRight, Check, Coffee, Mail } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Footer } from '@/components/Footer';
import { Navbar } from '@/components/Navbar';
import { useLocale } from '@/components/LocaleProvider';
import {
  DONATE_URL,
  SPONSOR_CHECKOUT_URL,
  SPONSOR_CONTACT_EMAIL,
} from '@/lib/constants';

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, '')}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return n.toLocaleString();
}

export default function SponsorsPage() {
  const [downloads, setDownloads] = useState<number | null>(null);
  const [stars, setStars] = useState<number | null>(null);
  const locale = useLocale();
  const isRussian = locale === 'ru';

  useEffect(() => {
    fetch('/api/releases')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.totalDownloads != null) setDownloads(data.totalDownloads);
      })
      .catch(() => {});
    fetch('/api/stars')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (typeof data?.count === 'number') setStars(data.count);
      })
      .catch(() => {});
  }, []);

  return (
    <>
      <Navbar />

      {/* ── Hero ─────────────────────────────────────────────────── */}
      <section className="relative pt-32 pb-16">
        <div className="hero-glow hero-glow-fade pointer-events-none absolute inset-0 -top-32">
          <div className="absolute left-1/2 top-0 -translate-x-1/2 w-[900px] h-[500px] rounded-full bg-accent/12 blur-[140px]" />
          <div className="absolute left-1/2 top-16 -translate-x-1/2 w-[520px] h-[360px] rounded-full bg-accent/8 blur-[80px]" />
        </div>

        <div className="relative mx-auto max-w-4xl px-6 text-center">
          <div
            className="fade-in mb-6 text-[11px] font-semibold uppercase tracking-[0.22em] text-accent"
            style={{ animationDelay: '50ms' }}
          >
            {isRussian ? 'VIP-спонсор' : 'VIP Sponsor'}
          </div>

          <h1
            className="fade-in text-5xl font-bold tracking-tighter leading-[0.95] text-foreground md:text-6xl lg:text-7xl"
            style={{ animationDelay: '100ms' }}
          >
            {isRussian ? 'Покажите бренд полумиллиону создателей.' : 'Get your brand in front of half a million creators.'}
          </h1>

          <p
            className="fade-in mx-auto mt-6 max-w-2xl text-lg text-muted-foreground md:text-xl"
            style={{ animationDelay: '200ms' }}
          >
            {isRussian
              ? 'Voicebox — open-source AI voice studio, которой пользуются авторы, подкастеры, дикторы, писатели, разработчики, люди, которым важна доступность, и просто любопытные пользователи по всему миру. Станьте спонсором проекта и покажите им свой логотип.'
              : 'Voicebox is the open-source AI voice studio used by creators, podcasters, voice artists, writers, developers, accessibility users, and curious humans all over the world. Sponsor the project, get your logo in front of all of them.'}
          </p>

          <div
            className="fade-in mt-10 flex flex-row items-center justify-center gap-3 sm:gap-4"
            style={{ animationDelay: '300ms' }}
          >
            <a
              href="#sponsor"
              className="rounded-full bg-accent px-8 py-3.5 text-sm font-semibold uppercase tracking-wider text-white shadow-[0_4px_20px_hsl(43_60%_50%/0.3),inset_0_2px_0_rgba(255,255,255,0.2),inset_0_-2px_0_rgba(0,0,0,0.1)] transition-all hover:bg-accent-faint active:shadow-[0_2px_10px_hsl(43_60%_50%/0.3),inset_0_4px_8px_rgba(0,0,0,0.3)]"
            >
              {isRussian ? 'Стать спонсором' : 'Become a sponsor'}
            </a>
            <a
              href={`mailto:${SPONSOR_CONTACT_EMAIL}`}
              className="flex items-center gap-2 rounded-full border border-border/60 bg-card/40 backdrop-blur-sm px-6 py-3 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground hover:border-border"
            >
              <Mail className="h-4 w-4" />
              {isRussian ? 'Связаться с нами' : 'Talk to us'}
            </a>
          </div>
        </div>
      </section>

      {/* ── Traction ────────────────────────────────────────────── */}
      <section className="border-t border-border py-20">
        <div className="mx-auto max-w-5xl px-6">
          <div className="text-center mb-12">
            <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-4">
              {isRussian ? 'Охват' : 'Reach'}
            </div>
            <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">
              {isRussian ? 'Реальное внимание, реальное распространение.' : 'Real distribution, real attention.'}
            </h2>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Stat
              value={downloads != null ? formatCount(downloads) : '500k+'}
              label={isRussian ? 'Загрузки' : 'Downloads'}
              note={isRussian ? 'С момента запуска в феврале 2026' : 'Since launch in February 2026'}
            />
            <Stat
              value={stars != null ? formatCount(stars) : '22k+'}
              label={isRussian ? 'Звёзды GitHub' : 'GitHub stars'}
              note={isRussian ? 'Трендовый #1 мейнтейнер, #4 репозиторий' : 'Trending #1 maintainer, #4 repo'}
            />
            <Stat
              value="170k+"
              label={isRussian ? 'Посетители сайта в месяц' : 'Monthly site visitors'}
              note={isRussian ? 'voicebox.sh, последние 30 дней · рост в 5×' : 'voicebox.sh, last 30 days · growing 5×'}
            />
            <Stat
              value={isRussian ? 'Миллионы' : 'Millions'}
              label={isRussian ? 'Охват в соцсетях' : 'Social reach'}
              note={isRussian ? 'Туториалы, reels, TikTok' : 'Tutorials, reels, TikToks'}
            />
          </div>

          <p className="text-center text-sm text-muted-foreground mt-10 max-w-2xl mx-auto">
            {isRussian
              ? 'Пользователи Voicebox — это авторы, подкастеры, дикторы, писатели, разработчики, люди, которым важна доступность, энтузиасты и хоббисты. Они сознательно выбрали local-first инструмент вместо облачной подписки и ценят контроль над своим софтом и брендами, которые их поддерживают.'
              : 'Voicebox users are content creators, podcasters, voice artists, writers, developers, accessibility users, hobbyists, and AI enthusiasts. They picked a local-first tool over a cloud subscription — they care about owning their software and the brands behind it.'}
          </p>
        </div>
      </section>

      {/* ── What you get ────────────────────────────────────────── */}
      <section className="border-t border-border py-20">
        <div className="mx-auto max-w-5xl px-6">
          <div className="text-center mb-12">
            <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-4">
              {isRussian ? 'Размещение' : 'Placement'}
            </div>
            <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">
              {isRussian ? 'Где будет виден ваш логотип.' : 'Where your logo shows up.'}
            </h2>
          </div>

          <div className="rounded-2xl border-2 border-accent/40 bg-card/60 backdrop-blur-sm p-8 mb-4 shadow-[0_8px_40px_hsl(43_60%_50%/0.08)]">
            <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-2">
              {isRussian ? 'Главная позиция' : 'Headline placement'}
            </div>
            <h3 className="text-xl md:text-2xl font-semibold tracking-tight text-foreground mb-2">
              {isRussian ? 'voicebox.sh — сразу под hero-блоком.' : 'voicebox.sh — directly below the hero.'}
            </h3>
            <p className="text-sm md:text-base text-muted-foreground leading-relaxed max-w-2xl mb-8">
              {isRussian
                ? 'Ваш логотип занимает главный слот на главной странице — первое, что видит посетитель сразу после hero-блока. Он стоит в одном ряду с остальными VIP-спонсорами и ведёт на выбранный вами URL. Именно такое размещение действительно работает.'
                : 'Your logo lives in the prime slot on the homepage — the first thing every visitor sees after the hero. Same row as every other VIP Sponsor, linked to your URL of choice. This is the placement that actually moves the needle.'}
            </p>

            {/* Preview of what the placement looks like on the homepage */}
            <div className="rounded-xl border border-dashed border-border/80 bg-background/50 p-6 md:p-8">
              <div className="text-center mb-6">
                <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-muted-foreground/80">
                  {isRussian ? 'При поддержке' : 'Sponsored by'}
                </div>
              </div>
              <div className="flex flex-wrap items-center justify-center gap-5 md:gap-6">
                <div className="flex h-32 min-w-[260px] items-center justify-center rounded-2xl border border-border bg-card/40 backdrop-blur-sm px-10">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src="/sponsors/openai.svg"
                    alt={isRussian ? 'Пример логотипа спонсора' : 'Example sponsor logo'}
                    className="h-14 w-auto max-w-[220px] object-contain brightness-0 invert opacity-80"
                  />
                </div>
              </div>
              <p className="mt-6 text-center text-xs text-muted-foreground/70 italic">
                {isRussian
                  ? 'Это только пример. Реальные логотипы спонсоров появятся здесь после запуска размещений.'
                  : 'Example only — actual sponsor logos appear here once placements are live.'}
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <Perk
              title="GitHub README"
              body={isRussian ? 'Логотип в секции Sponsors репозитория. README — один из самых просматриваемых документов любого трендового проекта на GitHub.' : "Logo in the repo's Sponsors section. The README is one of the most-viewed docs on GitHub for any trending project."}
            />
            <Perk
              title="/sponsors page"
              body={isRussian ? 'Отдельная карточка на этой странице: ваш логотип, слоган, описание компании и прямая ссылка.' : 'Dedicated logo card on this page with your tagline, what your company does, and a direct link out.'}
            />
            <Perk
              title={isRussian ? 'Релиз-ноты' : 'Release notes'}
              body={isRussian ? 'Отдельная строка благодарности в следующем большом посте о релизе — её видит длинный хвост пользователей, которые следят за обновлениями Voicebox.' : 'One-line acknowledgement in the next major release post — read by the long tail of users who follow Voicebox updates.'}
            />
          </div>
        </div>
      </section>

      {/* ── Pricing ─────────────────────────────────────────────── */}
      <section id="sponsor" className="border-t border-border py-20">
        <div className="mx-auto max-w-3xl px-6">
          <div className="text-center mb-10">
            <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-accent mb-4">
              {isRussian ? 'Стоимость' : 'Pricing'}
            </div>
            <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">
              {isRussian ? 'Один тариф. Помесячно. Отмена в любой момент.' : 'One tier. Month-to-month. Cancel anytime.'}
            </h2>
          </div>

          <div className="rounded-2xl border-2 border-accent/40 bg-card/60 backdrop-blur-sm p-8 md:p-10 shadow-[0_8px_40px_hsl(43_60%_50%/0.1)]">
            <div className="flex items-baseline gap-2 mb-2">
              <span className="text-5xl font-bold tracking-tight text-foreground">$500</span>
              <span className="text-base text-muted-foreground">{isRussian ? '/ месяц' : '/ month'}</span>
            </div>
            <p className="text-sm text-muted-foreground mb-8">
              {isRussian
                ? 'Списание раз в месяц через Stripe. Логотип появляется в течение 48 часов после оплаты.'
                : 'Billed monthly via Stripe. Logo goes live within 48 hours of payment.'}
            </p>

            <ul className="space-y-3 mb-8">
              <PerkRow text={isRussian ? 'Логотип на voicebox.sh — сразу под hero-блоком' : 'Logo on voicebox.sh — directly below the hero'} />
              <PerkRow text={isRussian ? 'Логотип в секции sponsors в GitHub README' : 'Logo in the GitHub README sponsors section'} />
              <PerkRow text={isRussian ? 'Карточка на /sponsors с вашим слоганом и ссылкой' : 'Featured card on /sponsors with your tagline and link'} />
              <PerkRow text={isRussian ? 'Упоминание в следующем большом релизном посте' : 'Acknowledgement in the next major release post'} />
              <PerkRow text={isRussian ? 'Прямой контакт с командой для сотрудничества' : 'Direct line to the team for collaboration'} />
            </ul>

            <a
              href={SPONSOR_CHECKOUT_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 w-full rounded-full bg-accent px-6 py-3.5 text-sm font-semibold uppercase tracking-wider text-white shadow-[0_4px_20px_hsl(43_60%_50%/0.3),inset_0_2px_0_rgba(255,255,255,0.2),inset_0_-2px_0_rgba(0,0,0,0.1)] transition-all hover:bg-accent-faint"
            >
              {isRussian ? 'Стать спонсором Voicebox' : 'Sponsor Voicebox'}
              <ArrowRight className="h-4 w-4" />
            </a>

            <p className="text-center text-xs text-muted-foreground/70 mt-4">
              {isRussian ? 'Нужен годовой контракт, инвойсы или более заметное размещение? ' : 'Need an annual contract, invoicing, or higher placement? '}{' '}
              <a
                href={`mailto:${SPONSOR_CONTACT_EMAIL}`}
                className="text-foreground/80 underline-offset-4 hover:underline"
              >
                {SPONSOR_CONTACT_EMAIL}
              </a>
            </p>
          </div>
        </div>
      </section>

      {/* ── Individual / policy ─────────────────────────────────── */}
      <section className="border-t border-border py-20">
        <div className="mx-auto max-w-4xl px-6 grid md:grid-cols-2 gap-4">
          <div className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6">
            <div className="flex items-center gap-2 mb-3">
              <Coffee className="h-5 w-5 text-[#FFDD00]" />
              <h3 className="text-[15px] font-semibold text-foreground">{isRussian ? 'Вы не компания?' : 'Not a company?'}</h3>
            </div>
            <p className="text-sm leading-relaxed text-muted-foreground mb-4">
              {isRussian
                ? 'Индивидуальные донаты тоже помогают Voicebox жить дальше. Оставьте чаевые через Buy Me a Coffee, и ваше имя появится в списке сторонников.'
                : 'Individual supporters keep Voicebox running too. Drop a tip on Buy Me a Coffee and your name shows up in the supporters list.'}
            </p>
            <a
              href={DONATE_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-sm font-medium text-foreground/80 hover:text-foreground transition-colors"
            >
              {isRussian ? 'Поддержать через Buy Me a Coffee' : 'Support on Buy Me a Coffee'}
              <ArrowRight className="h-4 w-4" />
            </a>
          </div>

          <div className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6">
            <h3 className="text-[15px] font-semibold text-foreground mb-3">
              {isRussian ? 'Кого мы принимаем' : 'Who we accept'}
            </h3>
            <p className="text-sm leading-relaxed text-muted-foreground">
              {isRussian
                ? 'Voicebox — local-first и privacy-first проект. Мы не принимаем спонсорство от компаний, чья бизнес-модель этому противоречит: брокеров голосовых данных, ad-tech поверх речи или поставщиков инструментов слежки. Всем остальным рады.'
                : "Voicebox is local-first and privacy-first. We don't accept sponsorships from companies whose business model conflicts with that — voice-data brokers, ad-tech built on speech, or surveillance vendors. Everyone else is welcome."}
            </p>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}

function Stat({ value, label, note }: { value: string; label: string; note: string }) {
  return (
    <div className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-5 text-center">
      <div className="text-3xl md:text-4xl font-bold tracking-tight text-foreground">{value}</div>
      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-accent mt-2">
        {label}
      </div>
      <div className="text-xs text-muted-foreground mt-2">{note}</div>
    </div>
  );
}

function Perk({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-xl border border-border bg-card/40 backdrop-blur-sm p-6">
      <h3 className="text-[15px] font-semibold text-foreground mb-2">{title}</h3>
      <p className="text-sm leading-relaxed text-muted-foreground">{body}</p>
    </div>
  );
}

function PerkRow({ text }: { text: string }) {
  return (
    <li className="flex items-start gap-3 text-sm text-foreground/90">
      <Check className="h-5 w-5 shrink-0 text-accent mt-px" />
      <span>{text}</span>
    </li>
  );
}
