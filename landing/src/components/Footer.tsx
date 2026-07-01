'use client';

import { ArrowUpRight, Coffee, Coins } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { CopyAddress } from '@/components/CopyAddress';
import { useLocale } from '@/components/LocaleProvider';
import {
  DONATE_URL,
  GITHUB_REPO,
  TOKEN_CONTRACT_ADDRESS,
  TOKEN_TICKER,
} from '@/lib/constants';
import { getLocalizedPath } from '@/lib/i18n';

export function Footer() {
  const locale = useLocale();
  const isRussian = locale === 'ru';

  return (
    <footer className="border-t border-border py-12">
      <div className="mx-auto max-w-7xl px-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-8 mb-10">
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
              {isRussian
                ? 'Локальная open-source студия клонирования голоса. Бесплатно и без привязки к облаку.'
                : 'Open source voice cloning studio. Local-first, free forever.'}
            </p>
            <a
              href={DONATE_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-lg border border-border/60 bg-card/60 px-3 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground hover:border-[#FFDD00]/40"
              aria-label={isRussian ? 'Поддержать через Buy Me a Coffee' : 'Donate via Buy Me a Coffee'}
            >
              <Coffee className="h-4 w-4 text-[#FFDD00]" />
              <span className="text-[13px] font-medium">{isRussian ? 'Поддержать' : 'Donate'}</span>
            </a>
          </div>

          <div>
            <h4 className="text-sm font-semibold mb-3">{isRussian ? 'Продукт' : 'Product'}</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>
                <a
                  href={getLocalizedPath(locale, '/#features')}
                  className="hover:text-foreground transition-colors"
                >
                  {isRussian ? 'Клонирование' : 'Clone'}
                </a>
              </li>
              <li>
                <a
                  href={getLocalizedPath(locale, '/capture')}
                  className="hover:text-foreground transition-colors"
                >
                  {isRussian ? 'Записи' : 'Capture'}
                </a>
              </li>
              <li>
                <a
                  href={getLocalizedPath(locale, '/#mcp')}
                  className="hover:text-foreground transition-colors"
                >
                  MCP
                </a>
              </li>
              <li>
                <a
                  href={getLocalizedPath(locale, '/#about')}
                  className="hover:text-foreground transition-colors"
                >
                  {isRussian ? 'Модели' : 'Models'}
                </a>
              </li>
              <li>
                <a
                  href={getLocalizedPath(locale, '/#api')}
                  className="hover:text-foreground transition-colors"
                >
                  API
                </a>
              </li>
              <li>
                <a
                  href="/cloud"
                  className="hover:text-foreground transition-colors"
                >
                  Cloud
                </a>
              </li>
              <li>
                <a
                  href="/pricing"
                  className="hover:text-foreground transition-colors"
                >
                  {isRussian ? 'Цены' : 'Pricing'}
                </a>
              </li>
              <li>
                <a
                  href={getLocalizedPath(locale, '/download')}
                  className="hover:text-foreground transition-colors"
                >
                  {isRussian ? 'Скачать' : 'Download'}
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="text-sm font-semibold mb-3">{isRussian ? 'Ресурсы' : 'Resources'}</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>
                <a
                  href="/blog"
                  className="hover:text-foreground transition-colors"
                >
                  {isRussian ? 'Блог' : 'Blog'}
                </a>
              </li>
              <li>
                <Link
                  href="https://docs.voicebox.sh"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors"
                >
                  {isRussian ? 'Документация' : 'Documentation'}
                </Link>
              </li>
              <li>
                <Link
                  href={GITHUB_REPO}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors"
                >
                  {isRussian ? 'Исходный код' : 'Source Code'}
                </Link>
              </li>
              <li>
                <Link
                  href={`${GITHUB_REPO}/releases`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors"
                >
                  {isRussian ? 'Релизы' : 'Releases'}
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

          <div>
            <h4 className="text-sm font-semibold mb-3">{isRussian ? 'Другие проекты' : 'Also By'}</h4>
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
                {isRussian ? 'Подробнее о токене' : 'Token details'}
                <ArrowUpRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>
        </div>

        <div className="border-t border-border pt-6">
          <p className="text-center text-sm text-muted-foreground">
            {isRussian
              ? `© ${new Date().getFullYear()} Voicebox. Open source по лицензии MIT.`
              : `© ${new Date().getFullYear()} Voicebox. Open source under MIT license.`}
          </p>
        </div>
      </div>
    </footer>
  );
}
