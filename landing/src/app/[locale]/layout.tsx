import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { LocaleProvider } from '@/components/LocaleProvider';
import {
  getLocalizedPath,
  isSecondaryLocale,
  type SecondaryLocale,
} from '@/lib/i18n';

type LocaleLayoutProps = {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
};

const LOCALE_METADATA: Record<SecondaryLocale, Metadata> = {
  ru: {
    title: 'Voicebox - локальная студия клонирования голоса с открытым исходным кодом',
    description:
      'Voicebox — локальная desktop-студия для клонирования голоса, диктовки и генерации речи. Бесплатно, без облака, для macOS, Windows и Linux.',
    openGraph: {
      title: 'Voicebox',
      description:
        'Локальная open-source студия для клонирования голоса, диктовки и голосовых ответов для AI-агентов.',
      url: 'https://voicebox.sh/ru',
      locale: 'ru_RU',
    },
    twitter: {
      title: 'Voicebox',
      description:
        'Локальная open-source студия для клонирования голоса, диктовки и голосовых ответов для AI-агентов.',
    },
  },
};

export const dynamicParams = false;

export function generateStaticParams() {
  return [{ locale: 'ru' }];
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  if (!isSecondaryLocale(locale)) return {};

  return {
    ...LOCALE_METADATA[locale],
    alternates: {
      canonical: `https://voicebox.sh${getLocalizedPath(locale, '/')}`,
      languages: {
        en: 'https://voicebox.sh',
        ru: 'https://voicebox.sh/ru',
      },
    },
  };
}

export default async function LocaleLayout({ children, params }: LocaleLayoutProps) {
  const { locale } = await params;
  if (!isSecondaryLocale(locale)) notFound();

  return <LocaleProvider locale={locale}>{children}</LocaleProvider>;
}
