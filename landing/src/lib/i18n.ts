export const DEFAULT_LOCALE = 'en' as const;
export const SECONDARY_LOCALES = ['ru'] as const;
export const SUPPORTED_LOCALES = [DEFAULT_LOCALE, ...SECONDARY_LOCALES] as const;

export type Locale = (typeof SUPPORTED_LOCALES)[number];
export type SecondaryLocale = (typeof SECONDARY_LOCALES)[number];

export function isSecondaryLocale(locale: string): locale is SecondaryLocale {
  return SECONDARY_LOCALES.includes(locale as SecondaryLocale);
}

export function isLocale(locale: string): locale is Locale {
  return SUPPORTED_LOCALES.includes(locale as Locale);
}

export function getLocalizedPath(locale: Locale, path = '/'): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  if (locale === DEFAULT_LOCALE) return normalizedPath;
  if (normalizedPath === '/') return `/${locale}`;
  return `/${locale}${normalizedPath}`;
}

export function getLocaleTag(locale: Locale): string {
  return locale === 'ru' ? 'ru-RU' : 'en-US';
}
