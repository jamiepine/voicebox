import i18n from '@/i18n';

/**
 * Supported languages for voice generation, per engine.
 *
 * Qwen3-TTS supports 10 languages.
 * LuxTTS is English-only.
 * Chatterbox Multilingual supports 23 languages.
 * Chatterbox Turbo is English-only.
 * Kokoro supports 8 languages.
 */

/** All languages that any engine supports. */
export const ALL_LANGUAGES = {
  ar: 'Arabic',
  da: 'Danish',
  de: 'German',
  el: 'Greek',
  en: 'English',
  es: 'Spanish',
  fi: 'Finnish',
  fr: 'French',
  he: 'Hebrew',
  hi: 'Hindi',
  it: 'Italian',
  ja: 'Japanese',
  ko: 'Korean',
  ms: 'Malay',
  nl: 'Dutch',
  no: 'Norwegian',
  pl: 'Polish',
  pt: 'Portuguese',
  ru: 'Russian',
  sv: 'Swedish',
  sw: 'Swahili',
  tr: 'Turkish',
  zh: 'Chinese',
} as const;

export type LanguageCode = keyof typeof ALL_LANGUAGES;

const LANGUAGE_LABELS_RU: Record<LanguageCode, string> = {
  ar: 'Арабский',
  da: 'Датский',
  de: 'Немецкий',
  el: 'Греческий',
  en: 'Английский',
  es: 'Испанский',
  fi: 'Финский',
  fr: 'Французский',
  he: 'Иврит',
  hi: 'Хинди',
  it: 'Итальянский',
  ja: 'Японский',
  ko: 'Корейский',
  ms: 'Малайский',
  nl: 'Нидерландский',
  no: 'Норвежский',
  pl: 'Польский',
  pt: 'Португальский',
  ru: 'Русский',
  sv: 'Шведский',
  sw: 'Суахили',
  tr: 'Турецкий',
  zh: 'Китайский',
};

export function getLanguageLabel(code: LanguageCode): string {
  const uiLanguage = i18n.resolvedLanguage ?? i18n.language;
  const primaryLanguage = uiLanguage.split('-')[0];

  if (uiLanguage === 'ru' || primaryLanguage === 'ru') {
    return LANGUAGE_LABELS_RU[code];
  }

  return ALL_LANGUAGES[code];
}

/** Per-engine supported language codes. */
export const ENGINE_LANGUAGES: Record<string, readonly LanguageCode[]> = {
  qwen: ['zh', 'en', 'ja', 'ko', 'de', 'fr', 'ru', 'pt', 'es', 'it'],
  luxtts: ['en'],
  chatterbox: [
    'ar',
    'da',
    'de',
    'el',
    'en',
    'es',
    'fi',
    'fr',
    'he',
    'hi',
    'it',
    'ja',
    'ko',
    'ms',
    'nl',
    'no',
    'pl',
    'pt',
    'ru',
    'sv',
    'sw',
    'tr',
    'zh',
  ],
  chatterbox_turbo: ['en'],
  tada: ['en', 'ar', 'zh', 'de', 'es', 'fr', 'it', 'ja', 'pl', 'pt'],
  kokoro: ['en', 'es', 'fr', 'hi', 'it', 'pt', 'ja', 'zh'],
  qwen_custom_voice: ['zh', 'en', 'ja', 'ko', 'de', 'fr', 'ru', 'pt', 'es', 'it'],
} as const;

/** Helper: get language options for a given engine. */
export function getLanguageOptionsForEngine(engine: string) {
  const codes = ENGINE_LANGUAGES[engine] ?? ENGINE_LANGUAGES.qwen;
  return codes.map((code) => ({
    value: code,
    label: getLanguageLabel(code),
  }));
}

// ── Backwards-compatible exports used elsewhere ──────────────────────
export const SUPPORTED_LANGUAGES = ALL_LANGUAGES;
export const LANGUAGE_CODES = Object.keys(ALL_LANGUAGES) as LanguageCode[];
export function getLanguageOptions() {
  return LANGUAGE_CODES.map((code) => ({
    value: code,
    label: getLanguageLabel(code),
  }));
}
export const LANGUAGE_OPTIONS = LANGUAGE_CODES.map((code) => ({
  value: code,
  label: ALL_LANGUAGES[code],
}));
