/**
 * Supported languages for voice generation.
 * Most languages use Qwen3-TTS; Hebrew uses Chatterbox TTS.
 */

export const SUPPORTED_LANGUAGES = {
  zh: 'Chinese',
  en: 'English',
  ja: 'Japanese',
  ko: 'Korean',
  de: 'German',
  fr: 'French',
  ru: 'Russian',
  pt: 'Portuguese',
  es: 'Spanish',
  it: 'Italian',
  he: 'Hebrew',
} as const;

export type LanguageCode = keyof typeof SUPPORTED_LANGUAGES;

export const LANGUAGE_CODES = Object.keys(SUPPORTED_LANGUAGES) as LanguageCode[];

export const LANGUAGE_OPTIONS = LANGUAGE_CODES.map((code) => ({
  value: code,
  label: SUPPORTED_LANGUAGES[code],
}));
