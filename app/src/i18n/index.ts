import i18n from 'i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import { initReactI18next } from 'react-i18next';
import en from './locales/en/translation.json';
import ja from './locales/ja/translation.json';
import ptBR from './locales/pt-BR/translation.json';
import zhCN from './locales/zh-CN/translation.json';
import zhTW from './locales/zh-TW/translation.json';

export const SUPPORTED_LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'pt-BR', label: 'Português (Brasil)' },
  { code: 'ja', label: '日本語' },
  { code: 'zh-CN', label: '简体中文' },
  { code: 'zh-TW', label: '繁體中文' },
] as const;

export type LanguageCode = (typeof SUPPORTED_LANGUAGES)[number]['code'];

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      'pt-BR': { translation: ptBR },
      ja: { translation: ja },
      'zh-CN': { translation: zhCN },
      'zh-TW': { translation: zhTW },
    },
    fallbackLng: 'en',
    supportedLngs: SUPPORTED_LANGUAGES.map((l) => l.code),
    load: 'currentOnly',
    interpolation: { escapeValue: false },
    react: { useSuspense: false },
    detection: {
      order: ['localStorage', 'navigator'],
      lookupLocalStorage: 'voicebox:lang',
      caches: ['localStorage'],
    },
  });

export default i18n;
