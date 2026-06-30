import { defineI18n } from 'fumadocs-core/i18n';

// English stays at `/`, Spanish at `/es`.
export const i18n = defineI18n({
  languages: ['en', 'es'],
  defaultLanguage: 'en',
  hideLocale: 'default-locale',
});
