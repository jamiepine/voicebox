import { createFromSource } from 'fumadocs-core/search/server';
import { source } from '@/lib/source';

// https://docs.orama.com/docs/orama-js/supported-languages
export const { GET } = createFromSource(source, {
  localeMap: {
    en: { language: 'english' },
    es: { language: 'spanish' },
  },
});
