import { source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';

const search = createFromSource(source, {
  // https://docs.orama.com/docs/orama-js/supported-languages
  language: 'english',
});

export const GET = process.env.NEXT_STATIC_EXPORT
  ? search.staticGET
  : search.GET;
