import { source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';

const search = createFromSource(source, {
  // https://docs.orama.com/docs/orama-js/supported-languages
  language: 'english',
});

const isStaticExport = process.env.NEXT_STATIC_EXPORT === 'true';

export const GET = isStaticExport
  ? search.staticGET
  : search.GET;
