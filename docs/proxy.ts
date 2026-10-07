import { createI18nMiddleware } from 'fumadocs-core/i18n/middleware';
import { i18n } from '@/lib/i18n';

export default createI18nMiddleware(i18n);

export const config = {
  // Run on every path except API, Next internals, static assets and special routes.
  matcher: [
    '/((?!api|_next/static|_next/image|favicon.ico|images|og|llms-full.txt|robots.txt|sitemap.xml).*)',
  ],
};
