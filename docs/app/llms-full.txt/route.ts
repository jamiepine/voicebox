import { i18n } from '@/lib/i18n';
import { getLLMText, source } from '@/lib/source';

export const revalidate = false;

export async function GET() {
  const scan = source.getPages(i18n.defaultLanguage).map(getLLMText);
  const scanned = await Promise.all(scan);

  return new Response(scanned.join('\n\n'));
}
