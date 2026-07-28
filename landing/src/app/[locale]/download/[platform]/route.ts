import { type NextRequest, NextResponse } from 'next/server';
import { getLocalizedPath, isSecondaryLocale } from '@/lib/i18n';

export const dynamic = 'force-dynamic';

const PLATFORM_ALIAS: Record<string, string> = {
  'mac-arm': 'macArm',
  macArm: 'macArm',
  'mac-intel': 'macIntel',
  macIntel: 'macIntel',
  windows: 'windows',
};

function readForwardedHeaderValue(headerValue: string | null): string | null {
  if (!headerValue) return null;
  const firstValue = headerValue.split(',')[0]?.trim();
  return firstValue || null;
}

function isValidForwardedHost(host: string): boolean {
  try {
    const candidate = new URL(`https://${host}`);
    return (
      candidate.host === host &&
      candidate.username === '' &&
      candidate.password === '' &&
      candidate.pathname === '/' &&
      candidate.search === '' &&
      candidate.hash === ''
    );
  } catch {
    return false;
  }
}

function getPublicOrigin(request: NextRequest): string {
  const fallbackOrigin = new URL(request.url).origin;
  const forwardedHost = readForwardedHeaderValue(request.headers.get('x-forwarded-host'));
  const forwardedProto = readForwardedHeaderValue(request.headers.get('x-forwarded-proto'));

  if (
    forwardedHost &&
    forwardedProto &&
    (forwardedProto === 'http' || forwardedProto === 'https') &&
    isValidForwardedHost(forwardedHost)
  ) {
    return `${forwardedProto}://${forwardedHost}`;
  }

  return fallbackOrigin;
}

export async function GET(
  request: NextRequest,
  {
    params,
  }: {
    params: Promise<{ locale: string; platform: string }>;
  },
) {
  const origin = getPublicOrigin(request);
  const { locale, platform } = await params;

  if (!isSecondaryLocale(locale)) {
    return NextResponse.redirect(new URL('/', origin), 307);
  }

  if (platform === 'linux') {
    return NextResponse.redirect(new URL(getLocalizedPath(locale, '/linux-install'), origin), 307);
  }

  const normalized = PLATFORM_ALIAS[platform];
  const target = new URL(getLocalizedPath(locale, '/download'), origin);
  if (normalized) target.searchParams.set('platform', normalized);
  return NextResponse.redirect(target, 307);
}
