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

function getPublicOrigin(request: NextRequest): string {
  const forwardedHost = request.headers.get('x-forwarded-host');
  const forwardedProto = request.headers.get('x-forwarded-proto');

  if (forwardedHost && forwardedProto) {
    return `${forwardedProto}://${forwardedHost}`;
  }

  return new URL(request.url).origin;
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
