import { type NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

// Pretty URLs from README / docs (e.g. /download/mac-arm) are kept for
// compatibility, but we now always route through the /download page so users
// see context + a donate prompt + resources while the download kicks off.
// The page handles the actual file trigger itself — no more silent redirects
// to GitHub or direct asset URLs.
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
    // Behind reverse proxies/CDNs, request.url can be an internal origin
    // (for example localhost:8080). Prefer forwarded headers so redirects
    // keep users on the public domain.
    return `${forwardedProto}://${forwardedHost}`;
  }

  return new URL(request.url).origin;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ platform: string }> },
) {
  const origin = getPublicOrigin(request);
  const { platform } = await params;
  // No prebuilt Linux binary yet — send straight to the build-from-source page.
  if (platform === 'linux') {
    return NextResponse.redirect(new URL('/linux-install', origin), 307);
  }
  const normalized = PLATFORM_ALIAS[platform];
  const target = new URL('/download', origin);
  if (normalized) target.searchParams.set('platform', normalized);
  return NextResponse.redirect(target, 307);
}
