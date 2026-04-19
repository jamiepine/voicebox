import { NextResponse } from 'next/server';
import { getStarCount } from '@/lib/releases';

export const dynamic = 'force-dynamic';
export const revalidate = 600;

export async function GET() {
  try {
    const count = await getStarCount();
    return NextResponse.json({ count });
  } catch (error) {
    console.error('Error fetching star count:', error);
    return NextResponse.json({ error: 'Failed to fetch star count' }, { status: 500 });
  }
}
