// Fetch latest release information from GitHub
export interface DownloadLinks {
  macArm: string;
  macIntel: string;
  windows: string;
  linux: string;
}

export interface ReleaseInfo {
  version: string;
  downloadLinks: DownloadLinks;
  totalDownloads: number;
}

const GITHUB_REPO = 'jamiepine/voicebox';
const GITHUB_API_BASE = 'https://api.github.com';

// Cache for release info (in-memory cache, resets on server restart)
let cachedReleaseInfo: ReleaseInfo | null = null;
let cacheTimestamp: number = 0;
const CACHE_DURATION = 1000 * 60 * 5; // 5 minutes

// Cache for star count
let cachedStarCount: number | null = null;
let starCacheTimestamp: number = 0;

/**
 * Fetches the latest release from GitHub and extracts download links
 */
export async function getLatestRelease(): Promise<ReleaseInfo> {
  // Return cached data if still valid
  const now = Date.now();
  if (cachedReleaseInfo && now - cacheTimestamp < CACHE_DURATION) {
    return cachedReleaseInfo;
  }

  try {
    const response = await fetch(`${GITHUB_API_BASE}/repos/${GITHUB_REPO}/releases/latest`, {
      cache: 'no-store',
      headers: {
        Accept: 'application/vnd.github.v3+json',
      },
    });

    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.status}`);
    }

    const release = await response.json();
    const version = release.tag_name;
    const assets = release.assets || [];

    // Extract download links based on file patterns
    const downloadLinks: Partial<DownloadLinks> = {};

    for (const asset of assets) {
      const name = asset.name.toLowerCase();
      const url = asset.browser_download_url;

      // Skip signature files and other non-downloadable files
      if (name.endsWith('.sig') || name.endsWith('.json') || name.endsWith('.txt')) {
        continue;
      }

      if ((name.includes('aarch64') || name.includes('arm64')) && name.endsWith('.dmg')) {
        downloadLinks.macArm = url;
      } else if (name.includes('x64') && name.endsWith('.dmg')) {
        downloadLinks.macIntel = url;
      } else if (name.endsWith('.msi')) {
        downloadLinks.windows = url;
      } else if (name.endsWith('.appimage') || name.endsWith('.deb')) {
        downloadLinks.linux = url;
      }
    }

    // Fetch total downloads across ALL releases
    const totalDownloads = await getTotalDownloads();

    // Fallback: construct URLs if not found in assets
    const baseUrl = `https://github.com/${GITHUB_REPO}/releases/download/${version}`;

    const releaseInfo: ReleaseInfo = {
      version,
      totalDownloads,
      downloadLinks: {
        macArm:
          downloadLinks.macArm || `${baseUrl}/Voicebox_${version.replace('v', '')}_aarch64.dmg`,
        macIntel:
          downloadLinks.macIntel || `${baseUrl}/Voicebox_${version.replace('v', '')}_x64.dmg`,
        windows:
          downloadLinks.windows || `${baseUrl}/voicebox_${version.replace('v', '')}_x64_en-US.msi`,
        linux: downloadLinks.linux || `${baseUrl}/voicebox_x86_64-unknown-linux-gnu.AppImage`,
      },
    };

    // Update cache
    cachedReleaseInfo = releaseInfo;
    cacheTimestamp = now;

    return releaseInfo;
  } catch (error) {
    console.error('Failed to fetch latest release:', error);
    throw error;
  }
}

// Cache for total download count
let cachedTotalDownloads: number | null = null;
let downloadsCacheTimestamp: number = 0;

/**
 * Fetches download counts across ALL releases (paginated)
 */
async function getTotalDownloads(): Promise<number> {
  const now = Date.now();
  if (cachedTotalDownloads !== null && now - downloadsCacheTimestamp < CACHE_DURATION) {
    return cachedTotalDownloads;
  }

  let total = 0;
  let page = 1;

  try {
    while (true) {
      const response = await fetch(
        `${GITHUB_API_BASE}/repos/${GITHUB_REPO}/releases?per_page=100&page=${page}`,
        {
          cache: 'no-store',
          headers: { Accept: 'application/vnd.github.v3+json' },
        },
      );

      if (!response.ok) break;

      const releases = await response.json();
      if (!Array.isArray(releases) || releases.length === 0) break;

      for (const release of releases) {
        for (const asset of release.assets || []) {
          total += asset.download_count || 0;
        }
      }

      if (releases.length < 100) break;
      page++;
    }

    cachedTotalDownloads = total;
    downloadsCacheTimestamp = now;
  } catch (error) {
    console.error('Failed to fetch total downloads:', error);
    if (cachedTotalDownloads !== null) return cachedTotalDownloads;
  }

  return total;
}

/**
 * Fetches the star count for the repo from GitHub
 */
export async function getStarCount(): Promise<number> {
  const now = Date.now();
  if (cachedStarCount !== null && now - starCacheTimestamp < CACHE_DURATION) {
    return cachedStarCount;
  }

  try {
    const response = await fetch(`${GITHUB_API_BASE}/repos/${GITHUB_REPO}`, {
      next: { revalidate: 600 },
      headers: {
        Accept: 'application/vnd.github.v3+json',
      },
    });

    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.status}`);
    }

    const repo = await response.json();
    const count = repo.stargazers_count ?? 0;

    cachedStarCount = count;
    starCacheTimestamp = now;

    return count;
  } catch (error) {
    console.error('Failed to fetch star count:', error);
    if (cachedStarCount !== null) return cachedStarCount;
    throw error;
  }
}
