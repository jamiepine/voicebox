export interface ChangelogEntry {
  version: string;
  date: string | null;
  body: string;
}

/**
 * Parses a Keep-a-Changelog style markdown string into structured entries.
 *
 * Splits on `## [version]` headings and extracts the version + date from each.
 * The body is the raw markdown between headings (trimmed), with the leading
 * `# Changelog` title and trailing link references stripped.
 */
export function parseChangelog(raw: string): ChangelogEntry[] {
  const entries: ChangelogEntry[] = [];

  // Strip trailing link reference definitions (e.g. [0.1.0]: https://...)
  const cleaned = raw.replace(/^\[[\w.]+\]:.*$/gm, '').trimEnd();

  // Match `## [version]` or `## [version] - date`
  const headingRe = /^## \[(.+?)\](?:\s*-\s*(.+))?$/gm;
  const matches = [...cleaned.matchAll(headingRe)];

  for (let i = 0; i < matches.length; i++) {
    const match = matches[i];
    const version = match[1];
    const date = match[2]?.trim() || null;

    const start = match.index! + match[0].length;
    const end = i + 1 < matches.length ? matches[i + 1].index! : cleaned.length;
    const body = cleaned.slice(start, end).trim();

    entries.push({ version, date, body });
  }

  return entries;
}
