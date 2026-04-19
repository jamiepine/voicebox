import changelogRaw from 'virtual:changelog';
import { useMemo, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { type ChangelogEntry, parseChangelog } from '@/lib/utils/parseChangelog';

function renderMarkdown(md: string): React.ReactNode[] {
  const lines = md.split('\n');
  const elements: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Skip empty lines
    if (line.trim() === '') {
      i++;
      continue;
    }

    // Tables — collect all lines starting with |
    if (line.trim().startsWith('|')) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        tableLines.push(lines[i]);
        i++;
      }
      elements.push(renderTable(tableLines, elements.length));
      continue;
    }

    // Headings
    if (line.startsWith('#### ')) {
      elements.push(
        <h5 key={elements.length} className="text-sm font-medium mt-5 mb-1">
          {inlineMarkdown(line.slice(5))}
        </h5>,
      );
      i++;
      continue;
    }
    if (line.startsWith('### ')) {
      elements.push(
        <h4 key={elements.length} className="text-sm font-medium mt-6 mb-2">
          {inlineMarkdown(line.slice(4))}
        </h4>,
      );
      i++;
      continue;
    }

    // List items — collect consecutive
    if (line.startsWith('- ')) {
      const items: string[] = [];
      while (i < lines.length && lines[i].startsWith('- ')) {
        items.push(lines[i].slice(2));
        i++;
      }
      elements.push(
        <ul key={elements.length} className="space-y-1 my-2">
          {items.map((item, idx) => (
            <li key={idx} className="text-sm text-muted-foreground flex gap-2">
              <span className="text-muted-foreground/50 select-none shrink-0">&bull;</span>
              <span>{inlineMarkdown(item)}</span>
            </li>
          ))}
        </ul>,
      );
      continue;
    }

    // Paragraph
    elements.push(
      <p key={elements.length} className="text-sm text-muted-foreground my-2">
        {inlineMarkdown(line)}
      </p>,
    );
    i++;
  }

  return elements;
}

function renderTable(tableLines: string[], keyBase: number): React.ReactNode {
  const parseRow = (line: string) =>
    line
      .split('|')
      .slice(1, -1)
      .map((c) => c.trim());

  const headers = parseRow(tableLines[0]);
  // Skip separator line (index 1)
  const rows = tableLines.slice(2).map(parseRow);

  return (
    <div key={keyBase} className="overflow-x-auto my-3">
      <table className="text-sm w-full">
        <thead>
          <tr className="border-b">
            {headers.map((h, hIdx) => (
              <th
                key={hIdx}
                className="text-left py-1.5 pr-4 text-muted-foreground font-medium text-xs"
              >
                {inlineMarkdown(h)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIdx) => (
            <tr key={rowIdx} className="border-b border-border/50">
              {row.map((cell, cellIdx) => (
                <td key={cellIdx} className="py-1.5 pr-4 text-muted-foreground">
                  {inlineMarkdown(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function inlineMarkdown(text: string): React.ReactNode {
  // Process inline markdown: bold, code, links
  const parts: React.ReactNode[] = [];
  // Regex matches: **bold**, `code`, [text](url)
  const inlineRe = /\*\*(.+?)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null = inlineRe.exec(text);

  while (match !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    if (match[1] !== undefined) {
      // Bold
      parts.push(
        <strong key={parts.length} className="font-medium text-foreground">
          {match[1]}
        </strong>,
      );
    } else if (match[2] !== undefined) {
      // Code
      parts.push(
        <code key={parts.length} className="px-1 py-0.5 rounded bg-muted text-xs font-mono">
          {match[2]}
        </code>,
      );
    } else if (match[3] !== undefined && match[4] !== undefined) {
      // Link
      parts.push(
        <a
          key={parts.length}
          href={match[4]}
          target="_blank"
          rel="noopener noreferrer"
          className="text-accent hover:underline"
        >
          {match[3]}
        </a>,
      );
    }

    lastIndex = match.index + match[0].length;
    match = inlineRe.exec(text);
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length === 1 ? parts[0] : parts;
}

function ChangelogEntryCard({ entry }: { entry: ChangelogEntry }) {
  const [expanded, setExpanded] = useState(false);
  const content = useMemo(() => renderMarkdown(entry.body), [entry.body]);
  const isLong = entry.body.split('\n').length > 12;

  return (
    <div className="border-b border-border/50 pb-6">
      <div className="flex items-baseline gap-3 mb-3">
        <h3 className="text-xl font-semibold tracking-tight">{entry.version}</h3>
        {entry.date && <span className="text-xs text-muted-foreground">{entry.date}</span>}
        {entry.version === 'Unreleased' && <Badge variant="outline">dev</Badge>}
      </div>

      <div className={isLong && !expanded ? 'max-h-48 overflow-hidden relative' : ''}>
        {content}
        {isLong && !expanded && (
          <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-background to-transparent" />
        )}
      </div>

      {isLong && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-accent hover:underline mt-2"
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  );
}

export function ChangelogPage() {
  const entries = useMemo(() => parseChangelog(changelogRaw), []);

  return (
    <div className="space-y-6 max-w-2xl">
      {entries.map((entry) => (
        <ChangelogEntryCard key={entry.version} entry={entry} />
      ))}
    </div>
  );
}
