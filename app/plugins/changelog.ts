import { readFileSync } from 'node:fs';
import path from 'node:path';
import type { Plugin } from 'vite';

/** Vite plugin that exposes CHANGELOG.md as `virtual:changelog`. */
export function changelogPlugin(repoRoot: string): Plugin {
  const virtualId = 'virtual:changelog';
  const resolvedId = '\0' + virtualId;
  const changelogPath = path.resolve(repoRoot, 'CHANGELOG.md');

  return {
    name: 'changelog',
    resolveId(id) {
      if (id === virtualId) return resolvedId;
    },
    load(id) {
      if (id === resolvedId) {
        const raw = readFileSync(changelogPath, 'utf-8');
        return `export default ${JSON.stringify(raw)};`;
      }
    },
  };
}
