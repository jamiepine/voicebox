#!/usr/bin/env bash
# Print the CHANGELOG.md section for one version, without its heading.
# Usage: scripts/changelog-section.sh <version> [changelog-file]
set -euo pipefail

version="${1:?usage: changelog-section.sh <version> [changelog-file]}"
changelog="${2:-CHANGELOG.md}"

awk -v heading="## [${version}]" '
  index($0, heading) == 1 { found = 1; next }
  found && /^## \[/ { exit }
  found { lines[++n] = $0 }
  END {
    first = 1
    while (first <= n && lines[first] ~ /^[[:space:]]*$/) first++
    last = n
    while (last >= first && lines[last] ~ /^[[:space:]]*$/) last--
    for (i = first; i <= last; i++) print lines[i]
  }
' "$changelog"
