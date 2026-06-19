#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Python version drift guard.
#
# /.python-version is the single source of truth for the project's Python
# minor version. This script asserts every other place that pins a Python
# version agrees with it, so the justfile, Dockerfile, pyproject and CI can
# never silently drift apart again. Run locally with `just check-py-version`
# or `bash scripts/check-python-version.sh`; it also runs in CI.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

CANON="$(tr -d '[:space:]' < .python-version)"          # e.g. 3.12
MAJOR="${CANON%%.*}"                                     # 3

fail=0
ok()   { printf '  \033[32mok\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }

# Assert that every value piped in equals CANON. $1 is a human label.
check() {
    local label="$1" found="" v matched=1
    while IFS= read -r v; do
        [ -z "$v" ] && continue
        found="$found $v"
        [ "$v" = "$CANON" ] || matched=0
    done
    if [ -z "$found" ]; then
        bad "$label — no version reference found (pattern moved?)"
    elif [ "$matched" = 1 ]; then
        ok "$label →$found"
    else
        bad "$label →$found  (expected $CANON)"
    fi
}

echo "Canonical Python version (.python-version): $CANON"
echo "Checking references…"

# NOTE: each check() reads via process substitution (not a pipe) so the
# fail flag it sets lands in this shell rather than a pipe subshell.

# justfile: python_version := "3.12"
check "justfile python_version" \
    < <(grep -oE 'python_version := "[0-9]+\.[0-9]+"' justfile | grep -oE '[0-9]+\.[0-9]+')

# Dockerfile: ARG PYTHON_VERSION=3.12
check "Dockerfile ARG PYTHON_VERSION" \
    < <(grep -oE 'ARG PYTHON_VERSION=[0-9]+\.[0-9]+' Dockerfile | grep -oE '[0-9]+\.[0-9]+')

# pyproject: requires-python = ">=3.12"  (floor must match)
check "pyproject requires-python floor" \
    < <(grep -oE 'requires-python = ">=[0-9]+\.[0-9]+"' backend/pyproject.toml | grep -oE '[0-9]+\.[0-9]+')

# pyproject ruff target-version: "py312"  → normalise to 3.12
check "pyproject ruff target-version" \
    < <(grep -oE 'target-version = "py[0-9]+"' backend/pyproject.toml | grep -oE 'py[0-9]+' | sed -E "s/py${MAJOR}/${MAJOR}./")

# CI workflows: python-version: "3.12"  (skip ${{ matrix }} expressions)
check "CI python-version literals" \
    < <(grep -rhoE 'python-version: "[0-9]+\.[0-9]+"' .github/workflows | grep -oE '[0-9]+\.[0-9]+' | sort -u)

echo
if [ "$fail" = 0 ]; then
    echo "All Python version references match $CANON."
else
    echo "Python version drift detected — update the offending file(s) to $CANON (or bump .python-version everywhere)." >&2
    exit 1
fi
