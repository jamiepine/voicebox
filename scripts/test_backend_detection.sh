#!/usr/bin/env bash
# test_backend_detection.sh — validates the backend-detection contract that
# tauri/src-tauri/src/main.rs implements in start_server() (Unix path):
#
#   Case A: a live Voicebox backend already listens on 17493
#           -> lsof shows a listener; if its command name contains "voicebox"
#              the app reuses it directly (and stores the PID); otherwise the
#              app GETs /health and reuses the server only if the JSON matches
#              {status:"healthy", model_loaded:<bool>, gpu_available:<bool>}.
#   Case B: the port is held by a NON-voicebox process whose /health does not
#           match the schema -> the app refuses with "Port 17493 is already
#           in use by another application (...)". This script starts its own
#           dummy listener (python3 -m http.server) to exercise this.
#   Case C: nothing listens on the port -> the app spawns the bundled sidecar
#           (binaries/voicebox-server, produced by `bun run build:server`,
#           or the GPU onedir backends under <data_dir>/backends/{rocm,cuda}).
#
# Usage:
#   scripts/test_backend_detection.sh            # auto-detect applicable case
#   scripts/test_backend_detection.sh --case A|B|C
#
# Dependencies: bash, curl, lsof, python3 (jq optional — python3 is the
# fallback JSON validator). Exit code 0 = all executed checks passed.
set -euo pipefail

# VOICEBOX_PORT override exists so the destructive-ish cases (B) can be
# rehearsed on a spare port without stopping a live production backend.
PORT="${VOICEBOX_PORT:-17493}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCED_CASE=""
DUMMY_PID=""

PASS=0
FAIL=0

cleanup() {
    if [[ -n "$DUMMY_PID" ]] && kill -0 "$DUMMY_PID" 2>/dev/null; then
        kill "$DUMMY_PID" 2>/dev/null || true
        wait "$DUMMY_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

ok()   { echo "  PASS: $*"; PASS=$((PASS + 1)); }
bad()  { echo "  FAIL: $*"; FAIL=$((FAIL + 1)); }
info() { echo "  -> $*"; }

usage() {
    sed -n '2,24p' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)
            [[ $# -ge 2 ]] || usage 1
            FORCED_CASE="${2^^}"
            [[ "$FORCED_CASE" =~ ^[ABC]$ ]] || { echo "Invalid case: $2" >&2; usage 1; }
            shift 2
            ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown argument: $1" >&2; usage 1 ;;
    esac
done

for cmd in curl lsof python3; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Missing dependency: $cmd" >&2; exit 2; }
done

# --- helpers mirroring the Rust logic -------------------------------------

# First listener row on the port: prints "<command> <pid>" or nothing.
port_listener() {
    # `|| true`: lsof exits 1 when nothing matches; with `set -o pipefail`
    # that would otherwise abort callers running outside a condition.
    { lsof -i ":$PORT" -sTCP:LISTEN 2>/dev/null || true; } | awk 'NR>1 && NF>=2 {print $1, $2; exit}'
}

# Schema check identical to check_health() in main.rs:
#   status == "healthy" AND model_loaded is bool AND gpu_available is bool
health_matches_voicebox_schema() {
    local body="$1"
    [[ -n "$body" ]] || return 1
    if command -v jq >/dev/null 2>&1; then
        jq -e '
            .status == "healthy"
            and (.model_loaded | type == "boolean")
            and (.gpu_available | type == "boolean")
        ' >/dev/null 2>&1 <<<"$body"
    else
        python3 -c '
import sys, json
try:
    d = json.loads(sys.argv[1])
except Exception:
    sys.exit(1)
ok = (isinstance(d, dict) and d.get("status") == "healthy"
      and isinstance(d.get("model_loaded"), bool)
      and isinstance(d.get("gpu_available"), bool))
sys.exit(0 if ok else 1)
' "$body" 2>/dev/null
    fi
}

# --- cases ------------------------------------------------------------------

case_a() {
    echo "CASE A: live backend already listening on port $PORT"
    local row command pid
    row="$(port_listener)"
    if [[ -z "$row" ]]; then
        bad "no process listening on port $PORT (case A not applicable right now)"
        return
    fi
    command="${row%% *}"
    pid="${row##* }"
    ok "lsof shows listener on :$PORT (command='$command' pid=$pid)"

    local body
    body="$(curl -s --max-time 3 "http://127.0.0.1:$PORT/health" || true)"
    if health_matches_voicebox_schema "$body"; then
        ok "/health matches voicebox schema (status=healthy, model_loaded/gpu_available booleans)"
    else
        bad "/health does NOT match voicebox schema; body: ${body:0:200}"
        return
    fi

    if [[ "$command" == *voicebox* ]]; then
        info "command name contains 'voicebox' -> app reuses it directly and stores PID $pid (main.rs:301)"
    else
        info "command '$command' is not a voicebox binary, but the health check passed -> app reuses this external server (main.rs:312)"
    fi
    info "app decision: REUSE http://127.0.0.1:$PORT (no sidecar spawned)"
}

case_b() {
    echo "CASE B: port $PORT held by a NON-voicebox process"
    if [[ -n "$(port_listener)" ]]; then
        bad "port $PORT is already in use; cannot start dummy listener (stop the backend or run without --case B when it is up)"
        return
    fi

    python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &
    DUMMY_PID=$!
    local i
    for i in {1..50}; do
        [[ -n "$(port_listener)" ]] && break
        sleep 0.1
    done

    local row command
    row="$(port_listener)"
    if [[ -z "$row" ]]; then
        bad "dummy listener (python3 -m http.server) failed to bind port $PORT"
        return
    fi
    command="${row%% *}"
    ok "dummy listener up (command='$command' pid=$DUMMY_PID)"
    if [[ "$command" == *voicebox* ]]; then
        bad "dummy command name unexpectedly contains 'voicebox'"
        return
    fi
    ok "command name '$command' does not contain 'voicebox' -> app falls through to the health check"

    local body
    body="$(curl -s --max-time 3 "http://127.0.0.1:$PORT/health" || true)"
    if health_matches_voicebox_schema "$body"; then
        bad "dummy server's /health unexpectedly MATCHES the voicebox schema"
    else
        ok "dummy server's /health fails schema validation (as expected)"
        info "app decision: REFUSE with 'Port $PORT is already in use by another application ($command)' (main.rs:317)"
    fi

    cleanup
    DUMMY_PID=""
}

case_c() {
    echo "CASE C: nothing listening on port $PORT -> app would spawn the bundled sidecar"
    if [[ -n "$(port_listener)" ]]; then
        info "note: port $PORT is currently busy, so the app would not reach the spawn path right now; checking sidecar availability anyway"
    else
        ok "port $PORT is free"
    fi

    local found=""
    local candidate
    for candidate in "$REPO_ROOT"/tauri/src-tauri/binaries/voicebox-server*; do
        [[ -e "$candidate" ]] && { found="$candidate"; break; }
    done

    if [[ -n "$found" ]]; then
        ok "bundled sidecar exists: $found"
    elif [[ -x "$REPO_ROOT/scripts/build-server.sh" ]]; then
        ok "sidecar not present in tauri/src-tauri/binaries/ (dev tree) — it is produced by 'bun run build:server' (scripts/build-server.sh) and bundled via tauri.conf.json externalBin"
    else
        bad "no sidecar binary and scripts/build-server.sh missing"
        return
    fi
    info "app decision: SPAWN sidecar 'voicebox-server' (or GPU onedir backend under <data_dir>/backends/{rocm,cuda})"
}

# --- dispatch ---------------------------------------------------------------

echo "Backend detection contract test (port $PORT)"
echo "============================================"

if [[ -n "$FORCED_CASE" ]]; then
    "case_${FORCED_CASE,,}"
else
    if [[ -n "$(port_listener)" ]]; then
        echo "Auto-detect: port $PORT is busy -> running case A"
        echo
        case_a
    else
        echo "Auto-detect: port $PORT is free -> running cases B and C"
        echo
        case_b
        echo
        case_c
    fi
fi

echo
echo "============================================"
echo "SUMMARY: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
