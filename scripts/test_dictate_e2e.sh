#!/usr/bin/env bash
# test_dictate_e2e.sh — end-to-end harness for the dictation toggle script.
#
# NOTE: the script under test is user-local tooling that lives OUTSIDE this
# repo at ~/.local/bin/voicebox-dictate.sh (Super+Alt+V -> record/stop ->
# transcribe via the Voicebox backend -> paste via wl-copy + ydotool).
# This harness tests the INSTALLED copy; it does not modify it.
#
# How it works:
#   * Creates a temp dir with stub executables for every external command the
#     script uses (arecord, curl, wl-copy, wl-paste, ydotool, xclip, xdotool,
#     wtype, notify-send, systemctl) and prepends it to PATH.
#   * Each stub appends its argv to calls.log. The arecord stub stays alive
#     like a real recorder and, on SIGTERM, drops a generated 3 s tone WAV
#     (python3 wave module; the repo ships no WAV fixtures) at the output
#     path. The curl stub answers /health and returns a canned successful
#     /transcribe JSON.
#   * HOME is pointed at a temp dir so the script's real state
#     (~/.cache/voicebox-dictate) is never touched.
#
# Scenarios:
#   happy        run1 starts recording, run2 stops -> transcribe -> clipboard
#                -> ydotool paste; asserts the full chain ran in order, exit 0.
#   fail-record  arecord exits 1 at start -> "No se pudo iniciar la grabación"
#   fail-save    recorder dies without WAV -> "No se guardó el audio"
#   fail-http    curl fails on /transcribe -> "Falló la petición al backend"
#   fail-wlcopy  wl-copy exits 1 -> later stages (ydotool) never run
#   fail-ydotool ydotool exits 1 -> text stays in clipboard fallback
#
# Dependencies: bash, python3. Exit code 0 = all scenarios passed.
set -euo pipefail

DICTATE_SCRIPT="${VOICEBOX_DICTATE_SCRIPT:-$HOME/.local/bin/voicebox-dictate.sh}"
[[ -f "$DICTATE_SCRIPT" ]] || { echo "Dictate script not found: $DICTATE_SCRIPT" >&2; exit 2; }

WORK="$(mktemp -d /tmp/voicebox-dictate-e2e.XXXXXX)"
STUB_DIR="$WORK/stubs"
FAKE_HOME="$WORK/home"
CALLS_LOG="$WORK/calls.log"
CLIP_FILE="$WORK/clipboard.txt"
FIXTURE="$WORK/fixture.wav"
LOG_FILE="$FAKE_HOME/.cache/voicebox-dictate/dictate.log"

PASS=0
FAIL=0

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

ok()      { echo "  PASS: $*"; PASS=$((PASS + 1)); }
bad()     { echo "  FAIL: $*"; FAIL=$((FAIL + 1)); }
section() { echo; echo "== $*"; }

# --- fixture WAV: 3 s of 440 Hz tone, 16 kHz mono s16 (~96 KB > 10000-byte
#     minimum the script validates) -----------------------------------------
python3 - "$FIXTURE" <<'PYEOF'
import math, struct, sys, wave
path = sys.argv[1]
rate, secs = 16000, 3
with wave.open(path, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(rate * secs)
    )
    w.writeframes(frames)
PYEOF

# --- stubs ------------------------------------------------------------------
mkdir -p "$STUB_DIR" "$FAKE_HOME"

cat > "$STUB_DIR/arecord" <<'EOF'
#!/usr/bin/env bash
echo "arecord $*" >> "$CALLS_LOG"
out="${@: -1}"
if [[ "${VB_STUB_ARECORD_FAIL:-0}" == "1" ]]; then
    exit 1
fi
if [[ "${VB_STUB_ARECORD_NOWAV:-0}" == "1" ]]; then
    trap 'exit 0' TERM
else
    trap 'cp "$VB_FIXTURE_WAV" "$out" 2>/dev/null; exit 0' TERM
fi
while :; do sleep 1; done
EOF

cat > "$STUB_DIR/curl" <<'EOF'
#!/usr/bin/env bash
echo "curl $*" >> "$CALLS_LOG"
url=""
ofile=""
prev=""
for a in "$@"; do
    [[ "$prev" == "-o" ]] && ofile="$a"
    [[ "$a" == http* ]] && url="$a"
    prev="$a"
done
if [[ "$url" == */health ]]; then
    echo '{"status":"healthy","model_loaded":true,"gpu_available":true}'
    exit 0
fi
if [[ "$url" == */transcribe ]]; then
    if [[ "${VB_STUB_CURL_FAIL:-0}" == "1" ]]; then
        exit 7
    fi
    [[ -n "$ofile" ]] && printf '%s' '{"text":"hola mundo esto es una prueba"}' > "$ofile"
    printf '200'
    exit 0
fi
exit 0
EOF

cat > "$STUB_DIR/wl-copy" <<'EOF'
#!/usr/bin/env bash
echo "wl-copy $*" >> "$CALLS_LOG"
[[ "${VB_STUB_WLCOPY_FAIL:-0}" == "1" ]] && exit 1
cat > "$CLIP_FILE"
EOF

cat > "$STUB_DIR/wl-paste" <<'EOF'
#!/usr/bin/env bash
echo "wl-paste $*" >> "$CALLS_LOG"
cat "$CLIP_FILE" 2>/dev/null || true
EOF

cat > "$STUB_DIR/ydotool" <<'EOF'
#!/usr/bin/env bash
echo "ydotool $*" >> "$CALLS_LOG"
[[ "${VB_STUB_YDOTOOL_FAIL:-0}" == "1" ]] && exit 1
exit 0
EOF

# Deterministic failing fallbacks so wl-copy/ydotool failures cannot leak into
# real xclip/xdotool/wtype binaries present on the host.
for tool in xclip xdotool wtype; do
    cat > "$STUB_DIR/$tool" <<EOF
#!/usr/bin/env bash
echo "$tool \$*" >> "\$CALLS_LOG"
exit 1
EOF
done

for tool in notify-send systemctl; do
    cat > "$STUB_DIR/$tool" <<EOF
#!/usr/bin/env bash
echo "$tool \$*" >> "\$CALLS_LOG"
exit 0
EOF
done

chmod +x "$STUB_DIR"/*

# --- helpers ----------------------------------------------------------------

# Runs the installed dictate script in a fully controlled environment.
run_dictate() {
    local rc=0
    env -i \
        HOME="$FAKE_HOME" \
        PATH="$STUB_DIR:/usr/bin:/bin" \
        WAYLAND_DISPLAY="wayland-vbtest" \
        CALLS_LOG="$CALLS_LOG" \
        CLIP_FILE="$CLIP_FILE" \
        VB_FIXTURE_WAV="$FIXTURE" \
        VB_STUB_ARECORD_FAIL="${VB_STUB_ARECORD_FAIL:-0}" \
        VB_STUB_ARECORD_NOWAV="${VB_STUB_ARECORD_NOWAV:-0}" \
        VB_STUB_CURL_FAIL="${VB_STUB_CURL_FAIL:-0}" \
        VB_STUB_WLCOPY_FAIL="${VB_STUB_WLCOPY_FAIL:-0}" \
        VB_STUB_YDOTOOL_FAIL="${VB_STUB_YDOTOOL_FAIL:-0}" \
        "$DICTATE_SCRIPT" >/dev/null 2>&1 || rc=$?
    return "$rc"
}

reset_scenario() {
    rm -rf "$FAKE_HOME"
    mkdir -p "$FAKE_HOME"
    : > "$CALLS_LOG"
    : > "$CLIP_FILE"
    VB_STUB_ARECORD_FAIL=0
    VB_STUB_ARECORD_NOWAV=0
    VB_STUB_CURL_FAIL=0
    VB_STUB_WLCOPY_FAIL=0
    VB_STUB_YDOTOOL_FAIL=0
}

called()         { grep -q "^$1 " "$CALLS_LOG"; }
not_called()     { ! grep -q "^$1 " "$CALLS_LOG"; }
notify_has()     { grep -q "^notify-send .*$1" "$CALLS_LOG"; }
script_log_has() { [[ -f "$LOG_FILE" ]] && grep -qF "$1" "$LOG_FILE"; }
first_line()     { grep -n -m1 "$1" "$CALLS_LOG" | cut -d: -f1; }
last_line()      { grep -n "$1" "$CALLS_LOG" | tail -1 | cut -d: -f1; }

# --- scenarios ----------------------------------------------------------------

echo "Dictate E2E harness"
echo "script under test: $DICTATE_SCRIPT"
echo "workdir: $WORK"

section "happy path: record -> transcribe -> wl-copy -> ydotool"
reset_scenario

rc=0; run_dictate || rc=$?
if [[ "$rc" -eq 0 ]] && called arecord; then
    ok "first toggle started recording (exit 0, arecord spawned)"
else
    bad "first toggle: exit=$rc, arecord called=$(called arecord && echo yes || echo no)"
fi

rc=0; run_dictate || rc=$?
[[ "$rc" -eq 0 ]] && ok "second toggle finished OK (exit 0)" || bad "second toggle exit=$rc"

a="$(last_line '^arecord')"
t="$(first_line '/transcribe')"
c="$(first_line '^wl-copy')"
y="$(first_line '^ydotool')"
if [[ -n "$a" && -n "$t" && -n "$c" && -n "$y" && "$a" -lt "$t" && "$t" -lt "$c" && "$c" -lt "$y" ]]; then
    ok "chain ran in order: arecord(L$a) -> /transcribe(L$t) -> wl-copy(L$c) -> ydotool(L$y)"
else
    bad "chain order wrong: arecord=${a:-none} transcribe=${t:-none} wl-copy=${c:-none} ydotool=${y:-none}"
fi

grep -Fq 'ydotool key ctrl+v' "$CALLS_LOG" \
    && ok "ydotool invoked as 'key ctrl+v'" \
    || bad "ydotool invocation missing/wrong: $(grep '^ydotool' "$CALLS_LOG" || echo none)"
script_log_has "Pasted text with ydotool (Ctrl+V)" \
    && ok "script log confirms paste" \
    || bad "script log missing 'Pasted text with ydotool'"
notify_has "Dictado listo" \
    && ok "user notified: 'Dictado listo'" \
    || bad "missing 'Dictado listo' notification"

section "failure injection: recorder fails to start"
reset_scenario
VB_STUB_ARECORD_FAIL=1
rc=0; run_dictate || rc=$?
[[ "$rc" -eq 1 ]] && ok "aborts with exit 1" || bad "expected exit 1, got $rc"
notify_has "No se pudo iniciar la grabación" \
    && ok "documented error: 'No se pudo iniciar la grabación'" \
    || bad "missing recorder-error notification"
not_called wl-copy && ok "later stages never called" || bad "wl-copy ran despite recorder failure"

section "failure injection: recording leaves no WAV"
reset_scenario
VB_STUB_ARECORD_NOWAV=1
run_dictate >/dev/null 2>&1 || true   # start (recorder stays up, writes nothing on TERM)
rc=0; run_dictate || rc=$?
[[ "$rc" -eq 1 ]] && ok "aborts with exit 1" || bad "expected exit 1, got $rc"
notify_has "No se guardó el audio" \
    && ok "documented error: 'No se guardó el audio'" \
    || bad "missing save-error notification"
if not_called ydotool && ! grep -q '/transcribe' "$CALLS_LOG"; then
    ok "transcribe/paste stages never called"
else
    bad "later stages ran despite missing WAV"
fi

section "failure injection: /transcribe request fails"
reset_scenario
VB_STUB_CURL_FAIL=1
run_dictate >/dev/null 2>&1 || true
rc=0; run_dictate || rc=$?
[[ "$rc" -eq 1 ]] && ok "aborts with exit 1" || bad "expected exit 1, got $rc"
notify_has "Falló la petición al backend" \
    && ok "documented error: 'Falló la petición al backend'" \
    || bad "missing http-error notification"
not_called wl-copy && ok "wl-copy never called" || bad "wl-copy ran despite transcribe failure"
not_called ydotool && ok "ydotool never called" || bad "ydotool ran despite transcribe failure"

section "failure injection: wl-copy fails"
reset_scenario
VB_STUB_WLCOPY_FAIL=1
run_dictate >/dev/null 2>&1 || true
rc=0; run_dictate || rc=$?
# Documented behavior: type_text() fails -> error notification, but the
# script still exits 0 after cleaning up the WAV.
[[ "$rc" -eq 0 ]] && ok "script exits 0 (documented: error is notified, not fatal)" || bad "expected exit 0, got $rc"
script_log_has "wl-copy falló" \
    && ok "script log: 'wl-copy falló'" \
    || bad "script log missing 'wl-copy falló'"
notify_has "No se pudo escribir ni copiar el texto" \
    && ok "documented error: 'No se pudo escribir ni copiar el texto'" \
    || bad "missing clipboard-error notification"
not_called ydotool && ok "ydotool (later stage) never called" || bad "ydotool ran despite wl-copy failure"

section "failure injection: ydotool fails"
reset_scenario
VB_STUB_YDOTOOL_FAIL=1
run_dictate >/dev/null 2>&1 || true
rc=0; run_dictate || rc=$?
# Documented behavior: text stays in the clipboard, user told to paste
# manually, type_text() still returns success -> exit 0.
[[ "$rc" -eq 0 ]] && ok "script exits 0 (documented clipboard fallback)" || bad "expected exit 0, got $rc"
called wl-copy && ok "wl-copy ran (text left in clipboard)" || bad "wl-copy did not run"
script_log_has "ydotool paste failed, leaving text in clipboard" \
    && ok "script log: 'ydotool paste failed, leaving text in clipboard'" \
    || bad "script log missing ydotool-failure entry"
notify_has "Texto en portapapeles" \
    && ok "documented fallback: 'Texto en portapapeles' (paste manually)" \
    || bad "missing clipboard-fallback notification"

echo
echo "============================================"
echo "SUMMARY: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
