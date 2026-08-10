#!/usr/bin/env bash
# Generate local, untracked Whisper evaluation fixtures from Common Voice.
#
# Pass a directory containing Common Voice clips and a TSV with
# client-provided transcriptions. The source archive and derived media stay
# outside version control.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/generated"
SOURCE_DIR="${1:-}"
TSV="${2:-${SOURCE_DIR:+${SOURCE_DIR}/validated.tsv}}"

if [[ -z "$SOURCE_DIR" || -z "$TSV" ]]; then
    cat >&2 <<'USAGE'
Usage: generate_fixtures.sh COMMON_VOICE_DIR [validated.tsv]

COMMON_VOICE_DIR must contain Common Voice audio clips and a TSV with at least
path and sentence columns. Obtain the source archive from Mozilla Common Voice
under its current dataset terms.
USAGE
    exit 2
fi

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg is required" >&2; exit 2; }
[[ -d "$SOURCE_DIR" ]] || { echo "source directory not found: $SOURCE_DIR" >&2; exit 2; }
[[ -f "$TSV" ]] || { echo "TSV not found: $TSV" >&2; exit 2; }

mkdir -p "$OUT_DIR"

python3 - "$TSV" "$SOURCE_DIR" "$OUT_DIR/selected.tsv" <<'PY'
import csv
import sys
from pathlib import Path

source_tsv, source_dir, output = sys.argv[1:]
rows = []
with open(source_tsv, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        path = Path(source_dir) / row["path"]
        sentence = row.get("sentence", "").strip()
        if path.is_file() and sentence:
            rows.append((str(path), sentence))
if not rows:
    raise SystemExit("No usable Common Voice rows found")
rows.sort(key=lambda item: item[0])
with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["path", "sentence"])
    writer.writerows(rows)
PY

row_count=$(( $(wc -l < "$OUT_DIR/selected.tsv") - 1 ))
(( row_count > 0 )) || { echo "No usable source clips" >&2; exit 2; }

# Normalize a long deterministic list first. Repeating rows makes it possible
# to construct exact-duration windows even when Common Voice clips are short.
concat_list="$OUT_DIR/concat.txt"
: > "$concat_list"
: > "$OUT_DIR/cv_6m43s.txt"
for i in $(seq 1 160); do
    row_number=$((2 + (i - 1) % row_count))
    row="$(sed -n "${row_number}p" "$OUT_DIR/selected.tsv")"
    clip="${row%%$'\t'*}"
    sentence="${row#*$'\t'}"
    normalized="$OUT_DIR/clip-${i}.wav"
    ffmpeg -hide_banner -loglevel error -y -i "$clip" -ac 1 -ar 16000 "$normalized"
    printf "file '%s'\n" "$normalized" >> "$concat_list"
    printf '%s ' "$sentence" >> "$OUT_DIR/cv_6m43s.txt"
done
printf '\n' >> "$OUT_DIR/cv_6m43s.txt"

# The concat stream is longer than 403 seconds; -t makes all three outputs
# exactly 6:43 (or fails rather than silently producing a short fixture).
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$concat_list" \
    -t 10 -ac 1 -ar 16000 "$OUT_DIR/cv_10s.wav"
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$concat_list" \
    -t 30 -ac 1 -ar 16000 "$OUT_DIR/cv_30s.wav"
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$concat_list" \
    -t 403 -ac 1 -ar 16000 "$OUT_DIR/cv_6m43s.wav"
ffmpeg -hide_banner -loglevel error -y -i "$OUT_DIR/cv_6m43s.wav" "$OUT_DIR/cv_6m43s.flac"
ffmpeg -hide_banner -loglevel error -y -i "$OUT_DIR/cv_6m43s.wav" "$OUT_DIR/cv_6m43s.mp3"

sha256sum "$OUT_DIR"/cv_*.{wav,flac,mp3} > "$OUT_DIR/SHA256SUMS"
printf '%s\n' "Generated from Common Voice source: $SOURCE_DIR" > "$OUT_DIR/README.generated.txt"
cp "$OUT_DIR/selected.tsv" "$OUT_DIR/references.tsv"
