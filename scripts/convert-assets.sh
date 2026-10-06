#!/bin/bash
set -e

# Asset Conversion Script
# Converts PNG → WebP and MOV → WebM in public folders
# Deletes original files after successful conversion

cd "$(dirname "$0")/.."

# Directories to process
DIRS=(
  "docs/public"
)

# Track counts
png_converted=0
mov_converted=0
png_failed=0
mov_failed=0

echo "🔄 Converting assets to web-optimized formats..."
echo ""

# Check for required tools
check_dependencies() {
  local missing=()
  
  if ! command -v cwebp &> /dev/null && ! command -v ffmpeg &> /dev/null; then
    missing+=("cwebp or ffmpeg (for PNG→WebP)")
  fi
  
  if ! command -v ffmpeg &> /dev/null; then
    missing+=("ffmpeg (for MOV→WebM)")
  fi
  
  if [ ${#missing[@]} -ne 0 ]; then
    echo "❌ Missing required tools:"
    for tool in "${missing[@]}"; do
      echo "   - $tool"
    done
    echo ""
    echo "Install with: brew install webp ffmpeg"
    exit 1
  fi
}

# Convert PNG to WebP
convert_png() {
  local input="$1"
  local output="${input%.png}.webp"
  
  echo "  Converting: $input"
  
  if command -v cwebp &> /dev/null; then
    # Use cwebp for best quality/size ratio
    if cwebp -q 90 "$input" -o "$output" 2>/dev/null; then
      rm "$input"
      echo "    ✓ → $output"
      return 0
    fi
  elif command -v ffmpeg &> /dev/null; then
    # Fallback to ffmpeg
    if ffmpeg -i "$input" -c:v libwebp -quality 90 "$output" -y 2>/dev/null; then
      rm "$input"
      echo "    ✓ → $output"
      return 0
    fi
  fi
  
  echo "    ✗ Failed to convert"
  return 1
}

# Convert MOV to WebM
convert_mov() {
  local input="$1"
  local output="${input%.mov}.webm"
  
  echo "  Converting: $input"
  
  if ffmpeg -i "$input" -c:v libvpx-vp9 -crf 30 -b:v 0 -c:a libopus "$output" -y 2>/dev/null; then
    rm "$input"
    echo "    ✓ → $output"
    return 0
  fi
  
  echo "    ✗ Failed to convert"
  return 1
}

# Main execution
check_dependencies

for dir in "${DIRS[@]}"; do
  if [ ! -d "$dir" ]; then
    echo "⚠ Directory not found: $dir (skipping)"
    continue
  fi
  
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "📁 Processing: $dir"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  # Find and convert PNGs (recursively)
  while IFS= read -r -d '' file; do
    if convert_png "$file"; then
      ((png_converted++))
    else
      ((png_failed++))
    fi
  done < <(find "$dir" -type f -name "*.png" -print0 2>/dev/null)
  
  # Find and convert MOVs (recursively)
  while IFS= read -r -d '' file; do
    if convert_mov "$file"; then
      ((mov_converted++))
    else
      ((mov_failed++))
    fi
  done < <(find "$dir" -type f -name "*.mov" -print0 2>/dev/null)
  
  # Also check for uppercase extensions
  while IFS= read -r -d '' file; do
    if convert_png "$file"; then
      ((png_converted++))
    else
      ((png_failed++))
    fi
  done < <(find "$dir" -type f -name "*.PNG" -print0 2>/dev/null)
  
  while IFS= read -r -d '' file; do
    if convert_mov "$file"; then
      ((mov_converted++))
    else
      ((mov_failed++))
    fi
  done < <(find "$dir" -type f -name "*.MOV" -print0 2>/dev/null)
  
  echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Conversion complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results:"
echo "  PNG → WebP: $png_converted converted, $png_failed failed"
echo "  MOV → WebM: $mov_converted converted, $mov_failed failed"

if [ $png_converted -eq 0 ] && [ $mov_converted -eq 0 ]; then
  echo ""
  echo "No PNG or MOV files found to convert."
fi
