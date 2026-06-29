#!/usr/bin/env bash
# Prefetch Ultralytics YOLO26 weights for cv_agent / Docker.
#
# Includes yolo26n.pt — required by Ultralytics AMP checks even when training yolo26m/s.
#
# Usage:
#   cd /path/to/cvagent
#   bash scripts/prefetch_weights.sh
#
# Optional:
#   WEIGHTS_DIR=./weights          # default: ./weights
#   MIRROR=github|ghfast           # default: ghfast

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT/weights}"
MIRROR="${MIRROR:-ghfast}"
RELEASE="v8.4.0"
FILES=(yolo26n.pt yolo26s.pt yolo26m.pt)

url_for() {
  local file="$1"
  case "$MIRROR" in
    github)
      echo "https://github.com/ultralytics/assets/releases/download/${RELEASE}/${file}"
      ;;
    ghfast)
      echo "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/${RELEASE}/${file}"
      ;;
    *)
      echo "Unknown MIRROR=$MIRROR (use github or ghfast)" >&2
      exit 1
      ;;
  esac
}

download() {
  local url="$1"
  local out="$2"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 16 -s 16 -c -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$out" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -C - -o "$out" "$url"
  else
    echo "Install aria2c, wget, or curl." >&2
    exit 1
  fi
}

mkdir -p "$WEIGHTS_DIR"
echo "==> weights dir: $WEIGHTS_DIR"

# Docker creates empty directories when mounting missing host files — remove before download.
for shadow in "$WEIGHTS_DIR"/*.pt "$ROOT"/*.pt; do
  if [[ -d "$shadow" ]]; then
    echo "==> removing shadow directory (bad prior Docker mount): $shadow"
    rm -rf "$shadow"
  fi
done

for file in "${FILES[@]}"; do
  dest="$WEIGHTS_DIR/$file"
  if [[ -d "$dest" ]]; then
    echo "==> removing shadow directory: $dest"
    rm -rf "$dest"
  fi
  if [[ -f "$dest" ]] && [[ "$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest")" -gt 1000000 ]]; then
    echo "==> skip $file (already present)"
    continue
  fi
  echo "==> downloading $file ..."
  download "$(url_for "$file")" "$dest"
done

echo "==> done. Docker mount (recommended — mount directory, not single files):"
echo "  -v \"$(realpath "$WEIGHTS_DIR"):/app/weights:ro\""
