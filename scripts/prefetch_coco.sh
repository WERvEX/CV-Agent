#!/usr/bin/env bash
# Prefetch MS COCO 2017 for Ultralytics / cv_agent (detection).
#
# Usage (on the server, outside a slow Docker download):
#   cd /path/to/cvagent
#   bash scripts/prefetch_coco.sh
#
# Optional env:
#   DATASETS_DIR=/data2/zhiyuanwei/cvagent/datasets   # default: ./datasets
#   LABEL_MIRROR=github|ghfast|ultralytics              # default: github
#   SKIP_IMAGES=1                                       # labels only (~168 MB)
#
# After this finishes, start cv_agent Docker with:
#   -v "$(pwd)/datasets:/app/datasets"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="${DATASETS_DIR:-$ROOT/datasets}"
LABEL_MIRROR="${LABEL_MIRROR:-github}"
LABEL_ZIP="coco2017labels-segments.zip"
COCO_ROOT="$DATASETS_DIR/coco"
IMG_DIR="$COCO_ROOT/images"

label_url() {
  case "$LABEL_MIRROR" in
    github)
      echo "https://github.com/ultralytics/assets/releases/download/v0.0.0/$LABEL_ZIP"
      ;;
    ghfast)
      echo "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v0.0.0/$LABEL_ZIP"
      ;;
    ghproxy)
      echo "https://mirror.ghproxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/$LABEL_ZIP"
      ;;
    ultralytics)
      echo "https://ultralytics.com/assets/$LABEL_ZIP"
      ;;
    *)
      echo "Unknown LABEL_MIRROR=$LABEL_MIRROR (use github, ghfast, ghproxy, ultralytics)" >&2
      exit 1
      ;;
  esac
}

download() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
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

echo "==> datasets dir: $DATASETS_DIR"
mkdir -p "$DATASETS_DIR" "$IMG_DIR"

LABEL_PATH="$DATASETS_DIR/$LABEL_ZIP"
if [[ -f "$LABEL_PATH" && "$(stat -c%s "$LABEL_PATH" 2>/dev/null || stat -f%z "$LABEL_PATH")" -gt 100000000 ]]; then
  echo "==> labels zip already present: $LABEL_PATH"
else
  echo "==> downloading labels ($LABEL_MIRROR mirror) ..."
  download "$(label_url)" "$LABEL_PATH"
fi

if [[ ! -d "$COCO_ROOT/labels" && ! -d "$COCO_ROOT/labels/train2017" ]]; then
  echo "==> extracting labels into $DATASETS_DIR ..."
  unzip -qo "$LABEL_PATH" -d "$DATASETS_DIR"
else
  echo "==> labels already extracted under $COCO_ROOT"
fi

if [[ "${SKIP_IMAGES:-0}" == "1" ]]; then
  echo "==> SKIP_IMAGES=1, done (labels only)."
  exit 0
fi

declare -A IMAGES=(
  ["val2017.zip"]="http://images.cocodataset.org/zips/val2017.zip"
  ["train2017.zip"]="http://images.cocodataset.org/zips/train2017.zip"
)

for zip_name in val2017.zip train2017.zip; do
  url="${IMAGES[$zip_name]}"
  zip_path="$IMG_DIR/$zip_name"
  split_dir="$IMG_DIR/${zip_name%.zip}"
  if [[ -d "$split_dir" && "$(find "$split_dir" -type f | head -1 | wc -l)" -gt 0 ]]; then
    echo "==> images already present: $split_dir"
    continue
  fi
  echo "==> downloading $zip_name (~1G val / ~19G train) ..."
  download "$url" "$zip_path"
  echo "==> extracting $zip_name ..."
  unzip -qo "$zip_path" -d "$IMG_DIR"
  rm -f "$zip_path"
done

echo "==> COCO prefetch complete."
echo "    $COCO_ROOT"
ls -la "$COCO_ROOT" 2>/dev/null || true
