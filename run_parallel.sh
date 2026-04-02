#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DIR="${1:-/home/majortom/majortom/datasets/nerf_llff_data/trex/images_4}"
OUTPUT_ROOT="${2:-$(dirname "$IMAGE_DIR")}"

JOBS="${JOBS:-4}"


echo "IMAGE_DIR=$IMAGE_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "JOBS=$JOBS"

shopt -s nullglob
imgs=("$IMAGE_DIR"/*.jpg "$IMAGE_DIR"/*.JPG "$IMAGE_DIR"/*.jpeg "$IMAGE_DIR"/*.JPEG "$IMAGE_DIR"/*.png "$IMAGE_DIR"/*.PNG "$IMAGE_DIR"/*.webp "$IMAGE_DIR"/*.WEBP)

if [ ${#imgs[@]} -eq 0 ]; then
  echo "No images found in: $IMAGE_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/outputs" \
        "$OUTPUT_ROOT/VLM_instances" \
        "$OUTPUT_ROOT/VLM_instances_vis" \
        "$OUTPUT_ROOT/VLM_instances_masks_pt"

# 并发处理
printf "%s\n" "${imgs[@]}" | xargs -I{} -P "$JOBS" bash "$SCRIPT_DIR/process_one_image.sh" "{}" "$OUTPUT_ROOT"

echo "All done."