#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash run_folder_pipeline.sh /path/to/images /path/to/output_root
# 也可通过环境变量覆盖：SAM_CKPT, SAM_ARCH, DEVICE, MAX_OBJECTS, POINTS_PER_TYPE, BOX_PAD, MIN_AREA, NMS_IOU

IMAGE_DIR="${1:-/home/majortom/majortom/datasets/desktopobjects_360/desk1_4_VLM/images_4}"
OUTPUT_ROOT="${2:-$(dirname "$IMAGE_DIR")}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "IMAGE_DIR=$IMAGE_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"

shopt -s nullglob
imgs=("$IMAGE_DIR"/*.jpg "$IMAGE_DIR"/*.JPG "$IMAGE_DIR"/*.jpeg "$IMAGE_DIR"/*.JPEG "$IMAGE_DIR"/*.png "$IMAGE_DIR"/*.PNG "$IMAGE_DIR"/*.webp "$IMAGE_DIR"/*.WEBP)

if [ ${#imgs[@]} -eq 0 ]; then
  echo "No images found in: $IMAGE_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/outputs"
mkdir -p "$OUTPUT_ROOT/VLM_instances"
mkdir -p "$OUTPUT_ROOT/VLM_instances_vis"
mkdir -p "$OUTPUT_ROOT/VLM_instances_masks_pt"

for img in "${imgs[@]}"; do
  stem="$(basename "$img")"
  stem="${stem%.*}"

  out_pt="$OUTPUT_ROOT/VLM_instances_masks_pt/instances_sam_masks_${stem}.pt"
  if [ -f "$out_pt" ]; then
    echo "============================================================"
    echo "Skip (already exists): $out_pt"
    continue
  fi
  
  echo "============================================================"
  echo "[1/2] VLM bboxes for: $img"

  python "$SCRIPT_DIR/run_one_image.py" \
    --i "$img" \
    # --output_root "$OUTPUT_ROOT" \

  # run_one_image 会把 json 写到：$OUTPUT_ROOT/outputs/bboxes_${stem}_${model}.json
  # 这里直接按通配符找最新的那一个
  bbox_json="$(ls -1t "$OUTPUT_ROOT/outputs/bboxes_${stem}_"*.json | head -n 1)"

  echo "[2/2] SAM refine for: $img"
  echo "bbox_json=$bbox_json"

  python "$SCRIPT_DIR/sam_segment_from_box.py" \
    --img "$img" \
    --bbox_json "$bbox_json" \
    --output_root "$OUTPUT_ROOT" \

  echo "Done: $stem"
done

echo "All done."