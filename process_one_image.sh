#!/usr/bin/env bash
set -euo pipefail

IMG="$1"
OUTPUT_ROOT="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stem="$(basename "$IMG")"
stem="${stem%.*}"

out_pt="$OUTPUT_ROOT/VLM_instances_masks_pt/instances_sam_masks_${stem}.pt"
if [ -f "$out_pt" ]; then
  echo "Skip (already exists): $out_pt"
  exit 0
fi

echo "[1/2] VLM bboxes for: $IMG"
python "$SCRIPT_DIR/run_one_image.py" --i "$IMG" 
    # --output_root "$OUTPUT_ROOT"

bbox_json="$(ls -1t "$OUTPUT_ROOT/outputs/bboxes_${stem}_"*.json | head -n 1)"
echo "[2/2] SAM refine for: $IMG"
python "$SCRIPT_DIR/sam_segment_from_box.py" --img "$IMG" --bbox_json "$bbox_json" --output_root "$OUTPUT_ROOT"

echo "Done: $stem"