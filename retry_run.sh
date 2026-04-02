#!/usr/bin/env bash

# IMAGE_DIR="${1:-/home/majortom/majortom/datasets/LERF_MASK/$scene/images_4}"
# OUTPUT_ROOT="${2:-$(dirname "$IMAGE_DIR")}"
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# while true; do
#     echo "============================================================"
#     echo "Starting/Restarting run.sh..."
    
#     # 执行原始脚本
#     bash "$SCRIPT_DIR/run.sh" "$IMAGE_DIR" "$OUTPUT_ROOT"
#     EXIT_CODE=$?

#     # 检查退出状态码
#     if [ $EXIT_CODE -eq 0 ]; then
#         echo "run.sh completed successfully. All done."
#         break
#     else
#         echo "run.sh exited with error (code $EXIT_CODE). Restarting in 3 seconds..."
#         sleep 3
#     fi
# done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-/home/majortom/majortom/datasets/nerf_llff_data}"

SCENES=(
    fern
    flower
    fortress
    horns
    leaves
    orchids
    trex
)

for scene in "${SCENES[@]}"; do
    IMAGE_DIR="$DATA_ROOT/$scene/images_4"
    OUTPUT_ROOT="$(dirname "$IMAGE_DIR")"

    echo "############################################################"
    echo "Processing scene: $scene"
    echo "IMAGE_DIR: $IMAGE_DIR"
    echo "OUTPUT_ROOT: $OUTPUT_ROOT"

    while true; do
        echo "============================================================"
        echo "Starting/Restarting run.sh for scene: $scene"

        bash "$SCRIPT_DIR/run.sh" "$IMAGE_DIR" "$OUTPUT_ROOT"
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "run.sh completed successfully for scene: $scene"
            break
        else
            echo "run.sh exited with error (code $EXIT_CODE) for scene: $scene. Restarting in 3 seconds..."
            sleep 3
        fi
    done
done

echo "All scenes completed."