import os
import json
import cv2
from vlm_bbox import vlm_get_all_bboxes
import numpy as np
import glob
import argparse

def _color_for_instance(i: int) -> tuple:
    """
    为每个物体生成稳定且区分度较高的 BGR 颜色。
    """
    rng = np.random.default_rng(12345 + int(i))
    bgr = rng.integers(0, 256, size=(3,), dtype=np.uint8)
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

def _draw_points_filled(out, pts, color, r=4):
    """正点：实心点"""
    if not pts:
        return
    for p in pts:
        if not p or len(p) != 2:
            continue
        x, y = map(int, p)
        cv2.circle(out, (x, y), r, color, -1, lineType=cv2.LINE_AA)

def _draw_points_ring(out, pts, color, r=5, thickness=2):
    """负点：圆圈"""
    if not pts:
        return
    for p in pts:
        if not p or len(p) != 2:
            continue
        x, y = map(int, p)
        cv2.circle(out, (x, y), r, color, thickness, lineType=cv2.LINE_AA)
        # 加一个小的中心点，便于看清圆心
        cv2.circle(out, (x, y), 1, color, -1, lineType=cv2.LINE_AA)

def draw_bboxes(image_bgr, bboxes):
    out = image_bgr.copy()

    # 画对象框 + 点提示
    for idx, it in enumerate(bboxes):
        bbox = it.get("bbox")
        label = it.get("label", "obj")
        if not bbox or len(bbox) != 4:
            continue

        color = _color_for_instance(idx)
        
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, label, (x1, max(0, y1 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 同一物体：正负点同色；不同物体：不同颜色
        _draw_points_filled(out, it.get("pos_points", []), color, r=4)          # 正点实心
        _draw_points_ring(out, it.get("neg_points", []), color, r=6, thickness=2)  # 负点圆圈


    # 画背景框（用不同颜色）
    # for it in bg_bbox:
    #     bbox = it.get("bbox")
    #     label = it.get("label", "background")
    #     if not bbox or len(bbox) != 4:
    #         continue
    #     x1, y1, x2, y2 = map(int, bbox)
    #     cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #     cv2.putText(out, label, (x1, max(0, y1 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i", dest="image_path", default=os.environ.get("IMG", ""), help="path to image")
    ap.add_argument("--output_root", default=os.environ.get("OUTPUT_ROOT", ""), help="output root dir")

    args = ap.parse_args()

    # image_path = os.environ.get("IMG", "/home/majortom/majortom/datasets/desktopobjects_360/desk1_4_VLM/images_4/0021.jpg")

    max_objects = int(os.environ.get("MAX_OBJECTS", "15"))
    # max_background = int(os.environ.get("MAX_BACKGROUND", "3"))
    image_path = args.image_path
    
    if not args.output_root:
        # 默认：图片目录的上级
        args.output_root = os.path.dirname(os.path.dirname(os.path.abspath(args.image_path)))

    os.makedirs(args.output_root, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.image_path))[0]
    output_dir = os.path.join(args.output_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # output = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # output_dir  = os.path.join(output, f"outputs_{os.path.basename(image_path).rsplit('.',1)[0]}")

    # os.makedirs(output_dir, exist_ok=True)

    json_pattern = os.path.join(output_dir, f"bboxes_{stem}_*.json")
    existing_jsons = glob.glob(json_pattern)

    if not existing_jsons:
        result = vlm_get_all_bboxes(
            image_path,
            max_objects=max_objects, 
            # max_background=max_background,
            )
        bboxes = result["bboxes"]
        # bg_bbox = result["background_bboxes"]
        model = result["model"]

        # 保存 JSON
        json_path = os.path.join(output_dir, f"bboxes_{stem}_{model}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        # 读取已有 JSON
        json_path = existing_jsons[0]
        with open(json_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        bboxes = result["bboxes"]
        # bg_bbox = result["background_bboxes"]
        model = result["model"]

    img = cv2.imread(image_path)
    vis = draw_bboxes(img, bboxes)

    out_path = os.path.join(output_dir, f"bboxes_vis__{stem}_{model}.png")
    cv2.imwrite(out_path, vis)

    print(f"model: {model}")
    print(f"saved: {json_path}")
    print(f"saved: {out_path}")
    print(f"objects: {len(bboxes)}")

if __name__ == "__main__":
    main()