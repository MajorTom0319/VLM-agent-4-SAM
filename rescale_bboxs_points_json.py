# use for unified lift data

# 将 4 倍降采样坐标（bbox/points）恢复到原图尺寸坐标的脚本
# 用法：
#   python scripts\rescale_bboxes_points_json.py ^
#       --in  "C:\path\to\bboxes_frame_00001_gemini-3.1-pro-preview.json" ^
#       --out "C:\path\to\bboxes_frame_00001_gemini-3.1-pro-preview.up4.json" ^
#       --target_width 986 --target_height 728
#
# 或批量处理：
#   python scripts\rescale_bboxes_points_json.py --in_dir "C:\...\outputs" --out_dir "C:\...\outputs_up4" --target_width 986 --target_height 728
#
# 说明：
# - 脚本假设 JSON 里的 bbox / bbox_tight / pos_points / neg_points 坐标基于“降采样后”的图像尺寸。
# - 恢复到原图：坐标 *= scale，image_size.width/height 也 *= scale。
# - 坐标会做 round 并 clamp 到 [0, width/height] 范围内（bbox 的 x2/y2 允许等于 width/height）。
# - bbox 格式： [x1, y1, x2, y2]

# filepath: \home\majortom\majortom\scripts\rescale_bboxes_points_json.py
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple
 

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _scale_point_xy(pt: List[float], sx: float, sy: float) -> List[int]:
    if not (isinstance(pt, list) and len(pt) == 2):
        raise ValueError(f"Invalid point (expect [x,y]): {pt}")
    x, y = pt
    return [int(round(x * sx)), int(round(y * sy))]


def _scale_bbox_xy(b: List[float], sx: float, sy: float) -> List[int]:
    if not (isinstance(b, list) and len(b) == 4):
        raise ValueError(f"Invalid bbox (expect [x1,y1,x2,y2]): {b}")
    x1, y1, x2, y2 = b
    return [
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    ]


def _fix_and_clip_bbox(b: List[int], W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = b
    # allow x2==W, y2==H
    x1 = _clamp(x1, 0, W)
    x2 = _clamp(x2, 0, W)
    y1 = _clamp(y1, 0, H)
    y2 = _clamp(y2, 0, H)

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _clip_point(p: List[int], W: int, H: int) -> List[int]:
    x, y = p
    # points should be within pixel indices [0..W-1],[0..H-1]
    x = _clamp(x, 0, W - 1)
    y = _clamp(y, 0, H - 1)
    return [x, y]


def rescale_json(
    data: Dict[str, Any],
    scale: float,
    target_size: Optional[Tuple[int, int]] = None,  # (W,H)
) -> Dict[str, Any]:
    if "image_size" not in data or "width" not in data["image_size"] or "height" not in data["image_size"]:
        raise ValueError("JSON missing image_size.width/height")

    w_small = int(data["image_size"]["width"])
    h_small = int(data["image_size"]["height"])

    if w_small <= 0 or h_small <= 0:
        raise ValueError(f"Invalid image_size in JSON: {w_small}x{h_small}")

    if target_size is not None:
        W, H = target_size
        if W <= 0 or H <= 0:
            raise ValueError(f"Invalid target size: {W}x{H}")
        sx = W / float(w_small)
        sy = H / float(h_small)
        meta = {
            "scaled_from": {"width": w_small, "height": h_small},
            "target_size": {"width": W, "height": H},
            "scale_factor_applied": None,
            "scale_xy_applied": {"sx": sx, "sy": sy},
        }
    else:
        # fallback: uniform factor
        W = int(round(w_small * scale))
        H = int(round(h_small * scale))
        sx = float(scale)
        sy = float(scale)
        meta = {
            "scaled_from": {"width": w_small, "height": h_small},
            "scale_factor_applied": scale,
            "scale_xy_applied": {"sx": sx, "sy": sy},
        }

    out = dict(data)
    out["image_size"] = {"width": W, "height": H}
    out.update(meta)

    bboxes = out.get("bboxes", [])
    if not isinstance(bboxes, list):
        raise ValueError("JSON field bboxes must be a list")

    for item in bboxes:
        if not isinstance(item, dict):
            continue

        if "bbox" in item:
            item["bbox"] = _fix_and_clip_bbox(_scale_bbox_xy(item["bbox"], sx, sy), W, H)

        if "bbox_tight" in item:
            item["bbox_tight"] = _fix_and_clip_bbox(_scale_bbox_xy(item["bbox_tight"], sx, sy), W, H)

        if "pos_points" in item and isinstance(item["pos_points"], list):
            item["pos_points"] = [
                _clip_point(_scale_point_xy(p, sx, sy), W, H) for p in item["pos_points"]
            ]

        if "neg_points" in item and isinstance(item["neg_points"], list):
            item["neg_points"] = [
                _clip_point(_scale_point_xy(p, sx, sy), W, H) for p in item["neg_points"]
            ]

    return out


def _process_file(in_path: str, out_path: str, scale: float, overwrite: bool, target_size: Optional[Tuple[int, int]]) -> None:
    if (not overwrite) and os.path.exists(out_path):
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = rescale_json(data, scale=scale, target_size=target_size)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def _parse_target_size(args) -> Optional[Tuple[int, int]]:
    # accept either --target_width/--target_height
    if args.target_width is None and args.target_height is None:
        return None
    if args.target_width is None or args.target_height is None:
        raise ValueError("Provide both --target_width and --target_height")
    return (int(args.target_width), int(args.target_height))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_file", type=str, default=None, help="Input JSON file")
    ap.add_argument("--out", dest="out_file", type=str, default=None, help="Output JSON file")
    ap.add_argument("--in_dir", type=str, default=None, help="Batch: input dir of JSONs")
    ap.add_argument("--out_dir", type=str, default=None, help="Batch: output dir")
    ap.add_argument("--glob", type=str, default="*.json", help="Batch: filename pattern (default: *.json)")
    ap.add_argument("--scale", type=float, default=4.0, help="Uniform scale factor (used if target size not provided)")
    ap.add_argument("--target_width", type=int, default=None, help="Force output width (original image width)")
    ap.add_argument("--target_height", type=int, default=None, help="Force output height (original image height)")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting outputs")
    args = ap.parse_args()

    if args.scale <= 0:
        raise ValueError("--scale must be > 0")

    target_size = _parse_target_size(args)

    # single file mode
    if args.in_file:
        if not args.out_file:
            root, ext = os.path.splitext(args.in_file)
            suffix = f".to{target_size[0]}x{target_size[1]}" if target_size else f".up{args.scale:g}"
            args.out_file = f"{root}{suffix}{ext}"
        _process_file(args.in_file, args.out_file, args.scale, args.overwrite, target_size)
        print(f"Wrote: {args.out_file}")
        return

    # batch mode
    if args.in_dir:
        if not args.out_dir:
            raise ValueError("Batch mode requires --out_dir")
        import glob

        in_paths = sorted(glob.glob(os.path.join(args.in_dir, args.glob)))
        if not in_paths:
            raise FileNotFoundError(f"No files match: {os.path.join(args.in_dir, args.glob)}")

        os.makedirs(args.out_dir, exist_ok=True)
        for p in in_paths:
            base = os.path.basename(p)
            out_path = os.path.join(args.out_dir, base)
            _process_file(p, out_path, args.scale, args.overwrite, target_size)
        print(f"Processed {len(in_paths)} files into: {args.out_dir}")
        return

    raise ValueError("Provide either --in (single file) or --in_dir (batch)")


if __name__ == "__main__":
    main()