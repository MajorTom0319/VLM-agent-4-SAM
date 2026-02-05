import os
import json
import glob
from typing import List, Tuple

import cv2
import numpy as np
import torch

from vlm_bbox import vlm_get_all_bboxes
from sam_prompted_segment import (
    load_sam,
    segment_instance_from_box,
    tight_bbox_from_mask,
    mask_iou,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _colorize_instances(h: int, w: int, masks: List[np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for m in masks:
        color = rng.integers(0, 255, size=(3,), dtype=np.uint8)
        out[m] = color
    return out


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _list_images(folder: str) -> List[str]:
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    files = sorted(set(files))
    return files


def _run_sam_for_bboxes(
    predictor,
    img_bgr: np.ndarray,
    bboxes: list,
    box_pad: float,
    min_area: int,
    nms_iou: float,
) -> Tuple[List[np.ndarray], List[dict]]:
    masks = []
    meta = []

    for it in bboxes:
        bbox = it.get("bbox")
        label = it.get("label", "obj")
        if not bbox:
            continue

        pos_points = it.get("pos_points")
        neg_points = it.get("neg_points")

        mask, score, used_box = segment_instance_from_box(
            predictor,
            img_bgr,
            bbox,
            pad_ratio=box_pad,
            multimask_output=True,
            pos_points=pos_points,
            neg_points=neg_points,
        )

        tb = tight_bbox_from_mask(mask)
        if tb is None:
            continue

        area = int(mask.sum())
        if area < min_area:
            continue

        masks.append(mask)
        meta.append(
            {
                "label": label,
                "score": float(score),
                "bbox_raw": bbox,
                "bbox_used": used_box,
                "bbox_tight": tb,
                "area": area,
            }
        )

    # mask NMS
    keep = []
    for i in range(len(masks)):
        ok = True
        for j in keep:
            if mask_iou(masks[i], masks[j]) > nms_iou:
                ok = False
                break
        if ok:
            keep.append(i)

    masks_keep = [masks[i] for i in keep]
    meta_keep = [meta[i] for i in keep]
    return masks_keep, meta_keep


def process_one_image(
    image_path: str,
    predictor,
    output_root: str,
    max_objects: int,
    points_per_type: int,
    box_pad: float,
    min_area: int,
    nms_iou: float,
    force_vlm: bool,
) -> str:
    base = os.path.basename(image_path)
    stem = os.path.splitext(base)[0]

    out_dir = os.path.join(output_root, f"outputs")
    _ensure_dir(out_dir)

    dir_inst = os.path.join(output_root, "VLM_instances")
    dir_vis = os.path.join(output_root, "VLM_instances_vis")
    dir_masks = os.path.join(output_root, "VLM_instances_masks_pt")

    _ensure_dir(dir_inst)
    _ensure_dir(dir_vis)
    _ensure_dir(dir_masks)

    result = vlm_get_all_bboxes(
        image_path,
        max_objects=max_objects,
        pad_ratio=0.05,
        points_per_type=points_per_type,
    )
    model = result.get("model", "vlm")
    json_path = os.path.join(out_dir, f"bboxes_{stem}_{model}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    bboxes = result.get("bboxes", [])

    # 2) SAM
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    h, w = img.shape[:2]

    masks_keep, meta_keep = _run_sam_for_bboxes(
        predictor=predictor,
        img_bgr=img,
        bboxes=bboxes,
        box_pad=box_pad,
        min_area=min_area,
        nms_iou=nms_iou,
    )

    # 3) 输出可视化
    inst = _colorize_instances(h, w, masks_keep)

    vis = img.copy()
    for it in meta_keep:
        x1, y1, x2, y2 = map(int, it["bbox_tight"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            vis,
            it["label"],
            (x1, max(0, y1 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    inst_path = os.path.join(dir_inst, f"instances_sam_{stem}.png")
    vis_path = os.path.join(dir_vis, f"instances_sam_vis_{stem}.png")
    # meta_path = os.path.join(out_dir, "instances_sam_meta.json")
    pt_path = os.path.join(dir_masks, f"instances_sam_masks_{stem}.pt")

    cv2.imwrite(inst_path, inst)
    cv2.imwrite(vis_path, vis)
    # with open(meta_path, "w", encoding="utf-8") as f:
    #     json.dump({"instances": meta_keep}, f, ensure_ascii=False, indent=2)

    # 4) 纯 masks tensor（同 extract_segment_everything_masks.py：只存 tensor）
    if len(masks_keep) > 0:
        masks_t = torch.from_numpy(np.stack(masks_keep, axis=0)).to(torch.bool)  # [N,H,W]
    else:
        masks_t = torch.zeros((0, h, w), dtype=torch.bool)
    torch.save(masks_t, pt_path)

    return out_dir


def main():
    image_dir = os.environ.get("IMAGE_DIR", "/home/majortom/majortom/datasets/desktopobjects_360/desk1_4_VLM/images_4")  # 改成你的图片文件夹
    output_root = os.environ.get("OUTPUT_ROOT", os.path.dirname(image_dir))

    sam_ckpt = os.environ.get("SAM_CKPT", "/home/majortom/majortom/datasets/ckpt/sam_vit_h_4b8939.pth")
    sam_arch = os.environ.get("SAM_ARCH", "vit_h")
    device = os.environ.get("DEVICE", "cuda")

    max_objects = int(os.environ.get("MAX_OBJECTS", "15"))
    points_per_type = int(os.environ.get("POINTS_PER_TYPE", "8"))

    box_pad = float(os.environ.get("BOX_PAD", "0.12"))
    min_area = int(os.environ.get("MIN_AREA", "300"))
    nms_iou = float(os.environ.get("NMS_IOU", "0.85"))

    force_vlm = os.environ.get("FORCE_VLM", "0") == "1"

    images = _list_images(image_dir)
    if not images:
        raise SystemExit(f"在目录中没有找到图片: {image_dir}")

    predictor = load_sam(sam_ckpt, sam_arch, device=device)

    print(f"IMAGE_DIR: {image_dir}")
    print(f"OUTPUT_ROOT: {output_root}")
    print(f"images: {len(images)}")

    for idx, p in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] processing: {p}")
        out_dir = process_one_image(
            image_path=p,
            predictor=predictor,
            output_root=output_root,
            max_objects=max_objects,
            points_per_type=points_per_type,
            box_pad=box_pad,
            min_area=min_area,
            nms_iou=nms_iou,
            force_vlm=force_vlm,
        )
        print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()