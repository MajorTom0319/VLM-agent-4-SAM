import os
import json
import argparse

import cv2
import numpy as np
import torch

from sam_prompted_segment import (
    load_sam,
    segment_instance_from_box,
    tight_bbox_from_mask,
    mask_iou,
)

def colorize_instances(h, w, masks):
    rng = np.random.default_rng(0)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for m in masks:
        color = rng.integers(0, 255, size=(3,), dtype=np.uint8)
        out[m] = color
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=os.environ.get("IMG", ""), help="path to image")
    ap.add_argument("--bbox_json", default=os.environ.get("BBOX_JSON", ""), help="path to bbox json produced by run_one_image")
    ap.add_argument("--output_root", default=os.environ.get("OUTPUT_ROOT", ""), help="root folder to save results")

    args = ap.parse_args()

    if not args.output_root:
        args.output_root = os.path.dirname(os.path.dirname(os.path.abspath(args.img)))
    os.makedirs(args.output_root, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.img))[0]

    # img_path = os.environ.get("IMG", "/home/majortom/majortom/datasets/desktopobjects_360/desk1_4_VLM/images_4/0021.jpg")
    # bbox_json = os.environ.get("BBOX_JSON", "/home/majortom/majortom/project/VLMforSAM/outputs_0021/bboxes_gemini-3-pro-preview.json")
    sam_ckpt = os.environ.get("SAM_CKPT", "/home/majortom/majortom/datasets/ckpt/sam_vit_h_4b8939.pth")
    sam_arch = os.environ.get("SAM_ARCH", "vit_h")

    # if not img_path or not bbox_json or not sam_ckpt:
    #     raise SystemExit("请设置环境变量 IMG, BBOX_JSON, SAM_CKPT")

    img = cv2.imread(args.img)
    if img is None:
        raise SystemExit(f"无法读取图像: {args.img}")
    h, w = img.shape[:2]

    with open(args.bbox_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    bboxes = data.get("bboxes", [])

    predictor = load_sam(sam_ckpt, sam_arch, device="cuda")

    masks = []
    meta = []

    # 逐框分割
    for it in bboxes:
        bbox = it.get("bbox")
        label = it.get("label", "obj")
        if not bbox:
            continue
        #  添加正负点
        pos_points = it.get("pos_points") 
        neg_points = it.get("neg_points") 

        mask, score, used_box = segment_instance_from_box(
            predictor,
            img,
            bbox,
            pad_ratio=float(os.environ.get("BOX_PAD", "0.12")),
            neg_margin=int(os.environ.get("NEG_MARGIN", "18")),
            fg_grid=(2, 2),
            multimask_output=True,
            pos_points=pos_points,  
            neg_points=neg_points, 
        )

        tb = tight_bbox_from_mask(mask)
        if tb is None:
            continue

        # 简单质量过滤：太小就丢
        area = int(mask.sum())
        if area < int(os.environ.get("MIN_AREA", "300")):
            continue

        masks.append(mask)
        meta.append(
            {
                "label": label,
                "score": score,
                "bbox_raw": bbox,
                "bbox_used": used_box,
                "bbox_tight": tb,
                "area": area,
            }
        )

    # 去重：IoU 太高认为是同一个实例
    iou_thr = float(os.environ.get("NMS_IOU", "0.85"))
    keep = []
    for i in range(len(masks)):
        ok = True
        for j in keep:
            if mask_iou(masks[i], masks[j]) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(i)

    masks_keep = [masks[i] for i in keep]
    meta_keep = [meta[i] for i in keep]

    # 输出：实例彩色图 + 可视化框
    inst = colorize_instances(h, w, masks_keep)

    vis = img.copy()
    for it in meta_keep:
        x1, y1, x2, y2 = map(int, it["bbox_tight"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, it["label"], (x1, max(0, y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # out_dir = os.path.join(out_dir, f"outputs_{os.path.basename(args.img_path).rsplit('.',1)[0]}")
    # os.makedirs(out_dir, exist_ok=True)

    dir_inst = os.path.join(args.output_root, "VLM_instances")
    dir_vis = os.path.join(args.output_root, "VLM_instances_vis")
    dir_masks = os.path.join(args.output_root, "VLM_instances_masks_pt")
    os.makedirs(dir_inst, exist_ok=True)
    os.makedirs(dir_vis, exist_ok=True)
    os.makedirs(dir_masks, exist_ok=True)

    inst_path = os.path.join(dir_inst, f"instances_sam_{stem}.png")
    vis_path = os.path.join(dir_vis, f"instances_sam_vis_{stem}.png")
    # meta_path = os.path.join(out_dir, "instances_sam_meta.json")

    # masks 的 .pt 输出，只保存 tensor
    pt_path = os.path.join(dir_masks, f"instances_sam_masks_{stem}.pt")
    if len(masks_keep) > 0:
        masks_t = torch.from_numpy(np.stack(masks_keep, axis=0)).to(torch.bool)  # [N,H,W]
    else:
        masks_t = torch.zeros((0, h, w), dtype=torch.bool)
    torch.save(masks_t, pt_path)

    cv2.imwrite(inst_path, inst)
    cv2.imwrite(vis_path, vis)
    # with open(meta_path, "w", encoding="utf-8") as f:
    #     json.dump({"instances": meta_keep}, f, ensure_ascii=False, indent=2)

    print(f"saved: {inst_path}")
    print(f"saved: {vis_path}")
    # print(f"saved: {meta_path}")
    print(f"saved: {pt_path}") 
    print(f"instances: {len(masks_keep)} (raw {len(masks)})")

if __name__ == "__main__":
    main()