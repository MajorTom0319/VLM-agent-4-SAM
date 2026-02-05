import os
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

# ---------------------------
# SAM loader
# ---------------------------
def load_sam(sam_checkpoint_path: str, sam_arch: str = "vit_h", device: str = "cuda"):
    sam = sam_model_registry[sam_arch](checkpoint=sam_checkpoint_path).to(device)
    predictor = SamPredictor(sam)
    return predictor

# ---------------------------
# Box utils
# ---------------------------
def clip_box_xyxy(box, w, h):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return [x1, y1, x2, y2]

def pad_box_xyxy(box, w, h, pad_ratio=0.10, pad_px_min=8):
    x1, y1, x2, y2 = map(int, box)
    bw = x2 - x1
    bh = y2 - y1
    pad_x = max(pad_px_min, int(round(bw * pad_ratio)))
    pad_y = max(pad_px_min, int(round(bh * pad_ratio)))
    return clip_box_xyxy([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], w, h)

# ---------------------------
# Mask scoring
# ---------------------------
def _touch_border(mask: np.ndarray, box, tol=2) -> float:
    x1, y1, x2, y2 = box
    m = mask.astype(np.uint8)
    sub = m[y1:y2, x1:x2]
    if sub.size == 0:
        return 1.0
    top = sub[:tol, :].sum()
    bottom = sub[-tol:, :].sum()
    left = sub[:, :tol].sum()
    right = sub[:, -tol:].sum()
    border = top + bottom + left + right
    area = sub.sum() + 1e-6
    return float(border / area)

def _mask_metrics(mask: np.ndarray, box):
    x1, y1, x2, y2 = box
    box_area = float((x2 - x1) * (y2 - y1) + 1e-6)

    inside = mask[y1:y2, x1:x2].astype(bool)
    area_in = float(inside.sum())
    area_total = float(mask.astype(bool).sum())
    area_ratio = area_in / box_area
    overspill = max(0.0, (area_total - area_in) / (area_total + 1e-6))
    touch = _touch_border(mask, box, tol=2)
    return area_ratio, overspill, touch

# ---------------------------
# Predictor.set_image cache
# ---------------------------
_last_image_sig = None

def _set_image_cached(predictor: SamPredictor, bgr_image: np.ndarray):
    global _last_image_sig
    sig = (int(bgr_image.__array_interface__["data"][0]), bgr_image.shape[0], bgr_image.shape[1])
    if _last_image_sig == sig:
        return
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    _last_image_sig = sig

# ---------------------------
# Point utils (for combining VLM points with SAM)
# ---------------------------
def _normalize_points(pos_points, neg_points):
    """
    Accepts points as:
      - [[x,y], ...] (preferred)
      - None / [] (allowed)
    Returns:
      point_coords: (N,2) float32
      point_labels: (N,) int32  (1 for pos, 0 for neg)
    """
    pos_points = pos_points or []
    neg_points = neg_points or []

    coords = []
    labels = []

    for p in pos_points:
        if p and len(p) == 2:
            coords.append([float(p[0]), float(p[1])])
            labels.append(1)

    for p in neg_points:
        if p and len(p) == 2:
            coords.append([float(p[0]), float(p[1])])
            labels.append(0)

    if len(coords) == 0:
        return None, None

    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int32)

def _filter_points_in_box(point_coords: np.ndarray, point_labels: np.ndarray, box_xyxy):
    """
    Keep points that are inside image bounds; allow neg points outside box,
    but it is usually safer to keep both inside a padded box region.
    Here we keep:
      - positive points must be inside box
      - negative points: allow inside the padded box neighborhood (handled by caller)
    """
    x1, y1, x2, y2 = box_xyxy
    keep_coords = []
    keep_labels = []
    for (x, y), lb in zip(point_coords, point_labels):
        if lb == 1:
            if x1 <= x < x2 and y1 <= y < y2:
                keep_coords.append([x, y])
                keep_labels.append(lb)
        else:
            keep_coords.append([x, y])
            keep_labels.append(lb)

    if len(keep_coords) == 0:
        return None, None
    return np.asarray(keep_coords, dtype=np.float32), np.asarray(keep_labels, dtype=np.int32)

def _add_box_center_if_missing(point_coords: np.ndarray, point_labels: np.ndarray, box_xyxy):
    """
    Ensure at least one positive point exists. If none, add center point as positive.
    """
    if point_coords is None or point_labels is None:
        x1, y1, x2, y2 = box_xyxy
        center = np.array([[0.5 * (x1 + x2), 0.5 * (y1 + y2)]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        return center, labels

    if not np.any(point_labels == 1):
        x1, y1, x2, y2 = box_xyxy
        center = np.array([[0.5 * (x1 + x2), 0.5 * (y1 + y2)]], dtype=np.float32)
        point_coords = np.concatenate([center, point_coords], axis=0)
        point_labels = np.concatenate([np.array([1], dtype=np.int32), point_labels], axis=0)

    return point_coords, point_labels

# ---------------------------
# Main API (keeps sam_segment_from_box.py compatible)
# ---------------------------
def segment_instance_from_box(
    predictor: SamPredictor,
    bgr_image: np.ndarray,
    bbox_xyxy: list[int],
    pad_ratio: float = 0.10,
    # neg_margin: int = 18,  # keep for backward compatibility (unused in VLM point mode)
    # fg_grid: tuple[int, int] = (2, 2),  # keep for backward compatibility (unused in VLM point mode)
    multimask_output: bool = True,
    pos_points=None,
    neg_points=None,
):
    """
    Unified SAM inference:
    - If pos_points/neg_points are provided (from VLM), uses BOX + POINT prompts jointly.
    - Otherwise falls back to BOX-only (still works with sam_segment_from_box.py).

    Returns: mask(bool HxW), score(float), used_box(list[int])
    """
    h, w = bgr_image.shape[:2]
    used_box = pad_box_xyxy(bbox_xyxy, w, h, pad_ratio=pad_ratio)
    _set_image_cached(predictor, bgr_image)

    # If VLM points exist -> use box + points
    point_coords, point_labels = _normalize_points(pos_points, neg_points)
    if point_coords is not None:
        # Make sure positive points are inside the (tight) bbox; negative points can be outside
        point_coords, point_labels = _filter_points_in_box(point_coords, point_labels, bbox_xyxy)
        point_coords, point_labels = _add_box_center_if_missing(point_coords, point_labels, bbox_xyxy)

    box = np.array(used_box, dtype=np.float32)[None, :]

    if point_coords is None:
        # Box-only mode (compat)
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=multimask_output,
        )
    else:
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output,
        )

    # pick best mask with a bit of heuristic scoring (reduce leakage)
    best_i = 0
    best_val = -1e9
    for i in range(masks.shape[0]):
        m = masks[i].astype(bool)
        area_ratio, overspill, touch = _mask_metrics(m, used_box)

        val = float(scores[i])
        if area_ratio < 0.05:
            val -= 0.6
        if area_ratio > 0.98:
            val -= 0.4
        val -= 0.8 * overspill
        val -= 0.4 * touch

        # When points exist, trust the model a bit more but still penalize heavy leakage
        if point_coords is not None and overspill > 0.35:
            val -= 0.4

        if val > best_val:
            best_val = val
            best_i = i

    return masks[best_i].astype(bool), float(scores[best_i]), used_box

def tight_bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return [x1, y1, x2, y2]

def save_mask_png(mask: np.ndarray, out_path: str):
    m = (mask.astype(np.uint8) * 255)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, m)

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum() + 1e-6
    return float(inter / union)