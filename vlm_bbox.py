import base64
import json
import os
from openai import OpenAI
from PIL import Image

MODEL = os.environ.get("VLM_MODEL", "gemini-3-pro-preview")
BASE_URL = os.environ.get("VLM_BASE_URL", "https://api.vectorengine.ai/v1")

client = OpenAI(
    base_url=BASE_URL,
    api_key=os.environ.get("VLM_API_KEY", ""),
)

def _image_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _extract_json(text: str) -> str:
    """
    兼容模型偶尔输出多余文本：截取首个 '{' 到最后一个 '}'。
    """
    if not text:
        return text
    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        return text[i : j + 1]
    return text

def _clip(v, lo, hi):
    return max(lo, min(hi, v))

def _norm1000_xyxy_px(box_2d, width: int, height: int):
    """
    box_2d: [y1,x1,y2,x2] in [0,1000]
    return: [x1,y1,x2,y2] pixel ints, clipped to image bounds
    """
    ymin, xmin, ymax, xmax = box_2d

    # 允许模型输出 float
    y1 = int(round(float(ymin) / 1000.0 * height))
    x1 = int(round(float(xmin) / 1000.0 * width))
    y2 = int(round(float(ymax) / 1000.0 * height))
    x2 = int(round(float(xmax) / 1000.0 * width))

    x1 = _clip(x1, 0, width - 1)
    x2 = _clip(x2, 1, width)
    y1 = _clip(y1, 0, height - 1)
    y2 = _clip(y2, 1, height)

    # 保证 x1<x2, y1<y2
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return [x1, y1, x2, y2]

def _pad_box_xyxy_px(box, width: int, height: int, pad_ratio: float = 0.05):
    """
    Expand pixel XYXY box by pad_ratio of its size (default 5%).
    """
    x1, y1, x2, y2 = map(int, box)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    x1 = _clip(x1 - pad_x, 0, width - 1)
    y1 = _clip(y1 - pad_y, 0, height - 1)
    x2 = _clip(x2 + pad_x, 1, width)
    y2 = _clip(y2 + pad_y, 1, height)
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]

def _parse_points(points, w, h):
    """
    points: [[y,x], ...] where coords are normalized to [0,1000]
    """
    if not points:
        return []
    out = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            y_norm, x_norm = p
            y_pix = y_norm * h / 1000.0
            x_pix = x_norm * w / 1000.0

            if 0 <= y_pix <= h or 0 <= x_pix <= w:
                out.append([float(y_norm), float(x_norm)])
    return out

def _norm1000_xy_to_px(pt, width: int, height: int):
    y, x = pt
    px = int(round(float(x) / 1000.0 * width))
    py = int(round(float(y) / 1000.0 * height))
    px = _clip(px, 0, width - 1)
    py = _clip(py, 0, height - 1)
    return [px, py]

def vlm_get_all_bboxes(
    image_path: str,
    max_objects: int = 15,
    # max_background: int = 3,
    pad_ratio: float = 0.05,
    points_per_type: int = 8,
) -> dict:
    """
    return:
    {
      "image_size": {"width": W, "height": H},
      "bboxes":[
        {
          "label":"...",
          "bbox":[ymin, xmin, ymax, xmax],            # padded pixel bbox (expanded by pad_ratio)
          "bbox_tight":[ymin, xmin, ymax, xmax],      # tight pixel bbox
          "pos_points":[[y,x],...],        # pixel points
          "neg_points":[[y,x],...]         # pixel points
        }, ...
      ],
      "model": MODEL
    }
    """
    data_url = _image_to_data_url(image_path)
    w, h = Image.open(image_path).size
    print(f"Image size: width={w}, height={h}")

    system = (
        "You are a precise visual grounding assistant specialized for segmentation prompting."

        "Goal:"
        "Given ONE image, detect distinct FOREGROUND object instances and output a JSON list. Each instance must include:"
        "- a tight 2D bounding box around the VISIBLE pixels of that single object instance"
        "- positive points on the object (to help SAM fill holes and cover reflective/textureless areas)"
        "- negative points on non-object regions near boundaries/adjacent objects (to prevent leakage/merging)"

        " IMPORTANT OUTPUT RULES:"
        "- Output MUST be strict JSON only (no markdown, no explanation)."
        "- Use normalized coordinates in [0,1000]."
        "- Box format: box_2d = [ymin, xmin, ymax, xmax]."
        "- Point format: [y, x]."
        "- Integers preferred; floats allowed but must still be within [0,1000]."

        "OBJECT INSTANCE DEFINITION:"
        "- Each entry represents ONE single physical object instance."
        "- Do NOT merge multiple objects into one box, even if they touch or are similar."
        "- Exclude pure background regions (tabletop/wall/floor) unless they are a physical object (e.g., a box, a book, a device)."

        "BOX QUALITY REQUIREMENTS (TIGHT-VISIBLE):"
        "- The box must tightly enclose the object's visible silhouette:"
        "- close to the outer boundary on all 4 sides"
        "- include all visible parts of the object (do NOT cut off)"
        "- do NOT include large background margins"
        "- If the object is partially occluded, box only the visible part of the object (not the occluder)."
        "- For thin parts (handles, cables, legs): ensure they are included if visible."
        "- If uncertain between too tight vs slightly larger: prefer SLIGHTLY larger, but still tight."

        "POINT REQUIREMENTS:"
        "Positive points (pos_points):"
        "- Place ON the target object, spread across different parts."
        "- Must include at least:"
        "- 1 point near the center mass of the object"
        "- multiple points on challenging regions: reflective, shadowed, low-texture, transparent, or dark areas (to avoid holes)"
        "- All pos_points MUST lie inside box_2d."

        "Negative points (neg_points):"
        "- Place on NON-target pixels."
        "- Focus on separation:"
        "- on adjacent/touching objects near the contact boundary"
        "- on occluders covering the target"
        "- on background close to edges of the target to stop mask leakage"
        "- Avoid placing neg_points far away; keep them near the object boundary."
        "- neg_points should preferably be outside box_2d; if not possible, they still must be outside the object region."

        "HARD CASE GUIDELINES:"
        "- Many small objects: output the most salient up to Max objects; avoid random tiny clutter unless clearly a distinct object."
        "- Similar repeated objects: output separate instances if spatially separated."
        "- Transparent/reflective objects: use more pos_points on highlights and dark regions; use neg_points on background behind/around."
        "- Text on surfaces is not a separate object unless it is a physical item (e.g., a sticker/label as a separate object)."

        "SELF-CHECK (must satisfy before final output):"
        "For each instance:"
        "1) box_2d values are within [0,1000] and ymin<ymax, xmin<xmax."
        "2) All pos_points are inside box_2d."
        "3) neg_points are not on the target object; prefer near boundaries."
        "4) The box is tight-visible: not missing object parts and not overly large."
        "If any check fails, revise internally and output only the corrected JSON."

        "Constraints:\n"
        f"- Max objects: {max_objects}.\n"
        f"- Max {points_per_type} positive and {points_per_type} negative points per object.\n"
        "- Do NOT treat multiple separate instances as a single object.\n\n"
        "JSON schema:\n"
        "{\n"
        "  \"bboxes\": [\n"
        "    {\n"
        "      \"label\": \"object\",\n"
        "      \"box_2d\": [ymin, xmin, ymax, xmax],\n"
        "      \"pos_points\": [[y,x], ...],\n"
        "      \"neg_points\": [[y,x], ...]\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )

        # "You are an expert visual grounding and segmentation assistant.\n"
        # "Task: Detect all distinct FOREGROUND objects (e.g., toys, tools, electronics) and generate prompts for segmentation.\n\n"
        # "Output MUST be strict JSON only. No extra text.\n\n"
        # "Coordinate System: normalized to [0-1000].\n"
        # "For EACH object, you MUST output:\n"
        # "1. box_2d [ymin, xmin, ymax, xmax]: Provide a tight 2D bounding box that encapsulates the entire visible part of the object.\n"
        # "2. pos_points [y,x]:\n"
        # "   - Must be placed on the target object, and spread across different parts.\n"
        # "   - CRITICAL: Place points on high-reflectance, shadowed, or low-texture regions within the object to prevent internal holes/noise.\n"
        # "   - Ensure points cover diverse parts (e.g., for a toy bus, place points on both the roof and the wheels).\n"
        # "   - Every pos_point must lie INSIDE the box_2d.\n"
        # "3. neg_points [y,x]:\n"
        # "Must be placed outside the target object; do not treat a portion of the target object as another object."
        # "   - CRITICAL: Focus on 'Occlusion Boundaries'. If object A is partially blocked by object B, place negative points on object B's surface near the contact line.\n"
        # "   - If two objects are adjacent/touching, place negative points on the neighboring object to prevent SAM from 'leaking' or merging them.\n"
        # "   - Can place some dots appropriately near the background of the objects.\n\n"

        # "你是视觉检测与定位助手。"
        # "任务：请从图像中识别出所有的前景物体，并为每个独立物体生成一个尽可能紧致的包围框（tight bbox），不要将不同对象合并为一个框，同时给出主要背景区域的大框（background_bboxes），不同的背景区域用不同的框代表。"
        # "输出要求："
        # "仅输出严格JSON，不要输出任何解释文字。\n\n"
        # "坐标体系：使用像素坐标，原点在左上角 (0,0)。图像尺寸："
        # f"width={w}, height={h}。\n\n"
        # f"- 最多输出 {max_objects} 个对象以及{max_background}个背景区域。\n"
        # "- 所有的bbox 必须是整数像素坐标，且满足 0<=x1<x2<=w, 0<=y1<y2<=h。\n"
        # "- 尽量覆盖：每个独立物体（玩偶、球、车、胶带卷、钟、笔等），不要把多个物体合成一个框。\n"
        # "- background_bboxes：选择主要“桌面/地面/墙面”等背景区域。"
        # "- label：为每个bbox生成简短的描述性标签，如“red ball”、“wooden table top”等。\n\n"
        # "输出格式：\n"
        # "{\n"
        # "  \"bboxes\": [\n"
        # "    {\"label\": \"object_1\", \"box_2d\": [x1,y1,x2,y2]},\n"
        # "    {\"label\": \"object_2\", \"box_2d\": [x1,y1,x2,y2]}\n"
        # "  ],\n"
        # "  \"background_bboxes\": [\n"
        # "    {\"label\": \"background_1\", \"box_2d\": [x1,y1,x2,y2]},\n"
        # "    {\"label\": \"background_2\", \"box_2d\": [x1,y1,x2,y2]}\n"
        # "  ]\n"
        # "}\n\n"

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Return JSON with all foreground objects, each with box_2d + pos_points + neg_points."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0,
        timeout=100,
    )

    text = resp.choices[0].message.content or ""
    text = _extract_json(text)
    obj = json.loads(text)

    # 轻量校验/裁剪
    raw_fg = (obj.get("bboxes") or [])[:max_objects]
    # raw_bg = (obj.get("background_bboxes") or [])[:max_background]

    # 转换为像素 xyxy
    fg = []
    for it in raw_fg:
        box_2d = it.get("box_2d")
        if not box_2d or len(box_2d) != 4:
            continue

        tight_box = _norm1000_xyxy_px(box_2d, w, h)
        padded_box = _pad_box_xyxy_px(tight_box, w, h, pad_ratio=pad_ratio)

        pos_norm = _parse_points(it.get("pos_points"), w, h)
        neg_norm = _parse_points(it.get("neg_points"), w, h)

        # ensure exactly points_per_type points (truncate; no hallucinated fill here)
        pos_norm = pos_norm[:points_per_type]
        neg_norm = neg_norm[:points_per_type]

        pos_px = [_norm1000_xy_to_px(p, w, h) for p in pos_norm]
        neg_px = [_norm1000_xy_to_px(p, w, h) for p in neg_norm]

        fg.append(
            {
                "label": it.get("label", "obj"),
                "bbox": padded_box,
                "bbox_tight": tight_box,
                "pos_points": pos_px,    # pixel points
                "neg_points": neg_px,    # pixel points
            }
        )

    # bg = []
    # for it in raw_bg:
    #     box_2d = it.get("box_2d")
    #     if not box_2d or len(box_2d) != 4:
    #         continue
    #     bg.append(
    #         {
    #             "label": it.get("label", "background"),
    #             "bbox": _norm1000_xyxy_px(box_2d, w, h),
    #             # "box_2d": box_2d,
    #         }
    #     )

    return {"image_size": {"width": w, "height": h},
            "bboxes": fg,
            # "background_bboxes": bg, 
            "model": MODEL,
    }