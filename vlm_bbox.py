import base64
import json
import os
from openai import OpenAI
from PIL import Image

MODEL = os.environ.get("VLM_MODEL", "gemini-3.1-pro-preview")
BASE_URL = os.environ.get("VLM_BASE_URL", "https://api.vectorengine.ai/v1")

client = OpenAI(
    base_url=BASE_URL,
    api_key=os.environ.get("VLM_API_KEY", "sk-7k47Hby1ncfVkDKrOyEzhAofEVwCuSw2LxPLT2Rtgv6Ff1x2"),
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

            if 0 <= y_pix <= h and 0 <= x_pix <= w:
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
    max_objects: int = 10,
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
        "ROLE:\n"
        "You are a precise visual grounding assistant specialized in INSTANCE-LEVEL object detection for segmentation.\n\n"

        "PRIMARY OBJECTIVE:\n"
        "Return prompts (box_2d + pos_points + neg_points) for all salient, standalone, instance-level objects in the image.\n"
        "Focus on WHOLE objects only, not object parts, attributes, textures, or background regions.\n\n"

        "WHAT COUNTS AS ONE OBJECT INSTANCE:\n"
        "- One physical object = one entry.\n"
        "- The object must be recognized as a complete instance at its natural semantic level.\n"
        "- Examples:\n"
        "  * one whole dinosaur skeleton = one object\n"
        "  * one whole flower = one object\n"
        "  * one whole tree = one object\n"
        "  * one whole car = one object\n"
        "  * one whole chair = one object\n\n"

        "INSTANCE CONSISTENCY (CRITICAL):\n"
        "- ONE ENTRY = ONE WHOLE OBJECT INSTANCE.\n"
        "- Do NOT split a single object into parts.\n"
        "- Do NOT label object parts as separate instances.\n"
        "- Do NOT output separate entries for head, body, tail, branch, leaf cluster, flower petal, wheel, window, handle, or other sub-parts if they belong to the same object.\n"
        "- If a single object contains many visible components but forms one semantic whole, output exactly ONE entry.\n"
        "- Do NOT duplicate the same object with different labels.\n"
        "- Do NOT create multiple overlapping boxes for the same physical object.\n\n"

        "WHOLE-OBJECT PRIORITY:\n"
        "- Prefer the complete semantic object over local visible parts.\n"
        "- For articulated, thin, or complex structures, still treat them as one instance if they belong to one object.\n"
        "- For example:\n"
        "  * a mounted T-rex skeleton is one object, not skull/ribs/legs/tail\n"
        "  * a flowering plant is one object if shown as a single plant instance\n"
        "  * a tree is one object, not trunk/branches/leaves separately\n\n"

        "OCCLUSION AND TRUNCATION:\n"
        "- If an object is partially occluded or truncated by the image boundary, still output it as one instance if it is clearly a single object.\n"
        "- The box should cover all visible parts of that object.\n"
        "- Do NOT hallucinate invisible parts outside the image.\n\n"

        "WHAT TO IGNORE:\n"
        "- Ignore background/stuff regions such as wall, floor, sky, grass, road, table surface, and other scene regions unless they are clearly standalone physical objects.\n"
        "- Ignore tiny clutter or ambiguous fragments that are not meaningful standalone instances.\n"
        "- Ignore patterns, shadows, reflections, and textures.\n\n"

        "LABELING RULES:\n"
        "- Use concise English noun phrases.\n"
        "- Prefer the most specific whole-object category that is visually justified.\n"
        "- Examples: 'Tyrannosaurus rex skeleton', 'flower', 'tree', 'car', 'person', 'chair'.\n"
        "- Avoid part-level names unless the part itself is a detached object.\n\n"

        "BOX REQUIREMENTS:\n"
        "- box_2d must be a tight box around the whole visible object instance.\n"
        "- Include all visible parts belonging to the same object, including thin or extended parts.\n"
        "- For complex shapes, the box should enclose the entire visible instance, not only the most salient part.\n"
        "- Do not make the box excessively large.\n\n"

        "POINT REQUIREMENTS:\n"
        "pos_points:\n"
        "- Place points on the target object itself.\n"
        "- Spread points across different visible regions of the same object.\n"
        "- For elongated or articulated objects, place points on separated visible parts to represent the whole instance.\n"
        "- All pos_points must lie inside box_2d.\n"
        f"- Up to {points_per_type} pos_points.\n\n"

        "neg_points:\n"
        "- Place points on nearby non-target regions close to the object boundary.\n"
        "- Use negatives to separate the target object from adjacent objects or background.\n"
        "- For thin or complex objects, place negatives around the outer contour to prevent leakage.\n"
        f"- Up to {points_per_type} neg_points.\n\n"

        "SELECTION PRIORITY:\n"
        "- Prioritize salient, complete, standalone objects.\n"
        "- If many objects are present, return the most visually important and clearly separable instances first.\n"
        f"- Maximum number of objects: {max_objects}.\n\n"

        "OUTPUT FORMAT (STRICT JSON ONLY):\n"
        "- Output must be strict JSON only. No markdown. No extra text.\n"
        "- Coordinates are normalized to [0,1000].\n"
        "- box_2d format: [ymin, xmin, ymax, xmax].\n"
        "- point format: [y, x].\n"
        "- Values must be within [0,1000] and boxes must satisfy ymin < ymax, xmin < xmax.\n\n"

        "JSON schema:\n"
        "{\n"
        "  \"bboxes\": [\n"
        "    {\n"
        "      \"label\": \"object name\",\n"
        "      \"box_2d\": [ymin, xmin, ymax, xmax],\n"
        "      \"pos_points\": [[y,x], ...],\n"
        "      \"neg_points\": [[y,x], ...]\n"
        "    }\n"
        "  ]\n"
        "}\n"

    )

        # "ROLE:\n"
        # "You are a precise visual grounding assistant specialized for INSTANCE-LEVEL scene annotation for segmentation.\n\n"

        # "PRIMARY OBJECTIVE:\n"
        # "Return prompts (box_2d + pos_points + neg_points) for ALL object-level instances and major stuff regions so that labeled regions collectively COVER THE ENTIRE IMAGE.\n\n"

        # "WHAT MUST BE LABELED:\n"
        # "A) THINGS: all countable object instances (each physical item is one entry).\n"
        # "   - Include objects inside other objects/containers (e.g., items inside a transparent box).\n"
        # "B) STUFF: all major background/structural regions that fill the scene.\n"
        # "   - MUST include when visible: tabletop surface, wall(s), window/glass pane(s), window frames/mullions.\n"
        # "   - If visible through glass, include outdoor view as 1–3 large regions (do not over-fragment).\n\n"

        # "INSTANCE CONSISTENCY (CRITICAL):\n"
        # "- ONE ENTRY = ONE WHOLE OBJECT/REGION.\n"
        # "- Do NOT split one object into parts: for a camera, output ONE entry for the entire camera (body+lens+buttons as one object).\n"
        # "- Do NOT output separate entries for different parts of the same object unless clearly detached separate items.\n"
        # "- OCCLUSION/FRAGMENT MERGE: if an object is partially occluded and appears separated into multiple visible fragments, output ONE entry covering ALL visible fragments (one box spanning the fragments).\n"
        # "- Do NOT duplicate the same object.\n"
        # "- Do NOT group multiple distinct objects into one entry.\n\n"

        # "FULL-SCENE COVERAGE AUDIT (DO THIS BEFORE OUTPUT):\n"
        # "- After listing things and stuff, verify coverage of:\n"
        # "  * Top-left, top-right, bottom-left, bottom-right corners\n"
        # "  * Along the top edge (usually window/sky), and large central background areas\n"
        # "  * Any large planar area (wall/glass/table) that remains unlabeled\n"
        # "- If any large region is unlabeled, add a stuff entry for it (wall/glass/outdoor/table/other plane).\n\n"

        # "OUTPUT FORMAT (STRICT JSON ONLY):\n"
        # "- Output MUST be strict JSON only. No markdown. No extra text.\n"
        # "- Coordinates are normalized to [0,1000].\n"
        # "- box_2d format: [ymin, xmin, ymax, xmax].\n"
        # "- point format: [y, x].\n"
        # "- Values must be within [0,1000] and boxes must satisfy ymin<ymax, xmin<xmax.\n\n"

        # "BOX REQUIREMENTS:\n"
        # "- Tight box around ONLY the target instance/region.\n"
        # "- For whole objects: include all visible parts (thin straps/cables/legs/handles).\n"
        # "- For occluded objects: include all visible fragments in ONE box.\n\n"

        # "POINT REQUIREMENTS:\n"
        # "pos_points:\n"
        # "- On the target, spread across distinct parts/areas.\n"
        # "- For large planar stuff (glass/wall/table): spread points broadly (center + near corners).\n"
        # "- All pos_points MUST lie inside box_2d.\n"
        # f"- Up to {points_per_type} pos_points.\n"
        # "neg_points:\n"
        # "- On nearby NON-target pixels near boundaries to prevent SAM leakage.\n"
        # "- Prefer on adjacent objects/surfaces at contact boundaries (object vs table, glass vs frame, wall vs window frame).\n"
        # f"- Up to {points_per_type} neg_points.\n\n"

        # "CONSTRAINTS:\n"
        # f"- Max objects(regions) allowed: {max_objects}. Use them wisely to maximize coverage.\n"
        # "- Prefer covering the entire scene over adding redundant tiny regions.\n\n"

        # "JSON schema:\n"
        # "{\n"
        # "  \"bboxes\": [\n"
        # "    {\n"
        # "      \"label\": \"object\",\n"
        # "      \"box_2d\": [ymin, xmin, ymax, xmax],\n"
        # "      \"pos_points\": [[y,x], ...],\n"
        # "      \"neg_points\": [[y,x], ...]\n"
        # "    }\n"
        # "  ]\n"
        # "}\n"

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", 
                    # "text": "Return JSON for ALL distinct things AND major stuff regions in the entire scene (including glass/window, window frames, walls, floor/ground, tabletop, outdoor view regions), with box_2d + pos_points + neg_points. Do not miss large planar regions."
                    "text": 
                            "Return JSON for all salient standalone object instances in the image. "
                            "Detect whole objects only. Do not split one object into parts. "
                            "Do not include background or stuff regions. "
                            "For each object, return label, box_2d, pos_points, and neg_points."
                    },
                    
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0,
        timeout=100,
    )
    # Return JSON with all foreground objects, each with box_2d + pos_points + neg_points

    
    text = resp.choices[0].message.content or ""
    text = _extract_json(text)
    obj = json.loads(text)

    # 轻量校验/裁剪
    # raw_fg = (obj.get("bboxes") or [])[:max_objects]
    raw_fg = (obj.get("bboxes") or [])
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