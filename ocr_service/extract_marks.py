"""通用「提取标记」：给一张图 → 印章框+章文 或 全文字框。

职责固定且单一：**输入一张图片**，输出检测框 + 文字。不碰 PDF、不做渲染、不回传图片
（PDF→图片、按框裁图都是调用方业务层的事）。不含任何应用概念（不出现网格/签名表/归档/主体等词）。

默认实现延迟导入 + 进程内缓存 paddle 模型（首次加载、后续请求复用），便于多页逐张调用；
单元测试注入 layout_fn/ocr_fn/seal_fn/page_image_fn 桩，不真跑模型。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

# 进程内模型缓存：容器生命周期内首次加载、后续请求复用，避免逐页重载。
_OCR_ENGINE = None
_LAYOUT_MODEL = None
_SEAL_PIPE = None


def _default_layout(image_path, w: int, h: int) -> list[dict]:
    """跑 PP-DocLayout，返回该图 role=seal 的像素 bbox。延迟导入 + 缓存 paddle 模型。"""
    global _LAYOUT_MODEL
    from ocr.probe_layout import detect_layout, MODEL_DIR, LAYOUT_DEVICE
    if _LAYOUT_MODEL is None:
        from paddlex import create_model
        _LAYOUT_MODEL = create_model(model_name="PP-DocLayoutV2",
                                     model_dir=str(MODEL_DIR), device=LAYOUT_DEVICE)
    pages = [{"page": 1, "path": image_path, "width_px": w, "height_px": h}]
    out = detect_layout(pages, score_threshold=0.5, model=_LAYOUT_MODEL)
    seals = []
    for pg in out:
        for r in pg["regions"]:
            if r["role"] != "seal":
                continue
            x1, y1, x2, y2 = r["bbox"]              # 归一化 0-1000
            seals.append({"bbox_px": [round(x1 / 1000 * w), round(y1 / 1000 * h),
                                      round(x2 / 1000 * w), round(y2 / 1000 * h)]})
    return seals


def _default_ocr(image_path) -> list[dict]:
    """对该图跑 OCR det+rec，返回每行 {text,score,cx,cy,top}。延迟导入 + 缓存引擎。"""
    global _OCR_ENGINE
    from ocr.build_document import build_ocr, ocr_page
    if _OCR_ENGINE is None:
        _OCR_ENGINE = build_ocr()
    return ocr_page(_OCR_ENGINE, Path(image_path))


def _default_seal(crop: np.ndarray) -> str:
    """裁图存临时 png → seal_recognition 读章文。延迟导入 + 缓存管线。"""
    global _SEAL_PIPE
    from ocr.recognize_seals import collect_rec_texts
    if _SEAL_PIPE is None:
        from paddlex import create_pipeline
        _SEAL_PIPE = create_pipeline("seal_recognition")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, crop)
        results = list(_SEAL_PIPE.predict(f.name, use_doc_orientation_classify=False,
                                          use_doc_unwarping=False, use_layout_detection=False))
    texts = []
    for r in results:
        j = r.json if hasattr(r, "json") else r
        texts += collect_rec_texts(j)
    return "".join(t for t in texts if t.strip())


def extract(kind: str, image_path: str, *, page_image_fn=None,
            layout_fn=_default_layout, ocr_fn=_default_ocr, seal_fn=_default_seal) -> dict:
    """给一张图 → marks。

    kind=seal: role=seal 框 + 每框章文 → marks=[{bbox,seal_text,text:null}]
    kind=signature: 全文字框 → marks=[{bbox,text,score,cx,cy}]（供调用方按各自规则用，如取某列文字）
    bbox 为像素坐标（对应传入的图）。返回 {width, height, marks}；不回传图片。
    """
    read = page_image_fn or cv2.imread
    img = read(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图片（OCR 只接受图片，PDF→图片请在调用方完成）：{image_path}")
    h, w = img.shape[:2]
    marks: list[dict] = []
    if kind == "seal":
        for r in layout_fn(image_path, w, h):
            x1, y1, x2, y2 = r["bbox_px"]
            crop = img[y1:y2, x1:x2]
            marks.append({"bbox": [x1, y1, x2, y2],
                          "seal_text": seal_fn(crop) if crop.size else "", "text": None})
    else:  # signature
        for ln in ocr_fn(image_path):
            b = ln.get("bbox") or [int(ln["cx"]), int(ln["top"]), int(ln["cx"]), int(ln["cy"])]
            marks.append({"bbox": b, "text": ln["text"], "score": ln.get("score"),
                          "cx": ln["cx"], "cy": ln["cy"]})
    return {"width": w, "height": h, "marks": marks}
