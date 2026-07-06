"""通用「提取标记」：渲染页图 + (印章框+章文 | 全文字框)。

保持通用：只做「渲染 + 检测框 + 文字」，不含任何应用/印鉴库概念
（不出现「网格/签名表/归档/主体/印鉴库」等词）。供调用方据此做各自的业务提取。

纯函数 extract(...) 接受可注入的 render_pages_fn/page_image_fn/layout_fn/ocr_fn/seal_fn，
默认实现延迟导入 paddle（仅 OCR 容器有），便于单元测试用桩而不真跑模型。
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _default_render(path, out_dir, dpi):
    """PDF → 每页 PNG；单张图片 → 当作单页。内容探测：cv2 能读出即图片，否则按 PDF 渲染。

    调用方可能传 PDF 或图片（截图）。pypdfium2 只吃 PDF，图片会抛异常，故先用 cv2 试读。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(path))              # 图片能解出；PDF/非图返回 None
    if img is not None:
        p = out_dir / "p1.png"
        cv2.imwrite(str(p), img)
        h, w = img.shape[:2]
        return [{"page": 1, "path": p, "width_px": w, "height_px": h}]
    from ocr.probe_layout import render_pages  # 延迟导入 pypdfium2（仅 PDF 路径需要）
    return render_pages(Path(path), out_dir, dpi)


def _default_page_image(png_path) -> np.ndarray:
    return cv2.imread(str(png_path))


def _default_layout(pages):
    """跑 PP-DocLayout，返回每页 role=seal 的像素 bbox。延迟导入 paddle。"""
    from ocr.probe_layout import detect_layout
    out = []
    for pg in detect_layout(pages, score_threshold=0.5):
        seals = []
        for r in pg["regions"]:
            if r["role"] != "seal":
                continue
            w, h = pg["width_px"], pg["height_px"]
            x1, y1, x2, y2 = r["bbox"]              # 归一化 0-1000
            seals.append({"role": "seal",
                          "bbox_px": [round(x1 / 1000 * w), round(y1 / 1000 * h),
                                      round(x2 / 1000 * w), round(y2 / 1000 * h)]})
        out.append({"page": pg["page"], "regions": seals})
    return out


def _default_ocr(png_path):
    """对单页图跑 OCR det+rec，返回每行 {text,score,cx,cy,top}。延迟导入 paddle。"""
    from ocr.build_document import build_ocr, ocr_page
    ocr = build_ocr()
    return ocr_page(ocr, Path(png_path))


def _default_seal(crop: np.ndarray) -> str:
    """裁图存临时 png → seal_recognition 读章文。延迟导入 paddle。"""
    from paddlex import create_pipeline
    from ocr.recognize_seals import collect_rec_texts
    pipe = create_pipeline("seal_recognition")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, crop)
        results = list(pipe.predict(f.name, use_doc_orientation_classify=False,
                                    use_doc_unwarping=False, use_layout_detection=False))
    texts = []
    for r in results:
        j = r.json if hasattr(r, "json") else r
        texts += collect_rec_texts(j)
    return "".join(t for t in texts if t.strip())


def _b64_png(img: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def extract(kind: str, src_path: str, *, render_pages_fn=_default_render,
            page_image_fn=_default_page_image, layout_fn=_default_layout,
            ocr_fn=_default_ocr, seal_fn=_default_seal, dpi: int = 200) -> dict:
    """提取标记。

    kind=seal: 渲染页图 + layout role=seal 框 + 每框章文 → marks=[{page,bbox,seal_text,text:null}]
    kind=signature: 渲染页图 + 全文字框 → marks=[{page,bbox,text,score,cx,cy}]（供调用方按网格取 col0 姓名）
    bbox 为像素坐标（对应返回的页图）。
    """
    with tempfile.TemporaryDirectory() as td:
        pages_meta = render_pages_fn(src_path, td, dpi)
        pages_out, marks = [], []
        page_imgs = {}
        for pg in pages_meta:
            img = page_image_fn(pg["path"])
            page_imgs[pg["page"]] = img
            pages_out.append({"page": pg["page"], "width": pg["width_px"],
                              "height": pg["height_px"], "image_b64": _b64_png(img)})
        if kind == "seal":
            for pg in layout_fn(pages_meta):
                img = page_imgs.get(pg["page"])
                if img is None:
                    continue
                for r in pg["regions"]:
                    x1, y1, x2, y2 = r["bbox_px"]
                    crop = img[y1:y2, x1:x2]
                    marks.append({"page": pg["page"], "bbox": [x1, y1, x2, y2],
                                  "seal_text": seal_fn(crop) if crop.size else "", "text": None})
        else:  # signature
            for pg in pages_meta:
                for ln in ocr_fn(pg["path"]):
                    b = ln.get("bbox") or [int(ln["cx"]), int(ln["top"]),
                                           int(ln["cx"]), int(ln["cy"])]
                    marks.append({"page": pg["page"], "bbox": b, "text": ln["text"],
                                  "score": ln.get("score"), "cx": ln["cx"], "cy": ln["cy"]})
        return {"pages": pages_out, "marks": marks}
