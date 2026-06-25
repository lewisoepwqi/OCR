"""阶段 B：对版面区域做 PP-OCRv6 文字识别 → 重建全文 → document.json + text.md。

依赖阶段 A 的产出 `derived/<id>/layout.json`（区域 + 归一化 bbox + role）与 `pages/p*.png`。
做法：整页 OCR（PP-OCRv6_medium）→ 把每行文字按中心点归到所属 region → 区域内按阅读顺序拼接 →
写 document.json（schema §2：区域 + 文本 + 双锁占位 + provenance）与 text.md（重建全文、剔页眉脚、带页码）。

用法：
    .venv/bin/python ocr/build_document.py [--id <合同id>]
不给 --id 则取 contracts/derived/ 下唯一/第一个有 layout.json 的合同。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = Path(os.environ.get("CR_CONTRACTS_ROOT", ROOT / "contracts"))
DERIVED = CONTRACTS / "derived"

# OCR 模型：固定 PP-OCRv6_medium（精度优先，见记忆 ocr-must-use-pp-ocrv6）。
OCR_DET_MODEL = "PP-OCRv6_medium_det"
OCR_REC_MODEL = "PP-OCRv6_medium_rec"
PIPELINE_VERSION = "ingest-0.1"
# 进重建全文 text.md 的 role（剔除 header/footer/table/seal/figure）
FULLTEXT_ROLES = {"text", "title"}


def build_ocr():
    """构造 PP-OCRv6_medium OCR 引擎（关文档方向/摆正，留文本行方向分类）。"""
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name=OCR_DET_MODEL,
        text_recognition_model_name=OCR_REC_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )


def ocr_page(ocr, img_path: Path) -> list[dict]:
    """对单页图做 OCR，返回每行 {text, score, cx, cy, top}（坐标为像素）。"""
    results = list(ocr.predict(str(img_path)))
    if not results:
        return []
    j = results[0].json if hasattr(results[0], "json") else results[0]
    data = j.get("res", j) if isinstance(j, dict) else {}
    texts = data.get("rec_texts", []) or []
    scores = data.get("rec_scores", []) or []
    boxes = data.get("rec_boxes", data.get("rec_polys", [])) or []
    lines: list[dict] = []
    for t, s, b in zip(texts, scores, boxes):
        try:
            b = b.tolist()
        except Exception:
            pass
        # rec_boxes 为轴对齐 [x1,y1,x2,y2]；rec_polys 为 4 点，取外接框
        if len(b) == 4 and not isinstance(b[0], (list, tuple)):
            x1, y1, x2, y2 = b
        else:
            xs = [p[0] for p in b]
            ys = [p[1] for p in b]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if not str(t).strip():
            continue
        lines.append({"text": str(t), "score": float(s),
                      "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "top": y1})
    return lines


def assign_line_to_region(nx: float, ny: float, regions: list[dict]) -> int | None:
    """把归一化中心点 (nx,ny) 归到包含它的最小面积 region；都不含则返回 None。"""
    best, best_area = None, None
    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r["bbox"]
        if x1 <= nx <= x2 and y1 <= ny <= y2:
            area = (x2 - x1) * (y2 - y1)
            if best_area is None or area < best_area:
                best, best_area = i, area
    return best


def nearest_region(nx: float, ny: float, regions: list[dict]) -> int | None:
    """落在所有区域外的行，归到中心最近的 region（兜底，避免丢字）。"""
    best, best_d = None, None
    for i, r in enumerate(regions):
        cx = (r["bbox"][0] + r["bbox"][2]) / 2
        cy = (r["bbox"][1] + r["bbox"][3]) / 2
        d = (nx - cx) ** 2 + (ny - cy) ** 2
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


def build_document(contract_id: str, derived_root: Path | None = None) -> dict:
    derived = (DERIVED if derived_root is None else Path(derived_root)) / contract_id
    layout = json.loads((derived / "layout.json").read_text(encoding="utf-8"))
    ocr = build_ocr()

    doc_pages = []
    for pg in layout["pages"]:
        w, h = pg["width_px"], pg["height_px"]
        regions = pg["regions"]
        buckets: dict[int, list[dict]] = {i: [] for i in range(len(regions))}
        loose = 0

        lines = ocr_page(ocr, derived / "pages" / f"p{pg['page']}.png")
        for ln in lines:
            nx, ny = ln["cx"] / w * 1000, ln["cy"] / h * 1000
            idx = assign_line_to_region(nx, ny, regions)
            if idx is None:
                idx = nearest_region(nx, ny, regions)
                loose += 1
            if idx is not None:
                buckets[idx].append(ln)

        out_regions = []
        for i, r in enumerate(regions):
            rl = sorted(buckets[i], key=lambda x: x["top"])
            text = "".join(x["text"] for x in rl)  # 中文行内直接拼接
            ocr_conf = round(min((x["score"] for x in rl), default=0.0), 3)  # 取最低行，保守
            out_regions.append({
                "region_id": r["region_id"],
                "bbox": r["bbox"],
                "role": r["role"],
                "text": text,
                "lock_layout": False,
                "lock_text": False,
                "role_source": "machine",
                "text_source": "machine",
                "provenance": {
                    "ocr_conf": ocr_conf,
                    "layout_label": r.get("label"),
                    "layout_score": r.get("score"),
                    "pipeline_version": PIPELINE_VERSION,
                },
            })
        doc_pages.append({"page": pg["page"], "width_px": w, "height_px": h, "regions": out_regions})
        print(f"      第 {pg['page']} 页：{len(lines)} 行 → {len(regions)} 区域"
              f"（区域外兜底 {loose} 行）")

    return {"contract_id": contract_id, "pipeline_version": PIPELINE_VERSION, "pages": doc_pages}


def build_fulltext_md(document: dict) -> str:
    """重建全文：剔 header/footer/table/seal/figure，按页+阅读顺序拼，带页码标记供 grep 定位。"""
    L = [f"# 重建全文 · {document['contract_id']}", ""]
    for pg in document["pages"]:
        L.append(f"<!-- 第 {pg['page']} 页 -->")
        for r in pg["regions"]:
            if r["role"] in FULLTEXT_ROLES and r["text"].strip():
                L.append(r["text"])
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id")
    args = ap.parse_args()

    if args.id:
        contract_id = args.id
    else:
        cands = sorted(p.parent.name for p in DERIVED.glob("*/layout.json"))
        if not cands:
            sys.exit(f"没有可用的 layout.json（先跑 ocr/probe_layout.py）：{DERIVED}")
        contract_id = cands[0]
    derived = DERIVED / contract_id

    print(f"[1/2] OCR 文字识别（PP-OCRv6_medium）+ 组装 document.json：{contract_id}")
    document = build_document(contract_id)
    (derived / "document.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      已写 {derived/'document.json'}")

    print("[2/2] 重建全文 text.md")
    md = build_fulltext_md(document)
    (derived / "text.md").write_text(md, encoding="utf-8")
    print(f"      已写 {derived/'text.md'}")

    # 小结
    n_text = sum(1 for pg in document["pages"] for r in pg["regions"]
                 if r["role"] in FULLTEXT_ROLES and r["text"].strip())
    low = [(r["region_id"], r["provenance"]["ocr_conf"]) for pg in document["pages"]
           for r in pg["regions"] if r["text"].strip() and r["provenance"]["ocr_conf"] < 0.9]
    print(f"\n小结：正文/标题区域有文本 {n_text} 个；低置信(<0.9) {len(low)} 个 → {low[:8]}")


if __name__ == "__main__":
    main()
