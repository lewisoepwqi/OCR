"""阶段 3：表格处理——简单表转 Markdown，复杂表（合并单元格/识别差）标 needs_review → tables/t*.json。

表格区域已由 PP-DocLayoutV2 检出（document.json 里 role=table）。本步：
裁表格区 → 喂 paddlex `table_recognition_v2` 管线（关其自带版面检测，复用结构识别+单元格检测+OCR）
→ 得表格 HTML → 转 Markdown；含合并单元格(rowspan/colspan) 或解析过少 → complex/needs_review。

用法：
    .venv/bin/python ocr/recognize_tables.py [--id <合同id>] [--debug]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = Path(os.environ.get("CR_CONTRACTS_ROOT", ROOT / "contracts"))
DERIVED = CONTRACTS / "derived"
PIPELINE_VERSION = "table-0.1"
PAD = 0.01


class _TableHTMLParser(HTMLParser):
    """把表格 HTML 解析成行/单元格，并标记是否含合并单元格。"""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] = []
        self._in_cell = False
        self.has_span = False

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell = []
            if int(d.get("rowspan", "1") or 1) > 1 or int(d.get("colspan", "1") or 1) > 1:
                self.has_span = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._in_cell = False
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)


def html_to_markdown(html: str) -> tuple[str, bool, int]:
    """HTML 表 → Markdown。返回 (markdown, has_span 合并单元格, 行数)。"""
    p = _TableHTMLParser()
    p.feed(html or "")
    rows = [r for r in p.rows if any(c for c in r)]
    if not rows:
        return "", p.has_span, 0
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    md = ["| " + " | ".join(c.replace("|", "\\|") for c in rows[0]) + " |",
          "| " + " | ".join(["---"] * ncol) + " |"]
    for r in rows[1:]:
        md.append("| " + " | ".join(c.replace("|", "\\|") for c in r) + " |")
    return "\n".join(md), p.has_span, len(rows)


def collect_html(obj) -> list[str]:
    """递归收集 paddlex 表格结果里的 pred_html。"""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "pred_html" and isinstance(v, str):
                found.append(v)
            else:
                found += collect_html(v)
    elif isinstance(obj, list):
        for it in obj:
            found += collect_html(it)
    return found


def crop_region(page_img: Image.Image, bbox_norm: list[int]) -> Image.Image:
    w, h = page_img.size
    x1, y1, x2, y2 = bbox_norm
    px = (x2 - x1) * PAD / 1000 * w
    py = (y2 - y1) * PAD / 1000 * h
    return page_img.crop((max(0, x1 / 1000 * w - px), max(0, y1 / 1000 * h - py),
                          min(w, x2 / 1000 * w + px), min(h, y2 / 1000 * h + py)))


def recognize(contract_id: str, debug: bool = False) -> list[dict]:
    derived = DERIVED / contract_id
    document = json.loads((derived / "document.json").read_text(encoding="utf-8"))
    table_regions = [(pg["page"], r) for pg in document["pages"]
                     for r in pg["regions"] if r["role"] == "table"]
    if not table_regions:
        print("未检出表格区域。")
        return []

    from paddlex import create_pipeline
    pipe = create_pipeline("table_recognition_v2")
    tdir = derived / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    page_cache: dict[int, Image.Image] = {}
    tables = []
    for page, r in table_regions:
        if page not in page_cache:
            page_cache[page] = Image.open(derived / "pages" / f"p{page}.png").convert("RGB")
        crop = crop_region(page_cache[page], r["bbox"])
        crop_path = tdir / f"{r['region_id']}.png"
        crop.save(crop_path)

        results = list(pipe.predict(str(crop_path), use_doc_orientation_classify=False,
                                    use_doc_unwarping=False, use_layout_detection=False))
        htmls: list[str] = []
        for res in results:
            htmls += collect_html(res.json if hasattr(res, "json") else res)
        if debug and results:
            j = results[0].json
            print(f"  [debug] {r['region_id']} keys:",
                  list((j.get("res", j)).keys()) if isinstance(j, dict) else type(j))

        html = max(htmls, key=len) if htmls else ""
        markdown, has_span, nrow = html_to_markdown(html)
        complexity = "complex" if (has_span or nrow < 2) else "simple"
        status = "needs_review" if complexity == "complex" else "ok"
        table_id = r["region_id"].replace("_r", "_t")
        tables.append({
            "table_id": table_id, "page": page, "bbox": r["bbox"],
            "complexity": complexity, "status": status,
            "markdown": markdown, "source_region": r["region_id"],
            "pipeline_version": PIPELINE_VERSION,
        })
        (tdir / f"{table_id}.json").write_text(
            json.dumps(tables[-1], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {r['region_id']} 第{page}页: {complexity}/{status} 行数={nrow} "
              f"{'(含合并单元格)' if has_span else ''}")
        if debug and markdown:
            print("\n".join("    " + ln for ln in markdown.splitlines()[:6]))

    return tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if args.id:
        contract_id = args.id
    else:
        cands = sorted(p.parent.name for p in DERIVED.glob("*/document.json"))
        if not cands:
            sys.exit(f"没有 document.json：{DERIVED}")
        contract_id = cands[0]

    print(f"表格识别（table_recognition_v2）：{contract_id}")
    tables = recognize(contract_id, args.debug)
    simple = sum(1 for t in tables if t["complexity"] == "simple")
    print(f"\n表格数={len(tables)}：简单 {simple}（转 Markdown）/ 复杂 {len(tables)-simple}（needs_review）")
    print(f"产出在 {DERIVED/contract_id/'tables'}/")


if __name__ == "__main__":
    main()
