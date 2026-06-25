"""阶段 A 探针：PDF → 页图 → PP-DocLayoutV2 版面检测 → 打印区域尺寸报告。

本脚本只为回答开放项 1（"版面区域实际有多大、区域≈段落成不成立、要不要按行间距拆段落"），
**不做 OCR 文字识别、不做字段抽取**。严格遵守 kickoff_prompt「先看尺寸再往下做」。

用法：
    .venv/bin/python -m ocr.probe_layout                 # 自动取 contracts/raw/ 下第一个 PDF
    .venv/bin/python -m ocr.probe_layout --pdf <路径> --id <合同id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = Path(os.environ.get("CR_CONTRACTS_ROOT", ROOT / "contracts"))
MODEL_DIR = ROOT / "models" / "pp_doclayout_v2"
# 版面检测设备：默认 gpu（独立 OCR 仓部署在 GPU 服务器）；纯 CPU 环境置 OCR_DEVICE=cpu 兜底。
LAYOUT_DEVICE = os.environ.get("OCR_DEVICE", "gpu")
DATA_RAW = CONTRACTS / "raw"
DATA_DERIVED = CONTRACTS / "derived"
RENDER_DPI = 200
SCORE_THRESHOLD = 0.5
PIPELINE_VERSION = "probe-layout-0.1"

# PP-DocLayoutV2 原始类别 → 本项目 7 类 role（text|title|table|seal|figure|header|footer）。
# 未列出的类别归到最接近的 role 或 "text" 兜底。
_LABEL_TO_ROLE = {
    "text": "text",
    "abstract": "text",
    "content": "text",
    "reference": "text",
    "aside_text": "text",
    "footnote": "footer",
    "paragraph_title": "title",
    "doc_title": "title",
    "title": "title",
    "chart_title": "title",
    "figure_title": "title",
    "table_title": "title",
    "table": "table",
    "figure": "figure",
    "image": "figure",
    "chart": "figure",
    "formula": "figure",
    "formula_number": "figure",
    "seal": "seal",
    "header": "header",
    "header_image": "header",
    "footer": "footer",
    "footer_image": "footer",
    "number": "footer",  # 页码
}


def map_role(label: str) -> str:
    """把 PP-DocLayoutV2 的原始类别名映射到本项目 role。"""
    return _LABEL_TO_ROLE.get(label.strip().lower(), "text")


def find_pdf(explicit: str | None) -> Path:
    """定位待处理 PDF：优先 --pdf，否则取 contracts/raw/ 下第一个。"""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            sys.exit(f"找不到 PDF：{p}")
        return p
    pdfs = sorted(DATA_RAW.rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"contracts/raw/ 下没有 PDF：{DATA_RAW}")
    if len(pdfs) > 1:
        print(f"[提示] raw/ 下有 {len(pdfs)} 个 PDF，取第一个：{pdfs[0].name}")
    return pdfs[0]


def render_pages(pdf_path: Path, out_dir: Path, dpi: int) -> list[dict]:
    """用 pypdfium2 渲染每页为 PNG（raw/ 原件只读，只读取不写回）。"""
    import pypdfium2 as pdfium  # 延迟导入，等依赖装好后才用

    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        scale = dpi / 72.0
        for i in range(len(pdf)):
            page = pdf[i]
            pil = page.render(scale=scale).to_pil()
            png_path = out_dir / f"p{i + 1}.png"
            pil.save(png_path)
            pages.append({"page": i + 1, "path": png_path,
                          "width_px": pil.width, "height_px": pil.height})
    finally:
        pdf.close()
    return pages


def extract_boxes(result) -> list[dict]:
    """从 paddlex 检测结果里取出 boxes（兼容 dict-like 与 .json 两种取法）。"""
    boxes = None
    try:
        boxes = result["boxes"]
    except Exception:
        try:
            boxes = result.json["res"]["boxes"]
        except Exception:
            boxes = []
    return boxes or []


def detect_layout(pages: list[dict], score_threshold: float) -> list[dict]:
    """对每页跑 PP-DocLayoutV2，返回归一化后的区域（按阅读顺序：上→下、左→右）。"""
    from paddlex import create_model  # 延迟导入

    model = create_model(model_name="PP-DocLayoutV2", model_dir=str(MODEL_DIR), device=LAYOUT_DEVICE)
    out_pages: list[dict] = []
    for pg in pages:
        results = model.predict(str(pg["path"]), batch_size=1)
        raw = []
        for res in results:
            raw.extend(extract_boxes(res))
        raw = [b for b in raw if float(b.get("score", 0.0)) >= score_threshold]
        raw.sort(key=lambda b: (b["coordinate"][1], b["coordinate"][0]))

        w, h = pg["width_px"], pg["height_px"]
        regions = []
        for idx, b in enumerate(raw, start=1):
            x1, y1, x2, y2 = b["coordinate"]
            label = str(b.get("label", ""))
            regions.append({
                "region_id": f"p{pg['page']}_r{idx:02d}",
                "bbox": [round(x1 / w * 1000), round(y1 / h * 1000),
                         round(x2 / w * 1000), round(y2 / h * 1000)],
                "role": map_role(label),
                "label": label,
                "score": round(float(b.get("score", 0.0)), 3),
            })
        out_pages.append({"page": pg["page"], "width_px": w, "height_px": h, "regions": regions})
        print(f"      第 {pg['page']} 页：{len(regions)} 个区域")
    return out_pages


def build_report(layout: dict, assumed_line_norm: float = 18.0) -> str:
    """生成区域尺寸报告（assumed_line_norm：假设单行文字归一化高度 ‰，用于粗估行数）。"""
    from statistics import mean, median

    L: list[str] = [f"# 版面区域尺寸报告 · {layout['contract_id']}", ""]
    text_heights: list[int] = []
    seal_pages, hf_total = [], 0

    for pg in layout["pages"]:
        regions = pg["regions"]
        roles: dict[str, int] = {}
        for r in regions:
            roles[r["role"]] = roles.get(r["role"], 0) + 1
        L.append(f"## 第 {pg['page']} 页（{pg['width_px']}×{pg['height_px']}px）　区域 {len(regions)}　{roles}")
        for r in regions:
            if r["role"] != "text":
                continue
            x1, y1, x2, y2 = r["bbox"]
            h, w = y2 - y1, x2 - x1
            est = max(1, round(h / assumed_line_norm))
            text_heights.append(h)
            L.append(f"  - {r['region_id']} [{r['label']}] 高={h}‰ 宽={w}‰ ≈{est}行 score={r['score']}")
        hf = [r for r in regions if r["role"] in ("header", "footer")]
        hf_total += len(hf)
        if hf:
            L.append("  · 页眉脚：" + ", ".join(f"{r['region_id']}({r['role']},y={r['bbox'][1]}‰)" for r in hf))
        if any(r["role"] == "seal" for r in regions):
            seal_pages.append(pg["page"])

    L += ["", "## 汇总"]
    if text_heights:
        L.append(f"- text 区域高度(‰)：n={len(text_heights)} 最小={min(text_heights)} "
                 f"中位={median(text_heights):.0f} 均值={mean(text_heights):.0f} 最大={max(text_heights)}")
        L.append("- 判断：多数 text 区域估行数 >1 → 区域≈多段，需在 text 区域内按行间距拆段落；"
                 "多在 1 行上下 → 区域≈段落，可不拆。")
    L.append(f"- 页眉/页脚区域共检出 {hf_total} 个（核对是否误判正文）")
    L.append(f"- 印章(seal)检出页：{seal_pages or '无'}（验证 v3 印章检测是否生效）")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf")
    ap.add_argument("--id")
    args = ap.parse_args()

    pdf_path = find_pdf(args.pdf)
    contract_id = args.id or pdf_path.stem
    derived = DATA_DERIVED / contract_id

    print(f"[1/3] 渲染页图：{pdf_path.name} → {derived/'pages'}（DPI={RENDER_DPI}）")
    pages = render_pages(pdf_path, derived / "pages", RENDER_DPI)
    print(f"      共 {len(pages)} 页")

    print("[2/3] 版面检测：PP-DocLayoutV2（CPU）")
    out_pages = detect_layout(pages, SCORE_THRESHOLD)
    layout = {"contract_id": contract_id, "pipeline_version": PIPELINE_VERSION, "pages": out_pages}
    (derived / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      已写 {derived/'layout.json'}")

    print("[3/3] 生成区域尺寸报告")
    report = build_report(layout)
    (derived / "layout_report.md").write_text(report, encoding="utf-8")
    print("\n" + report + "\n")
    print(f"报告已写 {derived/'layout_report.md'}")


if __name__ == "__main__":
    main()
