"""通用解析阶段：印章文字识别 → seals.json。

印章区域已由 PP-DocLayoutV2 检出（document.json 里 role=seal）。本步：
裁出每个印章区域 → 喂 paddlex `seal_recognition` 管线（关掉它自带的版面检测，复用其弯曲文字
unwarp+检测+识别）→ 得印章文字。

注意：这里只认章文字，不判断章是不是签约主体或我方主体；subject_match 属合同业务层。

用法：
    .venv/bin/python ocr/recognize_seals.py [--id <合同id>] [--debug]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = Path(os.environ.get("CR_CONTRACTS_ROOT", ROOT / "contracts"))
DERIVED = CONTRACTS / "derived"
PIPELINE_VERSION = "seal-0.1"
PAD = 0.03  # 裁剪外扩比例，避免切到印章边缘弯曲文字


def crop_seal(page_img: Image.Image, bbox_norm: list[int]) -> Image.Image:
    """按归一化 bbox(0–1000) 从页图裁出印章区域（外扩 PAD）。"""
    w, h = page_img.size
    x1, y1, x2, y2 = bbox_norm
    px = (x2 - x1) * PAD / 1000 * w
    py = (y2 - y1) * PAD / 1000 * h
    box = (max(0, x1 / 1000 * w - px), max(0, y1 / 1000 * h - py),
           min(w, x2 / 1000 * w + px), min(h, y2 / 1000 * h + py))
    return page_img.crop(box)


def collect_rec_texts(obj) -> list[str]:
    """递归从 paddlex 结果里收集所有 rec_texts（兼容嵌套结构）。"""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "rec_texts" and isinstance(v, list):
                found += [str(t) for t in v]
            else:
                found += collect_rec_texts(v)
    elif isinstance(obj, list):
        for it in obj:
            found += collect_rec_texts(it)
    return found


def recognize(contract_id: str, debug: bool = False) -> dict:
    derived = DERIVED / contract_id
    document = json.loads((derived / "document.json").read_text(encoding="utf-8"))

    # 收集印章区域
    seal_regions = [(pg["page"], r) for pg in document["pages"]
                    for r in pg["regions"] if r["role"] == "seal"]
    if not seal_regions:
        print("未检出印章区域。")
        return {"contract_id": contract_id, "pipeline_version": PIPELINE_VERSION, "seals": []}

    from paddlex import create_pipeline
    pipe = create_pipeline("seal_recognition")
    seal_dir = derived / "seals"
    seal_dir.mkdir(parents=True, exist_ok=True)

    page_cache: dict[int, Image.Image] = {}
    seals = []
    for page, r in seal_regions:
        if page not in page_cache:
            page_cache[page] = Image.open(derived / "pages" / f"p{page}.png").convert("RGB")
        crop = crop_seal(page_cache[page], r["bbox"])
        crop_path = seal_dir / f"{r['region_id']}.png"
        crop.save(crop_path)

        results = list(pipe.predict(
            str(crop_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=False,  # 关掉自带版面检测：整张裁图当印章区
        ))
        texts: list[str] = []
        for res in results:
            j = res.json if hasattr(res, "json") else res
            texts += collect_rec_texts(j)
        ocr_text = "".join(t for t in texts if t.strip())
        if debug and results:
            print(f"  [debug] {r['region_id']} 结果键:",
                  list((results[0].json.get("res", results[0].json)).keys())
                  if isinstance(results[0].json, dict) else type(results[0].json))

        seals.append({
            "seal_id": r["region_id"].replace("_r", "_s"),
            "page": page,
            "bbox": r["bbox"],
            "ocr_text": ocr_text,
            "source_region": r["region_id"],
        })
        print(f"  {r['region_id']} 第{page}页: 印章文字={ocr_text!r}")

    return {"contract_id": contract_id, "pipeline_version": PIPELINE_VERSION, "seals": seals}


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
    derived = DERIVED / contract_id

    print(f"印章文字识别 + 主体校验：{contract_id}")
    result = recognize(contract_id, args.debug)
    (derived / "seals.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写 {derived/'seals.json'}")

    print(f"\n印章数={len(result['seals'])}；主体一致性校验由合同业务层处理。")


if __name__ == "__main__":
    main()
