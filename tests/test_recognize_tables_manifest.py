"""recognize_tables 归一化：无论有无表格都落 derived/<cid>/tables.json（续跑标记）。"""
import json
from pathlib import Path

from ocr.recognize_tables import write_manifest


def test_write_manifest_empty(tmp_path: Path):
    derived = tmp_path / "derived" / "c1"
    derived.mkdir(parents=True)
    write_manifest(derived, [])
    data = json.loads((derived / "tables.json").read_text(encoding="utf-8"))
    assert data == {"tables": []}


def test_write_manifest_slim_index(tmp_path: Path):
    derived = tmp_path / "derived" / "c1"
    derived.mkdir(parents=True)
    tables = [{"table_id": "p1_t0", "page": 1, "status": "ok", "complexity": "simple",
               "markdown": "|a|b|", "bbox": [0, 0, 1, 1], "source_region": "p1_r0"}]
    write_manifest(derived, tables)
    data = json.loads((derived / "tables.json").read_text(encoding="utf-8"))
    # 只保留轻量索引字段，不复制 markdown/bbox（明细在 tables/<id>.json）
    assert data == {"tables": [{"table_id": "p1_t0", "page": 1,
                                "status": "ok", "complexity": "simple"}]}
