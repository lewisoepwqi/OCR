"""通用文档解析编排：只负责"看见"，不负责合同字段或判定。

阶段顺序：版面检测 → 文字重建 → 表格识别 → 印章文字识别。
续跑：跑每步前检查其规范产物已存在且可 JSON 解析则跳过（文件是真相源，不维护游标）。
逐步状态写 scratch 内 progress.json 供状态查询接口读取。
所有 stage 在私有 scratch 读写（CR_CONTRACTS_ROOT 重定向）；runner 可注入便于测试。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERIC_STAGES = ["probe_layout", "build_document", "recognize_tables", "recognize_seals"]
# 各 stage 的规范完成产物（相对 derived/<cid>/）；存在且可 JSON 解析即视为已完成。
EXPECTED_ARTIFACT = {
    "probe_layout": "layout.json",
    "build_document": "document.json",
    "recognize_tables": "tables.json",
    "recognize_seals": "seals.json",
}
PROGRESS_FILE = "progress.json"


def _now() -> str:
    return datetime.now().isoformat()


def read_progress(contracts_root) -> list[dict]:
    """读 scratch 内 progress.json；不存在则空列表。"""
    p = Path(contracts_root) / PROGRESS_FILE
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _write_progress(contracts_root, steps: list[dict]) -> None:
    (Path(contracts_root) / PROGRESS_FILE).write_text(
        json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")


def _artifact_complete(contracts_root, contract_id: str, stage: str) -> bool:
    """规范产物存在且可 JSON 解析 → 该步已完成（完整性校验，半写文件不算数）。"""
    art = Path(contracts_root) / "derived" / contract_id / EXPECTED_ARTIFACT[stage]
    if not art.exists():
        return False
    try:
        json.loads(art.read_text(encoding="utf-8"))
        return True
    except (json.JSONDecodeError, OSError):
        return False


def parse(contract_id: str, *, contracts_root, runner=subprocess.run) -> dict:
    """按序跑通用解析 stage（已完成步跳过）；全过→{ok}；失败→{error,stage,log}。

    contracts_root：OCR 私有 scratch 根（置 CR_CONTRACTS_ROOT，stage 读写都落它）。
    页图已由调用方业务层渲染落 contracts_root/derived/<cid>/pages/，probe_layout 只 --id 读它，
    本函数不再碰 PDF。逐步状态写 contracts_root/progress.json。
    """
    env = {**os.environ,
           "DISABLE_MODEL_SOURCE_CHECK": "True",
           "CR_CONTRACTS_ROOT": str(contracts_root)}
    steps: list[dict] = []
    for stage in GENERIC_STAGES:
        if _artifact_complete(contracts_root, contract_id, stage):
            steps.append({"stage": stage, "status": "skipped",
                          "started_at": _now(), "finished_at": _now()})
            _write_progress(contracts_root, steps)
            continue
        entry = {"stage": stage, "status": "running", "started_at": _now(), "finished_at": None}
        steps.append(entry)
        _write_progress(contracts_root, steps)

        cmd = [sys.executable, str(ROOT / "ocr" / f"{stage}.py"), "--id", contract_id]
        res = runner(cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            log = (res.stderr or res.stdout or "")[-2000:]
            entry["status"] = "error"
            entry["finished_at"] = _now()
            _write_progress(contracts_root, steps)
            return {"error": f"stage {stage} 失败", "stage": stage, "log": log}
        entry["status"] = "done"
        entry["finished_at"] = _now()
        _write_progress(contracts_root, steps)
    return {"ok": True}
