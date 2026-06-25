"""通用文档解析编排：只负责"看见"，不负责合同字段或判定。

阶段顺序：版面检测 → 文字重建 → 表格识别 → 印章文字识别。
所有 stage 在私有 scratch 读写（CR_CONTRACTS_ROOT 重定向）；runner 可注入便于测试。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERIC_STAGES = ["probe_layout", "build_document", "recognize_tables", "recognize_seals"]


def parse(contract_id: str, *, contracts_root, raw_pdf=None, runner=subprocess.run) -> dict:
    """按序跑通用解析 stage；全过→{ok}；失败→{error,stage,log}。

    contracts_root：OCR 私有 scratch 根（置 CR_CONTRACTS_ROOT，stage 读写都落它）。
    raw_pdf：scratch 内源 PDF 路径，传给 probe_layout --pdf。
    """
    env = {**os.environ,
           "DISABLE_MODEL_SOURCE_CHECK": "True",
           "CR_CONTRACTS_ROOT": str(contracts_root)}
    for stage in GENERIC_STAGES:
        cmd = [sys.executable, str(ROOT / "ocr" / f"{stage}.py"), "--id", contract_id]
        if stage == "probe_layout" and raw_pdf is not None:
            cmd += ["--pdf", str(raw_pdf)]
        res = runner(cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            log = (res.stderr or res.stdout or "")[-2000:]
            return {"error": f"stage {stage} 失败", "stage": stage, "log": log}
    return {"ok": True}
