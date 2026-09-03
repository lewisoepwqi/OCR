"""通用文档解析编排：只负责"看见"，不负责合同字段或判定。

阶段顺序：版面检测 → 文字重建 → 表格识别 → 印章文字识别。
续跑：跑每步前检查其规范产物已存在且可 JSON 解析则跳过（文件是真相源，不维护游标）。
逐步状态写 scratch 内 progress.json 供状态查询接口读取。
所有 stage 在私有 scratch 读写（CR_CONTRACTS_ROOT 重定向）；runner 可注入便于测试。

stages 白名单（可选）：调用方声明本次只跑哪几个阶段，缺省 None = 跑满四阶段
（默认路径与历史行为逐字一致）。校验规则：未知名 / 缺前置一律 StageSpecError（HTTP 层转 400），
绝不静默少跑；未跑的阶段在 progress.json 记 status="skipped-by-request"（区别于续跑的 "skipped"）。
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
# 阶段依赖（DAG 的边）：跑某阶段必须有其全部前置的产物可读。
#   build_document  读 layout.json；recognize_tables / recognize_seals 读 document.json。
# stages 白名单缺前置时不自动补齐（调用方声明「要什么」是成本契约，替他跑没要的阶段
# 等于静默花掉他正要省的算力/显存），而是一律报错让调用方显式声明。
STAGE_PREREQUISITES = {
    "probe_layout": [],
    "build_document": ["probe_layout"],
    "recognize_tables": ["build_document"],
    "recognize_seals": ["build_document"],
}
PROGRESS_FILE = "progress.json"


class StageSpecError(ValueError):
    """stages 声明非法（未知阶段名 / 缺前置依赖）。HTTP 层捕获后转 400。"""


def _validate_stage_list(requested) -> list[str]:
    """校验阶段名列表：未知名报错；缺前置报错。返回按规范顺序去重后的列表。"""
    seen: list[str] = []
    for name in requested:
        if name not in GENERIC_STAGES:
            raise StageSpecError(
                f"未知阶段名：{name!r}（合法阶段：{', '.join(GENERIC_STAGES)}）")
        if name not in seen:
            seen.append(name)
    for name in seen:
        missing = [p for p in STAGE_PREREQUISITES[name] if p not in seen]
        if missing:
            raise StageSpecError(
                f"阶段 {name} 依赖前置阶段 {', '.join(missing)}："
                f"请在 stages 里一并声明，或去掉 {name}（不会自动补跑前置）")
    return [s for s in GENERIC_STAGES if s in seen]


def parse_stages(spec: str | None) -> list[str] | None:
    """把 /ingest 的 stages 表单字段解析成规范顺序的阶段列表。

    None / 空白 / 全空段（如 " , "）→ None（表示跑满四阶段，默认行为）。
    容忍逗号间空白、重复项、多余空段；未知阶段名或缺前置 → StageSpecError。
    """
    if spec is None or not spec.strip():
        return None
    names = [t.strip() for t in spec.split(",")]
    requested = [n for n in names if n]
    if not requested:
        return None
    return _validate_stage_list(requested)


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


def parse(contract_id: str, *, contracts_root, runner=subprocess.run,
          stages: list[str] | None = None) -> dict:
    """按序跑通用解析 stage（已完成步跳过）；全过→{ok}；失败→{error,stage,log}。

    contracts_root：OCR 私有 scratch 根（置 CR_CONTRACTS_ROOT，stage 读写都落它）。
    页图已由调用方业务层渲染落 contracts_root/derived/<cid>/pages/，probe_layout 只 --id 读它，
    本函数不再碰 PDF。逐步状态写 contracts_root/progress.json。

    stages：白名单（已解析的阶段列表）；None = 跑满四阶段（默认路径，行为与历史逐字一致）。
    白名单外的阶段不跑，progress 记 "skipped-by-request"；其产物缺位属正常（打包/续跑都兼容）。
    续跑只对白名单内的阶段查 _artifact_complete——上次带 stages 跑过、这次不带 stages 重试时，
    缺的阶段因产物不存在自然补跑。非法列表（未知名/缺前置）→ StageSpecError（防御：HTTP 层已校验）。
    """
    if stages is not None:
        run_stages = _validate_stage_list(stages)
    else:
        run_stages = None
    env = {**os.environ,
           "DISABLE_MODEL_SOURCE_CHECK": "True",
           "CR_CONTRACTS_ROOT": str(contracts_root)}
    steps: list[dict] = []
    for stage in GENERIC_STAGES:
        if run_stages is not None and stage not in run_stages:
            steps.append({"stage": stage, "status": "skipped-by-request",
                          "started_at": _now(), "finished_at": _now()})
            _write_progress(contracts_root, steps)
            continue
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
