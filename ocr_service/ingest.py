"""OCR 服务 ingest 兼容入口。

实际工作委托给通用解析编排器；合同字段抽取、印章主体匹配、投影均归后端业务层。
stages 白名单透传：None = 跑满四阶段（默认）；否则只跑声明的阶段（依赖/命名校验在 parser.parse 内兜底）。
"""
from __future__ import annotations

import subprocess

from common.ids import sanitize_id  # re-export：既有测试仍可 ing.sanitize_id
from .parser import GENERIC_STAGES, parse

STAGES = GENERIC_STAGES


def ingest(contract_id: str, *, contracts_root, runner=subprocess.run,
           stages: list[str] | None = None) -> dict:
    return parse(contract_id, contracts_root=contracts_root, runner=runner, stages=stages)
