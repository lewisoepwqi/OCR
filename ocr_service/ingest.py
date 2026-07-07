"""OCR 服务 ingest 兼容入口。

实际工作委托给通用解析编排器；合同字段抽取、印章主体匹配、投影均归后端业务层。
"""
from __future__ import annotations

import subprocess

from common.ids import sanitize_id  # re-export：既有测试仍可 ing.sanitize_id
from .parser import GENERIC_STAGES, parse

STAGES = GENERIC_STAGES


def ingest(contract_id: str, *, contracts_root, runner=subprocess.run) -> dict:
    return parse(contract_id, contracts_root=contracts_root, runner=runner)
