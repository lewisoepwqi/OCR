"""中立 id 工具：净化 + 语法校验。后端与 OCR 服务两边共用，互不反向依赖。"""
from __future__ import annotations

import re
from pathlib import Path

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def is_valid_id_syntax(cid) -> bool:
    """语法闸：仅字母数字下划线连字符，杜绝 `/`、`..`、空白、非 ASCII。"""
    return isinstance(cid, str) and bool(_ID_RE.match(cid))


def sanitize_id(name) -> str | None:
    """取文件名 stem，仅保留 [A-Za-z0-9_-]；空或无合法字符 → None（防穿越/非法 id）。"""
    if not name:
        return None
    stem = Path(str(name)).stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", stem)
    return cleaned if cleaned and _ID_RE.match(cleaned) else None
