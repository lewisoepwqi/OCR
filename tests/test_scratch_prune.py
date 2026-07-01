"""scratch TTL 清理：过期目录清掉，未过期/新写的保留。"""
import os
import time
from pathlib import Path

from ocr_service.app import prune_scratch


def test_prune_removes_only_expired(tmp_path):
    old = tmp_path / "old_cid"
    old.mkdir()
    fresh = tmp_path / "fresh_cid"
    fresh.mkdir()
    # 把 old 的 mtime 拨到 50 小时前
    past = time.time() - 50 * 3600
    os.utime(old, (past, past))

    removed = prune_scratch(tmp_path, ttl_hours=48)
    assert removed == ["old_cid"]
    assert not old.exists()
    assert fresh.exists()


def test_prune_missing_root_is_noop(tmp_path):
    assert prune_scratch(tmp_path / "does-not-exist", ttl_hours=48) == []
