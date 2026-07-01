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


def test_prune_skips_non_directory_files(tmp_path):
    """prune_scratch 应跳过非目录文件，只清理过期目录。"""
    # 创建普通文件
    stray_file = tmp_path / "stray.txt"
    stray_file.write_text("x")

    # 创建过期目录
    old_dir = tmp_path / "old_cid"
    old_dir.mkdir()
    past = time.time() - 50 * 3600
    os.utime(old_dir, (past, past))

    removed = prune_scratch(tmp_path, ttl_hours=48)
    assert removed == ["old_cid"]
    assert stray_file.exists()  # 普通文件不应被清
    assert not old_dir.exists()  # 过期目录清掉了
