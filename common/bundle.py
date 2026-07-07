"""中立 tar 打包/解包：OCR↔后端的 derived 交接载体。逐成员防 zip-slip。

放中立 common/，后端与 OCR 服务两边共用——OCR（通用层）不反向 import 业务层 webapi。
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path


class BundleError(Exception):
    """打包/解包失败或检测到非法成员（路径穿越/前缀不符）。"""


def pack_dir(parent: Path, name: str, include: list[str] | None = None,
             exclude: list[str] | None = None) -> bytes:
    """把 parent/name/ 打成 tar（归档名以 name/ 为前缀）。

    include 给定时只打 parent/name/<inc> 这些顶层项；
    exclude 给定时打全部顶层项但跳过这些名字（用于 /ingest 返回时排除 pages，调用方已有）。
    include 与 exclude 互斥。页图本就 PNG 压缩，不再 gzip。
    """
    if include is not None and exclude is not None:
        raise ValueError("include 与 exclude 互斥，不能同时给")
    base = Path(parent) / name
    if not base.is_dir():
        raise BundleError(f"待打包目录不存在：{base}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        if include is not None:
            for inc in include:
                p = base / inc
                if p.exists():
                    tf.add(p, arcname=f"{name}/{inc}")
        elif exclude is not None:
            skip = set(exclude)
            for child in sorted(base.iterdir()):
                if child.name in skip:
                    continue
                tf.add(child, arcname=f"{name}/{child.name}")
        else:
            tf.add(base, arcname=name)
    return buf.getvalue()


def unpack_dir(data: bytes, dest_parent: Path, name: str) -> None:
    """解 tar 到 dest_parent/，每个成员必须落在 dest_parent/name/ 之内，否则 BundleError。"""
    dest_parent = Path(dest_parent).resolve()
    safe_root = (dest_parent / name).resolve()
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r")
    except tarfile.TarError as e:
        raise BundleError(f"非法 tar：{e}") from e
    with tf:
        members = tf.getmembers()
        for m in members:
            # 拒绝链接成员（符号链接/硬链接），防止 linkname 指向沙箱外
            if m.issym() or m.islnk():
                raise BundleError(f"不允许链接成员：{m.name}")
            target = (dest_parent / m.name).resolve()
            # 必须严格落在 dest_parent/name/ 之内（含等于 safe_root 自身）
            if target != safe_root and safe_root not in target.parents:
                raise BundleError(f"非法成员路径（疑似穿越/前缀不符）：{m.name}")
        # filter="data"：标准库纵深防护（拒绝绝对路径/链接/设备、钳制权限），与上方逐成员校验互补
        tf.extractall(dest_parent, members=members, filter="data")
