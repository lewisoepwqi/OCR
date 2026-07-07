"""/ingest scratch 持久化：成功清理并返 tar；失败保留 scratch 并返结构化 JSON。

新契约（Task 8）：/ingest 收页图 tar（非 PDF），故这里的 _post 造一份最小页图 tar 作输入。
"""
import io
import tempfile
import tarfile
from pathlib import Path

from fastapi.testclient import TestClient

from ocr_service.app import create_app, scratch_root
from common import bundle


def _pages_tar(cid: str) -> bytes:
    d = Path(tempfile.mkdtemp())
    (d / cid / "pages").mkdir(parents=True)
    (d / cid / "pages" / "p1.png").write_bytes(b"\x89PNG\r\n")
    return bundle.pack_dir(d, cid, include=["pages"])


def _post(c, cid="c1"):
    return c.post("/ingest", files={"file": (f"{cid}.tar", _pages_tar(cid), "application/x-tar")},
                  data={"contract_id": cid})


def test_success_returns_tar_and_cleans_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path))

    def fake_ingest(cid, *, contracts_root):
        # 假 parse：造出 derived/<cid>/ 一个文件，返回成功
        d = Path(contracts_root) / "derived" / cid
        d.mkdir(parents=True, exist_ok=True)
        (d / "document.json").write_text("{}", encoding="utf-8")
        return {"ok": True}

    app = create_app(ingest_fn=fake_ingest, warmup=False)
    with TestClient(app) as c:
        r = _post(c)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/x-tar"
    # tar 内含 c1/document.json；返回排除 pages
    names = tarfile.open(fileobj=io.BytesIO(r.content)).getnames()
    assert "c1/document.json" in names
    assert not any(n.startswith("c1/pages") for n in names)
    assert not (Path(tmp_path) / "c1").exists()   # 成功后 scratch 已清


def test_failure_keeps_scratch_and_returns_structured_json(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path))

    def fake_ingest(cid, *, contracts_root):
        # 造出前一步产物（模拟部分完成），返回结构化失败
        d = Path(contracts_root) / "derived" / cid
        d.mkdir(parents=True, exist_ok=True)
        (d / "layout.json").write_text("{}", encoding="utf-8")
        return {"error": "stage build_document 失败", "stage": "build_document", "log": "boom"}

    app = create_app(ingest_fn=fake_ingest, warmup=False)
    with TestClient(app) as c:
        r = _post(c)
    assert r.status_code == 500
    body = r.json()
    assert body["stage"] == "build_document" and body["log"] == "boom"
    assert "error" in body
    # 失败保留 scratch + 已完成产物（含已解包的页图 + 部分完成的 derived 产物）
    assert (Path(tmp_path) / "c1" / "derived" / "c1" / "pages" / "p1.png").exists()
    assert (Path(tmp_path) / "c1" / "derived" / "c1" / "layout.json").exists()


def test_scratch_root_default(monkeypatch):
    monkeypatch.delenv("OCR_SCRATCH_ROOT", raising=False)
    assert scratch_root().name == "cr-ocr-scratch"


def test_bad_ttl_env_does_not_block_ingest(tmp_path, monkeypatch):
    """OCR_SCRATCH_TTL_HOURS 非数字时不阻断 /ingest，应返 200。"""
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path))
    monkeypatch.setenv("OCR_SCRATCH_TTL_HOURS", "abc")  # 坏 TTL

    def fake_ingest(cid, *, contracts_root):
        # 造出 derived/<cid>/ 一个文件，返回成功
        d = Path(contracts_root) / "derived" / cid
        d.mkdir(parents=True, exist_ok=True)
        (d / "document.json").write_text("{}", encoding="utf-8")
        return {"ok": True}

    app = create_app(ingest_fn=fake_ingest, warmup=False)
    with TestClient(app) as c:
        r = _post(c)
    # 虽然 TTL 坏，但入库仍应成功（清理是尽力而为）
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/x-tar"
