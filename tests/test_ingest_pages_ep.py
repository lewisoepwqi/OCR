"""POST /ingest 新契约：收页图 tar（非 PDF），解包进 scratch/derived/<cid>/pages/，
返回的 derived tar 排除 pages（调用方本就有页图，无需再传回）。
"""
import io
import tarfile
from pathlib import Path

from fastapi.testclient import TestClient

from ocr_service.app import create_app
from common import bundle


def _pages_tar(cid: str) -> bytes:
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / cid / "pages").mkdir(parents=True)
    (d / cid / "pages" / "p1.png").write_bytes(b"\x89PNG\r\n")
    return bundle.pack_dir(d, cid, include=["pages"])


def test_ingest_unpacks_pages_and_returns_without_pages(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path / "scratch"))

    def fake_ingest(cid, *, contracts_root, **kw):
        # 断言页图已解包进 scratch；造出 layout.json 当产物
        pages = Path(contracts_root) / "derived" / cid / "pages" / "p1.png"
        assert pages.exists()
        (Path(contracts_root) / "derived" / cid / "layout.json").write_text("{}")
        return {"ok": True}

    app = create_app(ingest_fn=fake_ingest, warmup=False)
    client = TestClient(app)
    r = client.post("/ingest", files={"file": ("c1.tar", _pages_tar("c1"), "application/x-tar")},
                    data={"contract_id": "c1"})
    assert r.status_code == 200
    names = {m.name for m in tarfile.open(fileobj=io.BytesIO(r.content)).getmembers()}
    assert "c1/layout.json" in names
    assert not any(n.startswith("c1/pages/") for n in names)   # 返回不含 pages
