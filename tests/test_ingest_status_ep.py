"""GET /ingest/status/{cid}：读 scratch 内 progress.json。"""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from ocr_service.app import create_app


def test_status_reflects_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path))
    scratch = tmp_path / "c1"
    scratch.mkdir(parents=True)
    steps = [{"stage": "probe_layout", "status": "done", "started_at": "t0", "finished_at": "t1"},
             {"stage": "build_document", "status": "running", "started_at": "t1", "finished_at": None}]
    (scratch / "progress.json").write_text(json.dumps(steps), encoding="utf-8")

    app = create_app(warmup=False)
    with TestClient(app) as c:
        r = c.get("/ingest/status/c1")
    assert r.status_code == 200
    body = r.json()
    assert body["contract_id"] == "c1"
    assert body["steps"] == steps


def test_status_empty_when_no_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path))
    app = create_app(warmup=False)
    with TestClient(app) as c:
        r = c.get("/ingest/status/never-seen")
    assert r.status_code == 200
    assert r.json() == {"contract_id": "never-seen", "steps": []}
