"""GET /ingest/status/{cid}：读 scratch 内 progress.json。"""
import json

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


def test_status_rejects_path_traversal(tmp_path, monkeypatch):
    """contract_id=".."（穿越到 scratch_root 父目录）不得读到父目录的 progress.json。

    scratch_root() 本身就是 tmp_path，其父目录放一份"泄露标记"内容；未 sanitize 时
    scratch_root() / ".." 会解析到 tmp_path 的父目录，读到该标记。sanitize_id("..") 返回
    None（清空），端点应直接回空 steps，不去拼路径读取。
    路径参数用 %2E%2E（而非字面 ".."）传递——字面 ".." 在到达 ASGI 路由前就会被 HTTP 客户端
    做 URL 归一化折叠掉（导致 404/405，根本进不了这个端点），percent-encode 后才能让
    ".." 原样作为单段路径参数值到达 handler，复现"客户端可控 contract_id 含穿越意图"的场景。
    """
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path))
    leaked = tmp_path.parent / "progress.json"
    leaked.write_text(json.dumps([{"stage": "LEAKED"}]), encoding="utf-8")
    try:
        app = create_app(warmup=False)
        with TestClient(app) as c:
            r = c.get("/ingest/status/%2E%2E")
        assert r.status_code == 200
        assert r.json()["steps"] == []
    finally:
        leaked.unlink(missing_ok=True)
