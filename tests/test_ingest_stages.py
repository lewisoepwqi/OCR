"""stages 白名单参数：调用方声明 /ingest 只跑哪几个阶段。

判据覆盖：
- 白名单解析：逗号分隔、容忍空白/重复/空段；规范顺序；未知名 → 400；
- 依赖校验：build_document 需 probe_layout；recognize_tables/seals 需 build_document；缺前置 → 400；
- progress.json 区分 skipped-by-request（本次按请求跳过）与 skipped（续跑产物已存在）；
- 只跑白名单、缺 tables.json/seals.json 时打包不报错；
- 默认路径（不传 stages）行为不变：不透传 stages、全部四阶段照跑；
- 续跑边界：上次带 stages 失败留下的 scratch，这次不带 stages 重试要把缺的阶段补上。
"""
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ocr_service.app import create_app
from ocr_service.parser import StageSpecError, parse_stages, parse, read_progress, EXPECTED_ARTIFACT
from common import bundle

# 故意拼错的阶段名（未知名用例的输入，测「打错字必须 400 而非静默少跑」）
_TYPO_STAGE = "recgonize_tables"


def _make_runner(calls, fail_stage=None):
    """假 runner：记录 stage 调用；成功则造出规范产物；fail_stage 返回非零。"""
    def runner(cmd, capture_output, text, env):
        stage = Path(cmd[1]).stem
        calls.append(stage)
        root = Path(env["CR_CONTRACTS_ROOT"])
        cid = cmd[cmd.index("--id") + 1]
        if stage == fail_stage:
            return SimpleNamespace(returncode=1, stderr=f"{stage} boom", stdout="")
        derived = root / "derived" / cid
        art = derived / EXPECTED_ARTIFACT[stage]
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="ok")
    return runner


def _seed(derived: Path, stages: list[str]) -> None:
    """预置若干阶段的规范产物（模拟上次跑过）。"""
    derived.mkdir(parents=True, exist_ok=True)
    for s in stages:
        (derived / EXPECTED_ARTIFACT[s]).write_text("{}", encoding="utf-8")


# ── parse_stages：白名单字符串解析 ────────────────────────────────

@pytest.mark.parametrize("spec", [None, "", "  ", " , ,"])
def test_blank_stages_means_run_all(spec):
    """不传/空白 stages = 跑满四阶段（与今天行为一致），返回 None 表示「全跑」。"""
    assert parse_stages(spec) is None


def test_valid_subset_dedupe_and_canonical_order():
    """合法子集：去重、去空白，按规范阶段顺序返回（调用方写的顺序无关）。"""
    assert parse_stages("build_document, probe_layout,build_document") == \
        ["probe_layout", "build_document"]
    assert parse_stages("recognize_seals,build_document,probe_layout,recognize_tables") == \
        ["probe_layout", "build_document", "recognize_tables", "recognize_seals"]


def test_unknown_stage_name_rejected():
    """未知阶段名必须报错（不许静默少跑一个阶段），报错里给出合法名单。"""
    with pytest.raises(StageSpecError) as ei:
        parse_stages(f"probe_layout,{_TYPO_STAGE}")   # 打错字
    msg = str(ei.value)
    assert _TYPO_STAGE in msg
    for s in ("probe_layout", "build_document", "recognize_tables", "recognize_seals"):
        assert s in msg   # 合法值全列出，便于调用方自查


@pytest.mark.parametrize("spec,missing", [
    ("build_document", "probe_layout"),
    ("recognize_tables", "build_document"),
    ("recognize_seals", "build_document"),
    ("probe_layout,recognize_tables", "build_document"),   # 部分有前置也不行
])
def test_missing_prerequisite_rejected(spec, missing):
    """缺前置依赖必须报错：不许静默跑出缺 layout/document 的半成品。"""
    with pytest.raises(StageSpecError) as ei:
        parse_stages(spec)
    assert missing in str(ei.value)


# ── parse()：只跑白名单 + progress 状态 ─────────────────────────

def test_parse_runs_only_requested_stages(tmp_path: Path):
    calls = []
    res = parse("c1", contracts_root=tmp_path, runner=_make_runner(calls),
                stages=["probe_layout", "build_document"])
    assert res == {"ok": True}
    assert calls == ["probe_layout", "build_document"]
    prog = {p["stage"]: p["status"] for p in read_progress(tmp_path)}
    assert prog["probe_layout"] == "done" and prog["build_document"] == "done"
    # 未要求的阶段：skipped-by-request（区别于续跑的 skipped）
    assert prog["recognize_tables"] == "skipped-by-request"
    assert prog["recognize_seals"] == "skipped-by-request"


def test_parse_single_stage_ok(tmp_path: Path):
    """只要 probe_layout 也合法（它没有前置）。"""
    calls = []
    res = parse("c1", contracts_root=tmp_path, runner=_make_runner(calls),
                stages=["probe_layout"])
    assert res == {"ok": True}
    assert calls == ["probe_layout"]


def test_skipped_by_request_distinct_from_resume_skip(tmp_path: Path):
    """产物已存在 + 本次未要求 → skipped-by-request（不是续跑的 skipped）。

    这是 /ingest/status 读者区分「这次没跑」与「上次跑过所以续跑跳过」的依据。
    """
    _seed(tmp_path / "derived" / "c1", ["recognize_tables", "recognize_seals"])
    calls = []
    parse("c1", contracts_root=tmp_path, runner=_make_runner(calls),
          stages=["probe_layout", "build_document"])
    prog = {p["stage"]: p["status"] for p in read_progress(tmp_path)}
    assert prog["recognize_tables"] == "skipped-by-request"   # 产物在，但本次没要求
    assert "recognize_tables" not in calls
    # 对照：同样产物齐全 + 不带 stages（全跑）→ 才是续跑语义的 skipped
    prog2 = {p["stage"]: p["status"] for p in read_progress(tmp_path)}
    assert prog2["recognize_tables"] == "skipped-by-request"


def test_requested_stage_still_uses_resume_skip(tmp_path: Path):
    """被要求的阶段若产物已存在 → 仍走既有续跑跳过（status=skipped），不受白名单影响。"""
    _seed(tmp_path / "derived" / "c1", ["probe_layout", "build_document"])
    calls = []
    parse("c1", contracts_root=tmp_path, runner=_make_runner(calls),
          stages=["probe_layout", "build_document"])
    assert calls == []   # 两个都因产物已存在跳过
    prog = {p["stage"]: p["status"] for p in read_progress(tmp_path)}
    assert prog["probe_layout"] == "skipped" and prog["build_document"] == "skipped"


def test_default_path_runs_all_without_stages(tmp_path: Path):
    """不传 stages：四阶段全跑，progress 里没有任何 skipped-by-request（默认路径不变）。"""
    calls = []
    res = parse("c1", contracts_root=tmp_path, runner=_make_runner(calls))
    assert res == {"ok": True}
    assert calls == ["probe_layout", "build_document", "recognize_tables", "recognize_seals"]
    assert all(p["status"] == "done" for p in read_progress(tmp_path))


def test_parse_rejects_invalid_stage_list(tmp_path: Path):
    """parse 直接收到非法列表（绕过 HTTP 层）也要拒绝：未知名/缺前置都不许静默。"""
    with pytest.raises(StageSpecError):
        parse("c1", contracts_root=tmp_path, runner=_make_runner([]), stages=["bogus"])
    with pytest.raises(StageSpecError):
        parse("c1", contracts_root=tmp_path, runner=_make_runner([]), stages=["recognize_seals"])


def test_resume_after_staged_failure_fills_missing_stages(tmp_path: Path):
    """续跑边界：上次 stages=[p,b] 跑挂 → scratch 留下 layout.json；本次不带 stages 重试，
    应把 build_document/recognize_tables/recognize_seals 补上（而不是当成已完成）。"""
    calls = []
    res1 = parse("c1", contracts_root=tmp_path, runner=_make_runner(calls, fail_stage="build_document"),
                 stages=["probe_layout", "build_document"])
    assert res1["stage"] == "build_document"
    assert (tmp_path / "derived" / "c1" / "layout.json").exists()
    calls2 = []
    res2 = parse("c1", contracts_root=tmp_path, runner=_make_runner(calls2))
    assert res2 == {"ok": True}
    assert calls2 == ["build_document", "recognize_tables", "recognize_seals"]   # 缺的补上
    prog = {p["stage"]: p["status"] for p in read_progress(tmp_path)}
    assert prog["probe_layout"] == "skipped"   # 上次完成的产物 → 续跑跳过
    assert prog["recognize_tables"] == "done" and prog["recognize_seals"] == "done"


def test_staged_failure_keeps_scratch_for_retry(tmp_path: Path):
    """带 stages 的失败同样保留 scratch（续跑语义不因白名单改变）。"""
    calls = []
    res = parse("c1", contracts_root=tmp_path, runner=_make_runner(calls, fail_stage="build_document"),
                stages=["probe_layout", "build_document"])
    assert res["stage"] == "build_document"
    assert (tmp_path / "derived" / "c1" / "layout.json").exists()   # scratch 未清


# ── /ingest 端点 ─────────────────────────────────────────────────

def _pages_tar(cid: str) -> bytes:
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / cid / "pages").mkdir(parents=True)
    (d / cid / "pages" / "p1.png").write_bytes(b"\x89PNG\r\n")
    return bundle.pack_dir(d, cid, include=["pages"])


def _post(c, cid="c1", stages=None):
    data = {"contract_id": cid}
    if stages is not None:
        data["stages"] = stages
    return c.post("/ingest", files={"file": (f"{cid}.tar", _pages_tar(cid), "application/x-tar")},
                  data=data)


def test_ep_stages_subset_packs_without_missing_artifacts(tmp_path, monkeypatch):
    """stages=probe_layout,build_document：tables.json/seals.json 不存在是正常的，
    打包/返回不得因此报错（判据 6），tar 里自然没有这两个成员。"""
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path))
    seen = {}

    def fake_ingest(cid, *, contracts_root, stages=None):
        seen["stages"] = stages
        d = Path(contracts_root) / "derived" / cid
        d.mkdir(parents=True, exist_ok=True)
        (d / "layout.json").write_text("{}", encoding="utf-8")
        (d / "document.json").write_text("{}", encoding="utf-8")
        return {"ok": True}

    app = create_app(ingest_fn=fake_ingest, warmup=False)
    with TestClient(app) as c:
        r = _post(c, stages="probe_layout,build_document")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/x-tar"
    assert seen["stages"] == ["probe_layout", "build_document"]
    names = tarfile.open(fileobj=io.BytesIO(r.content)).getnames()
    assert "c1/document.json" in names and "c1/layout.json" in names
    assert "c1/tables.json" not in names and "c1/seals.json" not in names
    assert not any(n.startswith("c1/pages/") for n in names)


def test_ep_unknown_stage_returns_400(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path))
    app = create_app(ingest_fn=lambda *a, **k: {"ok": True}, warmup=False)
    with TestClient(app) as c:
        r = _post(c, stages=f"probe_layout,{_TYPO_STAGE}")   # 打错字
    assert r.status_code == 400
    assert _TYPO_STAGE in r.json()["detail"]
    assert "probe_layout" in r.json()["detail"]   # 合法名单在报错里


def test_ep_missing_prereq_returns_400(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path))
    app = create_app(ingest_fn=lambda *a, **k: {"ok": True}, warmup=False)
    with TestClient(app) as c:
        r = _post(c, stages="build_document")
    assert r.status_code == 400
    assert "probe_layout" in r.json()["detail"]


def test_ep_blank_stages_behaves_like_absent(tmp_path, monkeypatch):
    """stages 空串 = 未传：不把 stages 透传给 ingest_fn（默认路径调用形状不变）。"""
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path))

    def fake_ingest(cid, *, contracts_root):   # 故意不收 stages 参数
        d = Path(contracts_root) / "derived" / cid
        d.mkdir(parents=True, exist_ok=True)
        (d / "document.json").write_text("{}", encoding="utf-8")
        return {"ok": True}

    app = create_app(ingest_fn=fake_ingest, warmup=False)
    with TestClient(app) as c:
        r = _post(c, stages="")
    assert r.status_code == 200


def test_ep_default_path_calls_ingest_without_stages_kwarg(tmp_path, monkeypatch):
    """判据 1（默认路径）：不传 stages 时 ingest_fn 收不到 stages 关键字 —— 既有注入
    签名零改动即绿，证明默认路径调用形状与改动前逐字相同。"""
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path))

    def fake_ingest(cid, *, contracts_root):   # 不收 stages：若被传会 TypeError → 500
        d = Path(contracts_root) / "derived" / cid
        d.mkdir(parents=True, exist_ok=True)
        (d / "document.json").write_text("{}", encoding="utf-8")
        return {"ok": True}

    app = create_app(ingest_fn=fake_ingest, warmup=False)
    with TestClient(app) as c:
        r = _post(c)   # 不带 stages
    assert r.status_code == 200


def test_ep_downstream_unpack_of_partial_tar(tmp_path, monkeypatch):
    """模拟下游（KOS extractFile）：对缺 tables/seals 的返回包解包取 document.json 不报错。"""
    monkeypatch.setenv("OCR_SCRATCH_ROOT", str(tmp_path))

    def fake_ingest(cid, *, contracts_root, stages=None):
        d = Path(contracts_root) / "derived" / cid
        d.mkdir(parents=True, exist_ok=True)
        (d / "document.json").write_text('{"pages": []}', encoding="utf-8")
        return {"ok": True}

    app = create_app(ingest_fn=fake_ingest, warmup=False)
    with TestClient(app) as c:
        r = _post(c, stages="probe_layout,build_document")
    assert r.status_code == 200
    out = tmp_path / "unpack"
    bundle.unpack_dir(r.content, out, "c1")
    assert (out / "c1" / "document.json").exists()
    assert not (out / "c1" / "tables.json").exists()   # 没跑就没有，属正常
