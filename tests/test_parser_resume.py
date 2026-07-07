"""parser 续跑：已完成步（规范产物存在且可解析）跳过；progress.json 反映执行轨迹。"""
import json
from pathlib import Path

from ocr_service.parser import parse, EXPECTED_ARTIFACT, read_progress


def _make_runner(calls, fail_stage=None):
    """假 runner：记录每次 stage 调用；成功则造出该 stage 规范产物；fail_stage 返回非零。"""
    def runner(cmd, capture_output, text, env):
        # cmd 形如 [python, .../ocr/<stage>.py, --id, cid, ...]
        stage = Path(cmd[1]).stem
        calls.append(stage)
        root = Path(env["CR_CONTRACTS_ROOT"])
        cid = cmd[cmd.index("--id") + 1]
        if stage == fail_stage:
            class R: returncode = 1; stderr = f"{stage} boom"; stdout = ""
            return R()
        derived = root / "derived" / cid
        derived.mkdir(parents=True, exist_ok=True)
        art = derived / EXPECTED_ARTIFACT[stage]
        art.parent.mkdir(parents=True, exist_ok=True)
        art.write_text("{}", encoding="utf-8")   # 可 JSON 解析的规范产物
        class R: returncode = 0; stderr = ""; stdout = "ok"
        return R()
    return runner


def test_all_stages_run_when_fresh(tmp_path: Path):
    calls = []
    res = parse("c1", contracts_root=tmp_path, runner=_make_runner(calls))
    assert res == {"ok": True}
    assert calls == ["probe_layout", "build_document", "recognize_tables", "recognize_seals"]
    prog = read_progress(tmp_path)
    assert [p["stage"] for p in prog] == calls
    assert all(p["status"] == "done" for p in prog)


def test_completed_stages_skipped_on_resume(tmp_path: Path):
    # 预置前两步的规范产物（模拟上次已完成）
    derived = tmp_path / "derived" / "c1"
    derived.mkdir(parents=True)
    (derived / "layout.json").write_text("{}", encoding="utf-8")
    (derived / "document.json").write_text("{}", encoding="utf-8")
    calls = []
    res = parse("c1", contracts_root=tmp_path, runner=_make_runner(calls))
    assert res == {"ok": True}
    assert calls == ["recognize_tables", "recognize_seals"]   # 前两步跳过
    prog = {p["stage"]: p["status"] for p in read_progress(tmp_path)}
    assert prog["probe_layout"] == "skipped" and prog["build_document"] == "skipped"
    assert prog["recognize_tables"] == "done"


def test_half_written_artifact_not_trusted(tmp_path: Path):
    # 半写（非法 JSON）的产物不算已完成 → 该步仍重跑
    derived = tmp_path / "derived" / "c1"
    derived.mkdir(parents=True)
    (derived / "layout.json").write_text("{ broken", encoding="utf-8")
    calls = []
    parse("c1", contracts_root=tmp_path, runner=_make_runner(calls))
    assert "probe_layout" in calls


def test_failure_returns_stage_and_writes_progress(tmp_path: Path):
    calls = []
    res = parse("c1", contracts_root=tmp_path,
                runner=_make_runner(calls, fail_stage="recognize_tables"))
    assert res["stage"] == "recognize_tables" and "boom" in res["log"]
    prog = {p["stage"]: p["status"] for p in read_progress(tmp_path)}
    assert prog["recognize_tables"] == "error"
    assert prog["probe_layout"] == "done" and prog["build_document"] == "done"
    assert "recognize_seals" not in prog   # 失败后不再往下
