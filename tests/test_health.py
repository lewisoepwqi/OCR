"""OCR 服务健康监控：/health 存活、/ready 就绪（启动自检缓存）。

自检逻辑用注入的假函数测，不依赖 paddle/GPU——验证状态机与端点契约，不验真模型。
"""
import time

from fastapi.testclient import TestClient

from ocr_service.app import create_app


def test_health_liveness_always_ok():
    # 存活探针：进程活即 200，与就绪无关（哪怕模型没加载也 ok）
    app = create_app(warmup=False)
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json() == {"ok": True}


def test_ready_503_before_selftest():
    # 自检未跑 → 未就绪 → 503，且 body 如实标注 ready=false（不冒充就绪）
    app = create_app(warmup=False)
    with TestClient(app) as c:
        r = c.get("/ready")
        assert r.status_code == 503
        assert r.json()["ready"] is False


def test_ready_200_after_successful_selftest():
    # 同步驱动就绪自检（注入假自检，不碰 paddle）→ 就绪 200，回显设备与耗时
    app = create_app(warmup=False)
    app.state.readiness.run(lambda: {"device": "gpu", "gpu_count": 1})
    with TestClient(app) as c:
        r = c.get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is True and body["device"] == "gpu" and body["gpu_count"] == 1
        assert isinstance(body["ms"], int)


def test_ready_503_on_selftest_failure():
    # 自检抛异常（如 CUDA OOM）→ 不就绪 + 记下原因，/ready 503（可观测，不静默冒充）
    app = create_app(warmup=False)
    def boom():
        raise RuntimeError("CUDA out of memory")
    app.state.readiness.run(boom)
    with TestClient(app) as c:
        r = c.get("/ready")
        assert r.status_code == 503
        body = r.json()
        assert body["ready"] is False and "CUDA out of memory" in body["error"]


def test_warmup_runs_injected_selftest_on_startup():
    # warmup=True：lifespan 启动时后台跑自检（注入假自检，快、无 paddle），轮询到就绪
    calls = []
    app = create_app(warmup=True, selftest_fn=lambda: (calls.append(1), {"device": "cpu"})[1])
    with TestClient(app) as c:
        for _ in range(100):
            if c.get("/ready").status_code == 200:
                break
            time.sleep(0.02)
        assert c.get("/ready").status_code == 200
        assert calls == [1]              # 自检恰跑一次


def test_recheck_failure_downgrades_ready():
    # 运行期复检失败（GPU 消失）→ ready 翻 False、/ready 503 且 error 可观测
    app = create_app(warmup=False)
    app.state.readiness.run(lambda: {"device": "gpu", "gpu_count": 1})
    app.state.readiness.run_recheck(lambda: (_ for _ in ()).throw(RuntimeError("可见 GPU 数为 0")))
    with TestClient(app) as c:
        r = c.get("/ready")
        assert r.status_code == 503
        assert "可见 GPU 数为 0" in r.json()["error"]


def test_recheck_success_restores_after_recheck_failure():
    # 复检失败降级后、复检又通过 → 恢复就绪（GPU 无需重启容器即回来时不卡死在 503）
    app = create_app(warmup=False)
    rd = app.state.readiness
    rd.run(lambda: {"device": "gpu", "gpu_count": 1})
    rd.run_recheck(lambda: (_ for _ in ()).throw(RuntimeError("GPU 丢失")))
    rd.run_recheck(lambda: {"gpu_count": 1})
    with TestClient(app) as c:
        assert c.get("/ready").status_code == 200


def test_recheck_success_does_not_rescue_failed_selftest():
    # 启动自检失败（推理端到端没过）不因 GPU 复检通过而翻盘——推理没重验过，不冒充就绪
    app = create_app(warmup=False)
    rd = app.state.readiness
    rd.run(lambda: (_ for _ in ()).throw(RuntimeError("模型加载失败")))
    rd.run_recheck(lambda: {"gpu_count": 1})
    with TestClient(app) as c:
        assert c.get("/ready").status_code == 503


def test_default_selftest_rejects_gpu_missing(monkeypatch):
    # 2026-08-17 事故回归：期望 GPU 但数到 0 卡 → 自检必须失败（而非记录后放行）
    monkeypatch.setitem(__import__("os").environ, "OCR_DEVICE", "gpu")
    import ocr_service.health as h
    monkeypatch.setattr(h, "_visible_gpu_count", lambda: 0)
    try:
        h.default_selftest()
        raised = False
    except RuntimeError as e:
        raised = "可见 GPU 数为 0" in str(e)
    assert raised


def test_gpu_recheck_passes_with_cpu_device(monkeypatch):
    # OCR_DEVICE=cpu（纯 CPU 兜底部署）：复探不因 0 卡而失败
    monkeypatch.setitem(__import__("os").environ, "OCR_DEVICE", "cpu")
    import ocr_service.health as h
    monkeypatch.setattr(h, "_visible_gpu_count", lambda: 0)
    assert h.gpu_recheck() == {"gpu_count": 0}


def test_gpu_recheck_ignores_paddle_cached_count(monkeypatch):
    # 2026-08-19 二次事故回归：容器运行中丢 GPU 后，服务进程里的 paddle.device_count
    # 恒返回启动时的缓存旧值 1——复检若信它就永远「通过」。复检必须用真实系统调用
    # 探测（_visible_gpu_count），paddle 缓存说 1 而设备节点打不开时必须报失败。
    monkeypatch.setitem(__import__("os").environ, "OCR_DEVICE", "gpu")
    import sys
    import types
    fake_paddle = types.SimpleNamespace(device=types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 1)))
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    import ocr_service.health as h
    monkeypatch.setattr(h, "_visible_gpu_count", lambda: 0)
    try:
        h.gpu_recheck()
        raised = False
    except RuntimeError as e:
        raised = "GPU 设备节点数为 0" in str(e)
    assert raised


def test_visible_gpu_count_opens_real_device_nodes(monkeypatch, tmp_path):
    # 探测本身：数的是「能 open 的 /dev/nvidia[0-9]*」——节点在但被 cgroup 拒绝
    # （open 抛 EPERM）不计入；非数字后缀的 nvidiactl/nvidia-uvm 不误计。
    import ocr_service.health as h
    def fake_open(path, flags):
        if path.endswith("nvidia1"):
            raise PermissionError(1, "Operation not permitted")  # cgroup BPF 拒绝
        return 123
    monkeypatch.setattr(h.glob, "glob", lambda pat: ["/dev/nvidia0", "/dev/nvidia1", "/dev/nvidiactl"])
    monkeypatch.setattr(h.os, "open", fake_open)
    monkeypatch.setattr(h.os, "close", lambda fd: None)
    assert h._visible_gpu_count() == 1


def test_recheck_refreshes_checked_at_every_success(monkeypatch):
    # 2026-08-19 回归：成功路径也必须刷 checked_at——它是复检线程活着的心跳，
    # 只记翻转会让它冻结在启动时刻（本次事故中停在两天前，看起来「正常」）。
    app = create_app(warmup=False)
    rd = app.state.readiness
    rd.run(lambda: {"device": "gpu", "gpu_count": 1})
    import ocr_service.health as h
    stamps = iter(["2026-08-19T09:00:01", "2026-08-19T09:01:01", "2026-08-19T09:02:01"])
    monkeypatch.setattr(h.time, "strftime", lambda fmt: next(stamps))
    rd.run_recheck(lambda: {"gpu_count": 1})
    assert rd.get()["checked_at"] == "2026-08-19T09:00:01"
    assert rd.get()["gpu_count"] == 1
    rd.run_recheck(lambda: {"gpu_count": 1})
    assert rd.get()["checked_at"] == "2026-08-19T09:01:01"
