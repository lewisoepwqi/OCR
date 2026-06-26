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
