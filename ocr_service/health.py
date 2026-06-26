"""OCR 服务就绪自检：启动时后台跑一次极小 OCR 推理，证明 paddle+GPU+模型+推理端到端通；
结果缓存供 /ready 读取（每次探活只读缓存、不重复推理）。

两层语义：
- /health = 存活（liveness）：进程活即 200，用途「要不要重启容器」。
- /ready  = 就绪（readiness）：读本模块缓存，200/503，用途「能不能接流量」。
docker healthcheck 指向 /ready，start_period 给足首次模型加载时间。
"""
from __future__ import annotations

import os
import threading
import time


class Readiness:
    """就绪状态缓存：后台自检线程写、/ready 读。线程安全；每个 app 一份（不用全局，避免测试串扰）。"""

    def __init__(self) -> None:
        self._state: dict = {"ready": False, "device": None, "gpu_count": None,
                             "ms": None, "error": "warming up", "checked_at": None}
        self._lock = threading.Lock()

    def get(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _set(self, **kw) -> None:
        with self._lock:
            self._state.update(kw)

    def run(self, selftest_fn) -> None:
        """同步跑一次自检并写状态。selftest_fn() 返回 {device, gpu_count?}；抛异常→记 error、不就绪、不上抛
        （健康端点据此报 503，自检失败要可观测、绝不静默冒充就绪）。"""
        start = time.monotonic()
        try:
            info = selftest_fn() or {}
            self._set(ready=True, error=None,
                      device=info.get("device"), gpu_count=info.get("gpu_count"),
                      ms=int((time.monotonic() - start) * 1000),
                      checked_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        except Exception as e:  # noqa: BLE001 任何自检失败都归为「未就绪」并记原因
            self._set(ready=False,
                      error=f"{type(e).__name__}: {e}",
                      ms=int((time.monotonic() - start) * 1000),
                      checked_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    def start(self, selftest_fn) -> threading.Thread:
        """后台线程跑自检：不阻塞 HTTP 启动（自检期间 /health 已 200、/ready 返 503），跑完转就绪。"""
        t = threading.Thread(target=self.run, args=(selftest_fn,),
                             name="ocr-selftest", daemon=True)
        t.start()
        return t


def default_selftest() -> dict:
    """真实自检：构造 PP-OCRv6 引擎，对一张极小合成图跑一次推理——证明 paddle+GPU+模型+推理端到端通。

    仅 OCR 容器（装了 paddle）会跑；返回 {device, gpu_count}。失败抛异常（由 Readiness.run 记 error）。
    用合成图（白底黑框）驱动 det/rec 前向，不依赖字体/文字内容；跑完即释放（不常驻显存，沿用每请求构造的设计）。
    """
    import shutil
    import tempfile
    from pathlib import Path

    from PIL import Image, ImageDraw

    from ocr.build_document import build_ocr

    device = os.environ.get("OCR_DEVICE", "gpu")
    gpu_count = None
    try:
        import paddle
        gpu_count = paddle.device.cuda.device_count()
    except Exception:                       # noqa: BLE001 取不到 GPU 数不影响自检主体
        pass

    tmp = Path(tempfile.mkdtemp(prefix="cr-ocr-selftest-"))
    try:
        img = Image.new("RGB", (320, 96), "white")
        ImageDraw.Draw(img).rectangle([20, 30, 300, 60], outline="black", width=3)
        probe = tmp / "probe.png"
        img.save(probe)
        ocr = build_ocr()
        list(ocr.predict(str(probe)))       # 一次前向：加载模型 + GPU forward；无文字也证明链路通
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"device": device, "gpu_count": gpu_count}
