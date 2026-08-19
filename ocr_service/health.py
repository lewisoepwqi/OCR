"""OCR 服务就绪自检：启动时后台跑一次极小 OCR 推理，证明 paddle+GPU+模型+推理端到端通；
结果缓存供 /ready 读取（每次探活只读缓存、不重复推理）。

两层语义：
- /health = 存活（liveness）：进程活即 200，用途「要不要重启容器」。
- /ready  = 就绪（readiness）：读本模块缓存，200/503，用途「能不能接流量」。
docker healthcheck 指向 /ready，start_period 给足首次模型加载时间。
"""
from __future__ import annotations

import glob
import os
import re
import threading
import time

# 严格匹配 /dev/nvidia<数字>：排除 nvidiactl / nvidia-uvm / nvidia-modeset 等控制节点。
_GPU_DEVICE_RE = re.compile(r"^/dev/nvidia\d+$")


class Readiness:
    """就绪状态缓存：后台自检线程写、/ready 读。线程安全；每个 app 一份（不用全局，避免测试串扰）。"""

    def __init__(self) -> None:
        self._state: dict = {"ready": False, "device": None, "gpu_count": None,
                             "ms": None, "error": "warming up", "checked_at": None}
        self._lock = threading.Lock()
        self._recheck_failed = False   # 当前是否处于「因运行期复检失败而降级」状态

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

    def run_recheck(self, probe_fn) -> None:
        """运行期轻量复检（如 GPU 可见性探测，不跑推理）：
        - 失败 → ready=False 并记 error（运行期 GPU 丢失要可观测，不能冒充就绪）；
        - 此前**因复检失败**而降级、现在复检通过 → 恢复 ready=True。
          启动自检（端到端推理）失败不因复检成功翻盘——推理没重验过，不冒充就绪；
        - 无论成败都刷新 checked_at、成功时同步 gpu_count——checked_at 是复检线程
          还活着的心跳，只记录翻转会让它冻结在启动时刻，复检失明无人察觉
          （2026-08-19 二次事故：复检恒读 paddle 缓存旧值，checked_at 停在两天前）。"""
        try:
            info = probe_fn() or {}
            with self._lock:
                if self._recheck_failed:
                    self._recheck_failed = False
                    self._state.update(ready=True, error=None)
                if info.get("gpu_count") is not None:
                    self._state["gpu_count"] = info["gpu_count"]
                self._state["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception as e:  # noqa: BLE001 任何复检失败都归为「未就绪」并记原因
            with self._lock:
                self._recheck_failed = True
                self._state.update(ready=False,
                                   error=f"运行期复检失败 {type(e).__name__}: {e}",
                                   checked_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    def start_recheck(self, probe_fn, interval: float) -> threading.Thread | None:
        """后台 daemon 线程周期复检。interval<=0 时不启动（测试/纯 CPU 部署可关）。"""
        if interval <= 0:
            return None

        def loop() -> None:
            while True:
                time.sleep(interval)
                self.run_recheck(probe_fn)

        t = threading.Thread(target=loop, name="ocr-gpu-recheck", daemon=True)
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

    device = os.environ.get("OCR_DEVICE", "gpu")
    gpu_count = _visible_gpu_count()
    # 期望 GPU 却数到 0 卡 → 自检直接失败（/ready 503、容器 unhealthy）。
    # 2026-08-17 事故教训：cgroup 设备过滤器丢失后 paddle 静默降级 CPU，此处若只记录
    # 不拦截，/ready 会一直 200，OCR 在 CPU 上慢跑 13 天无人察觉。
    # 拦截放在重 import（PIL/paddle 链）之前：权限已丢就不必白费加载。
    if device == "gpu" and (gpu_count or 0) == 0:
        raise RuntimeError(
            f"OCR_DEVICE=gpu 但可见 GPU 数为 {gpu_count}——设备权限可能丢失"
            "（cgroup 设备过滤器/驱动变更），重启容器可恢复；确要纯 CPU 部署请显式设 OCR_DEVICE=cpu")

    from PIL import Image, ImageDraw

    from ocr.build_document import build_ocr

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


def _visible_gpu_count() -> int:
    """数容器内当前真实可打开的 /dev/nvidia[0-9]* 字符设备数。

    为什么不用 paddle.device.cuda.device_count()：paddle C++ 侧的设备数在进程内
    静态缓存——2026-08-19 事故：容器运行中丢 GPU（cgroup 设备 BPF 被冲掉）后，
    服务进程里的 paddle 恒报启动时的旧值 1，60s 周期复检形同虚设、OCR 静默
    CPU 慢跑两天。os.open 每次都发真实系统调用、不过任何进程内缓存：设备
    节点被 cgroup 设备过滤器拒绝时直接 EPERM，探测即时如实。
    """
    n = 0
    for path in sorted(glob.glob("/dev/nvidia*")):
        if not _GPU_DEVICE_RE.match(path):
            continue
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        os.close(fd)
        n += 1
    return n


def gpu_recheck() -> dict:
    """轻量运行期复检：真实探测 GPU 设备节点可打开性（毫秒级，供 /ready 周期刷新）。

    期望 gpu 而数到 0 卡 → 抛异常。动机：2026-08-17 宿主机重启后容器 cgroup 设备
    BPF 丢失，GPU 在容器运行中消失而启动自检早已通过、/ready 永远 200——OCR 静默
    降级 CPU 慢跑多日。2026-08-19 二次复发放大了另一漏洞：paddle 的 device_count
    进程内静态缓存，复检用它等于永远复读启动结论——故改 _visible_gpu_count()
    每次发真实 open 系统调用，不经过任何缓存。
    """
    device = os.environ.get("OCR_DEVICE", "gpu")
    n = _visible_gpu_count()
    if device == "gpu" and n == 0:
        raise RuntimeError(
            "期望 GPU（OCR_DEVICE=gpu）但可打开的 GPU 设备节点数为 0——设备权限可能丢失"
            "（cgroup 设备过滤器/驱动变更），请重启容器恢复")
    return {"gpu_count": n}
