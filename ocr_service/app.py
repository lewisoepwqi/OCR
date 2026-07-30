"""OCR 服务 FastAPI 绑定：/health、/ingest（收页图 tar→私有 scratch→返 derived tar，排除 pages）、/reocr（重跑产新 document）。

把进程内 ocr/ 包成 HTTP（HANDOFF §15 服务化口子）。ingest_fn/reocr_fn 可注入便于测试。
/ingest 在私有 scratch 干活，不写 contracts/；产出 derived/<cid>/ 打成 tar 返回。
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from common import bundle
from . import health as health_mod
from . import ingest as ingest_mod
from .parser import read_progress


def scratch_root() -> Path:
    """OCR 持久化 scratch 根：环境变量 OCR_SCRATCH_ROOT，缺省系统临时目录下 cr-ocr-scratch。"""
    return Path(os.environ.get("OCR_SCRATCH_ROOT", str(Path(tempfile.gettempdir()) / "cr-ocr-scratch")))


def prune_scratch(root: Path, ttl_hours: float) -> list[str]:
    """清掉 root 下 mtime 超过 ttl_hours 的 per-cid scratch 目录，返回被清 cid 名列表。

    以目录 mtime 判龄：stage 写产物/progress.json 会更新 mtime。
    注：按目录创建/首写时刻判龄，非最后访问——续跑覆盖写文件内容不会刷新目录本身的 mtime。
    """
    root = Path(root)
    if not root.is_dir():
        return []
    cutoff = time.time() - ttl_hours * 3600
    removed: list[str] = []
    for child in root.iterdir():
        if child.is_dir() and child.stat().st_mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child.name)
    return removed


def create_app(*, ingest_fn=None, reocr_fn=None, selftest_fn=None, warmup=True) -> FastAPI:
    _ingest = ingest_fn or ingest_mod.ingest

    def _reocr_default(cid: str, derived_root) -> dict:
        from ocr.build_document import build_document   # 延迟导入，仅 OCR 容器有 paddle
        return build_document(cid, derived_root=derived_root)

    _reocr = reocr_fn or _reocr_default
    _selftest = selftest_fn or health_mod.default_selftest
    readiness = health_mod.Readiness()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动即后台跑一次就绪自检（不阻塞监听；自检期间 /ready 返 503，跑完转就绪）
        if warmup:
            readiness.start(_selftest)
        yield

    app = FastAPI(title="合同雷达 OCR 服务", version="0.1", lifespan=lifespan)
    app.state.readiness = readiness

    @app.get("/health")
    def health() -> dict:
        # 存活探针：进程活即 200（与模型/GPU 是否就绪无关）。
        return {"ok": True}

    @app.get("/ready")
    def ready():
        # 就绪探针：读启动自检缓存——模型加载 + GPU + 推理端到端通过才 200，否则 503（如实回报，不冒充就绪）。
        st = readiness.get()
        return JSONResponse(st, status_code=200 if st["ready"] else 503)

    @app.post("/ingest")
    def ingest_ep(file: UploadFile = File(...), contract_id: str | None = Form(None)) -> Response:
        try:
            ttl_hours = float(os.environ.get("OCR_SCRATCH_TTL_HOURS", "48"))
        except ValueError:
            ttl_hours = 48.0   # 环境变量误配为非数字 → 退回默认，不阻断入库
        try:
            prune_scratch(scratch_root(), ttl_hours)
        except OSError:
            pass   # 清理是尽力而为，失败不影响本次入库
        cid = ingest_mod.sanitize_id(contract_id) if contract_id else ingest_mod.sanitize_id(file.filename)
        if not cid:
            raise HTTPException(status_code=400, detail="无法确定合法 contract_id（请提供 contract_id 或用 ASCII 文件名）")
        # 按 cid 派生固定 scratch（续跑复用）；成功打包后才清，失败保留在磁盘
        scratch = scratch_root() / cid
        derived = scratch / "derived"
        derived.mkdir(parents=True, exist_ok=True)
        # 收页图 tar（调用方业务层渲染），解包进 scratch/derived/<cid>/pages/（防穿越）
        try:
            bundle.unpack_dir(file.file.read(), derived, cid)
        except bundle.BundleError as e:
            raise HTTPException(status_code=400, detail=f"非法页图包：{e}（tar 顶层目录名须与 contract_id 一致，结构见 README）")
        res = _ingest(cid, contracts_root=scratch)
        if "error" in res:
            # 结构化失败：透传 stage/log（此前被丢弃）；保留 scratch 供续跑
            return JSONResponse(status_code=500, content={
                "error": res.get("error", "入库失败"),
                "stage": res.get("stage"),
                "log": res.get("log", ""),
            })
        data = bundle.pack_dir(scratch / "derived", cid, exclude=["pages"])
        shutil.rmtree(scratch, ignore_errors=True)   # 全成功打包完才清
        return Response(content=data, media_type="application/x-tar")

    @app.post("/reocr")
    def reocr_ep(file: UploadFile = File(...), contract_id: str = Form(...)) -> dict:
        cid = ingest_mod.sanitize_id(contract_id)
        if not cid or cid != contract_id:
            raise HTTPException(status_code=400, detail="contract_id 非法")
        scratch = Path(tempfile.mkdtemp(prefix="cr-reocr-"))
        try:
            derived = scratch / "derived"
            derived.mkdir(parents=True, exist_ok=True)
            try:
                bundle.unpack_dir(file.file.read(), derived, cid)
            except bundle.BundleError as e:
                raise HTTPException(status_code=400, detail=f"非法包：{e}（tar 顶层目录名须与 contract_id 一致，结构见 README）")
            return _reocr(cid, derived)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    @app.get("/ingest/status/{contract_id}")
    def ingest_status_ep(contract_id: str) -> dict:
        cid = ingest_mod.sanitize_id(contract_id)   # 归一，防只读目录穿越（与 /ingest 一致）
        steps = read_progress(scratch_root() / cid) if cid else []
        return {"contract_id": contract_id, "steps": steps}

    @app.post("/extract-marks")
    def extract_marks_ep(file: UploadFile = File(...), kind: str = Form(...)):
        """通用提取标记：输入一张页图 → (印章框+章文 | 全文字框)。不含任何应用概念。

        ⚠ 只接受图片（PNG/JPG 等）；PDF 须先在调用方转成图片再逐页上传（本接口不做 PDF 渲染，
        传 PDF 会以「无法读取图片（OCR 只接受图片，PDF→图片请在调用方完成）」报错）。
        失败返 500 + 结构化 {error, stage, log}。
        """
        if kind not in ("signature", "seal"):
            raise HTTPException(status_code=400, detail="kind 必须是 signature 或 seal")
        with tempfile.NamedTemporaryFile(
                suffix=Path(file.filename or "x.pdf").suffix, delete=False) as f:
            f.write(file.file.read())
            src = f.name
        from .extract_marks import extract
        try:
            return extract(kind, src)
        except Exception as exc:
            import traceback
            return JSONResponse(
                {"error": f"提取失败：{exc}", "stage": "extract_marks",
                 "log": traceback.format_exc()[-2000:]},
                status_code=500)

    return app
