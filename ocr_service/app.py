"""OCR 服务 FastAPI 绑定：/health、/ingest（上传 PDF→私有 scratch→返 derived tar）、/reocr（重跑产新 document）。

把进程内 ocr/ 包成 HTTP（HANDOFF §15 服务化口子）。ingest_fn/reocr_fn 可注入便于测试。
/ingest 在私有 scratch 干活，不写 contracts/；产出 derived/<cid>/ 打成 tar 返回。
"""
from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from common import bundle
from . import health as health_mod
from . import ingest as ingest_mod


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
        cid = ingest_mod.sanitize_id(contract_id) if contract_id else ingest_mod.sanitize_id(file.filename)
        if not cid:
            raise HTTPException(status_code=400, detail="无法确定合法 contract_id（请提供 contract_id 或用 ASCII 文件名）")
        scratch = Path(tempfile.mkdtemp(prefix="cr-ingest-"))
        try:
            (scratch / "raw").mkdir(parents=True, exist_ok=True)
            raw_pdf = scratch / "raw" / f"{cid}.pdf"
            raw_pdf.write_bytes(file.file.read())
            res = _ingest(cid, contracts_root=scratch, raw_pdf=raw_pdf)
            if "error" in res:
                raise HTTPException(status_code=500, detail=res.get("error", "入库失败"))
            data = bundle.pack_dir(scratch / "derived", cid)
            return Response(content=data, media_type="application/x-tar")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

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
                raise HTTPException(status_code=400, detail=f"非法包：{e}")
            return _reocr(cid, derived)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    return app
