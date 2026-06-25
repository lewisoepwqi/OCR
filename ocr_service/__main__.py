"""python -m ocr_service：启动 OCR 服务（默认 :8001）。私有 scratch 干活，不挂 contracts。"""
from __future__ import annotations

import os

import uvicorn

from ocr_service.app import create_app


def main() -> None:
    app = create_app()
    uvicorn.run(app, host=os.environ.get("OCR_HOST", "0.0.0.0"),
                port=int(os.environ.get("OCR_PORT", "8001")))


if __name__ == "__main__":
    main()
