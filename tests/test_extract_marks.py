from pathlib import Path

import cv2
import numpy as np
from ocr_service import extract_marks


def _fake_render(path, out_dir, dpi):
    return [{"page": 1, "path": "p1.png", "width_px": 100, "height_px": 100}]


def test_default_render_treats_image_as_single_page(tmp_path):
    """回归：上传图片（非 PDF）时，_default_render 用 cv2 当单页，不走 pypdfium2（否则 500）。"""
    img_path = tmp_path / "shot.png"
    cv2.imwrite(str(img_path), np.full((30, 50, 3), 255, np.uint8))
    pages = extract_marks._default_render(str(img_path), tmp_path / "out", dpi=200)
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert pages[0]["width_px"] == 50 and pages[0]["height_px"] == 30
    assert Path(pages[0]["path"]).exists()


def test_extract_image_input_end_to_end(tmp_path):
    """图片输入端到端（真 _default_render + 桩 ocr_fn，避开 paddle）：不再 500，返回单页。"""
    img_path = tmp_path / "sig.png"
    cv2.imwrite(str(img_path), np.full((40, 60, 3), 255, np.uint8))
    res = extract_marks.extract("signature", str(img_path), ocr_fn=lambda p: [])
    assert len(res["pages"]) == 1 and res["pages"][0]["width"] == 60


def test_signature_returns_text_marks():
    """signature 分支：OCR 文字框透传为 marks（供调用方按网格取 col0 姓名）。"""
    def fake_ocr(img_path):
        return [{"text": "李惟夏", "score": 0.99, "cx": 10, "cy": 10, "top": 5,
                 "bbox": [2, 2, 20, 15]}]
    res = extract_marks.extract("signature", "x.pdf",
                                render_pages_fn=_fake_render,
                                page_image_fn=lambda p: np.zeros((100, 100, 3), np.uint8),
                                ocr_fn=fake_ocr)
    assert res["marks"][0]["text"] == "李惟夏"
    assert res["marks"][0]["page"] == 1
    assert res["pages"][0]["width"] == 100


def test_seal_returns_seal_text():
    """seal 分支：layout role=seal 框 + 章文识别。"""
    res = extract_marks.extract("seal", "x.pdf",
        render_pages_fn=_fake_render,
        page_image_fn=lambda p: np.zeros((100, 100, 3), np.uint8),
        layout_fn=lambda pages: [{"page": 1, "regions": [
            {"role": "seal", "bbox_px": [10, 10, 40, 40]}]}],
        seal_fn=lambda crop: "某某专用章")
    m = res["marks"][0]
    assert m["seal_text"] == "某某专用章" and m["bbox"] == [10, 10, 40, 40]
    assert m["text"] is None   # seal 分支不出文字框


def test_pages_include_image_b64():
    """返回的页图带 base64（供调用方解码做抠图）。"""
    res = extract_marks.extract("signature", "x.pdf",
        render_pages_fn=_fake_render,
        page_image_fn=lambda p: np.full((10, 10, 3), 255, np.uint8),
        ocr_fn=lambda p: [])
    assert res["pages"][0]["image_b64"]   # 非空
    assert res["pages"][0]["height"] == 100   # 来自 render_meta（page_image 仅用于 b64 编码）
