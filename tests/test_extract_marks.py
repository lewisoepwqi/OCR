from pathlib import Path

import cv2
import numpy as np
from ocr_service import extract_marks


def test_signature_returns_text_marks():
    """signature 分支：OCR 文字框透传为 marks（供调用方按各自规则用，如取某列文字）。"""
    def fake_ocr(image_path):
        return [{"text": "李惟夏", "score": 0.99, "cx": 10, "cy": 10, "top": 5,
                 "bbox": [2, 2, 20, 15]}]
    res = extract_marks.extract("signature", "x.png",
                                page_image_fn=lambda p: np.zeros((100, 120, 3), np.uint8),
                                ocr_fn=fake_ocr)
    assert res["marks"][0]["text"] == "李惟夏"
    assert res["marks"][0]["bbox"] == [2, 2, 20, 15]
    assert res["width"] == 120 and res["height"] == 100
    assert "pages" not in res           # 不回传图片


def test_seal_returns_seal_text():
    """seal 分支：layout role=seal 框 + 章文识别。layout_fn 收 (image_path, w, h)。"""
    res = extract_marks.extract("seal", "x.png",
        page_image_fn=lambda p: np.zeros((100, 100, 3), np.uint8),
        layout_fn=lambda path, w, h: [{"bbox_px": [10, 10, 40, 40]}],
        seal_fn=lambda crop: "某某专用章")
    m = res["marks"][0]
    assert m["seal_text"] == "某某专用章" and m["bbox"] == [10, 10, 40, 40]
    assert m["text"] is None            # seal 分支不出文字框


def test_rejects_unreadable_input():
    """OCR 只接受图片：读不出（如误传 PDF）时明确报错，不静默。"""
    import pytest
    with pytest.raises(ValueError):
        extract_marks.extract("signature", "x.png",
                              page_image_fn=lambda p: None, ocr_fn=lambda p: [])


def test_extract_real_image_no_render(tmp_path):
    """真实图片走默认 cv2.imread（不经任何 PDF 渲染）：宽高正确、不报错。"""
    img_path = tmp_path / "shot.png"
    cv2.imwrite(str(img_path), np.full((40, 60, 3), 255, np.uint8))
    res = extract_marks.extract("signature", str(img_path), ocr_fn=lambda p: [])
    assert res["width"] == 60 and res["height"] == 40 and res["marks"] == []
