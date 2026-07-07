from pathlib import Path
from PIL import Image
from ocr import probe_layout


def test_load_pages_reads_existing_pngs(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    Image.new("RGB", (120, 200), "white").save(pages / "p1.png")
    Image.new("RGB", (100, 180), "white").save(pages / "p2.png")
    metas = probe_layout.load_pages(pages)
    assert [m["page"] for m in metas] == [1, 2]
    assert metas[0]["width_px"] == 120 and metas[0]["height_px"] == 200
    assert Path(metas[0]["path"]).name == "p1.png"
