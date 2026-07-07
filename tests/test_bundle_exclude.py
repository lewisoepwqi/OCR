from pathlib import Path
from common.bundle import pack_dir, unpack_dir
import tarfile, io


def _members(data: bytes) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        return {m.name for m in tf.getmembers() if m.isfile()}


def test_exclude_skips_named_toplevel(tmp_path):
    base = tmp_path / "c1"
    (base / "pages").mkdir(parents=True)
    (base / "pages" / "p1.png").write_bytes(b"img")
    (base / "layout.json").write_text("{}")
    (base / "document.json").write_text("{}")
    data = pack_dir(tmp_path, "c1", exclude=["pages"])
    names = _members(data)
    assert "c1/layout.json" in names and "c1/document.json" in names
    assert not any(n.startswith("c1/pages/") for n in names)


def test_include_exclude_mutually_exclusive(tmp_path):
    (tmp_path / "c1").mkdir()
    import pytest
    with pytest.raises(ValueError):
        pack_dir(tmp_path, "c1", include=["pages"], exclude=["pages"])
