import os
import tempfile
from pathlib import Path

import pytest

from rag_app import utils


def test_sanitize_text_removes_null_and_strips():
    assert utils.sanitize_text("\x00abc \x00 ") == "abc"
    assert utils.sanitize_text("") == ""
    assert utils.sanitize_text(None) == ""


def test_approx_token_len_basic():
    assert utils.approx_token_len("abcd") == 1
    assert utils.approx_token_len("abcd" * 5) >= 5


def test_sha256_file_and_none_for_missing():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "file.txt"
        p.write_text("hello world", encoding="utf-8")
        h = utils.sha256_file(p)
        assert isinstance(h, str) and len(h) == 64
        assert utils.sha256_file(Path(td) / "missing.txt") is None


def test_make_chunk_id_changes_with_source_filename():
    cid1 = utils.make_chunk_id(
        doc_id="doc1",
        strategy="pages",
        text="some content",
        page_start=1,
        page_end=2,
        source_filename="a.pdf",
    )
    cid2 = utils.make_chunk_id(
        doc_id="doc1",
        strategy="pages",
        text="some content",
        page_start=1,
        page_end=2,
        source_filename="b.pdf",
    )
    assert cid1 != cid2


def test_to_vec_formatting():
    v = utils.to_vec([0.1234567, 1.0, 2])
    assert v == "[0.123457,1.000000,2.000000]"


def test_load_config_roundtrip(tmp_path: Path):
    yml = tmp_path / "cfg.yml"
    yml.write_text("key: value\nnum: 3\n", encoding="utf-8")
    cfg = utils.load_config(str(yml))
    assert cfg["key"] == "value"
    assert cfg["num"] == 3
