import sys
import types
import importlib
from pathlib import Path
import tempfile
import zipfile


def _install_fake_psycopg2():
    fake = types.ModuleType("psycopg2")
    fake.OperationalError = type("OperationalError", (Exception,), {})
    fake.connect = lambda *a, **k: None
    extras = types.ModuleType("psycopg2.extras")
    extras.register_uuid = lambda: None
    extras.execute_values = lambda *a, **k: None
    sys.modules["psycopg2"] = fake
    sys.modules["psycopg2.extras"] = extras


def _install_fake_fitz():
    mod = types.ModuleType("fitz")
    # we won't open PDFs in these tests; just provide placeholder
    mod.open = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unused"))
    sys.modules["fitz"] = mod


def _reload_core():
    _install_fake_psycopg2()
    _install_fake_fitz()
    if "rag_app.core" in sys.modules:
        del sys.modules["rag_app.core"]
    return importlib.import_module("rag_app.core")


def test_aplainar_e_mapear_meta():
    core = _reload_core()
    meta = {
        "compra": {
            "cnpj": "000000000001",
            "objeto": "Cadeiras",
            "ano": 2024,
            "esfera": "federal",
            "poder": "executivo",
            "sequencial": 12,
            "data_publicacao": "2024-01-01",
        },
        "unidadeOrgao": {"municipioNome": "São Paulo", "ufSigla": "SP"},
    }
    out = core._aplainar_e_mapear_meta(meta)
    assert out["cnpj"] == "000000000001"
    assert out["orgao_nome"] is None or "orgao_nome" in out  # can be None-filtered
    assert out["estado_sigla"] == "SP"
    assert out["municipio_nome"] == "São Paulo"
    assert out["ano"] == 2024


def test_iter_docs_from_entry_pdf_and_zip():
    core = _reload_core()
    from rag_app.utils import sha256_file

    with tempfile.TemporaryDirectory() as td:
        # PDF case
        pdf = Path(td) / "doc.pdf"
        pdf.write_bytes(b"fake-pdf-bytes")
        entry = {"file_path": str(pdf), "sha256": "Z"}
        out = core.iter_docs_from_entry(entry)
        assert out and out[0][1].suffix == ".pdf"
        assert out[0][0].startswith("Z::")
        assert out[0][2] is False  # not tmp
        assert out[0][3] == sha256_file(pdf)

        # ZIP case
        zpath = Path(td) / "docs.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("inner/f1.pdf", b"data1")
            zf.writestr("inner/f2.txt", b"not pdf")
            zf.writestr("f3.PDF", b"data3")

        entry2 = {"file_path": str(zpath), "sha256": "H"}
        out2 = core.iter_docs_from_entry(entry2)
        # should include only pdf members, marked as tmp
        assert out2 and all(p[2] is True for p in out2)
        for _, pth, is_tmp, _ in out2:
            if is_tmp:
                try:
                    Path(pth).unlink(missing_ok=True)
                except Exception:
                    pass


def test_chunk_functions_mapping():
    core = _reload_core()
    from rag_app import chunking

    assert core.CHUNK_FUNCTIONS["pages"] is chunking.chunk_pages
    assert core.CHUNK_FUNCTIONS["paras"] is chunking.chunk_paragraphs
    assert core.CHUNK_FUNCTIONS["sent"] is chunking.chunk_sentences
    assert core.CHUNK_FUNCTIONS["chars"] is chunking.chunk_chars
    assert core.CHUNK_FUNCTIONS["rec"] is chunking.chunk_recursive

