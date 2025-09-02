import sys
import types
import importlib


class _FakeCursor:
    def __init__(self):
        self.executed = []
        self.params = []
        self._rows = []
        self.connection = types.SimpleNamespace(commit=lambda: None, rollback=lambda: None)

    def execute(self, sql, params=None):
        self.executed.append(sql)
        if params is not None:
            self.params.append(params)

    def fetchall(self):
        return self._rows

    # context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self):
        self._cur = _FakeCursor()
        self.commits = 0

    def cursor(self):
        return self._cur

    def commit(self):
        self.commits += 1

    def close(self):
        pass


def _install_fake_psycopg2():
    fake = types.ModuleType("psycopg2")
    fake.OperationalError = type("OperationalError", (Exception,), {})
    fake.connect = lambda *a, **k: _FakeConn()

    extras = types.ModuleType("psycopg2.extras")
    def register_uuid():
        return None
    def execute_values(cur, sql, rows, page_size=200):
        # naive execute: record call on cursor
        cur.executed.append(sql)
        cur.params.append((list(rows), page_size))
    extras.register_uuid = register_uuid
    extras.execute_values = execute_values

    sys.modules["psycopg2"] = fake
    sys.modules["psycopg2.extras"] = extras
    return fake


def _reload_db():
    # ensure fake psycopg2 is present before import
    _install_fake_psycopg2()
    if "rag_app.db" in sys.modules:
        del sys.modules["rag_app.db"]
    return importlib.import_module("rag_app.db")


def test_ensure_schema_executes_all_ddls():
    db = _reload_db()
    conn = _FakeConn()
    db.ensure_schema(conn, docs_table="docs", chunk_table="chunks_t", dim=8, use_hnsw=True)
    # vector extension attempt + docs, indexes, chunks, meta index, hnsw index
    executed = "\n".join(conn._cur.executed)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in executed
    assert "CREATE TABLE IF NOT EXISTS docs" in executed
    assert "CREATE INDEX IF NOT EXISTS docs_meta_gin" in executed
    assert "CREATE TABLE IF NOT EXISTS chunks_t" in executed
    assert "CREATE INDEX IF NOT EXISTS chunks_t_meta_gin" in executed
    assert "CREATE INDEX IF NOT EXISTS chunks_t_hnsw" in executed


def test_upsert_document_uses_expected_params():
    db = _reload_db()
    cur = _FakeCursor()
    entry = {"sha256": "s1", "file_path": "/p.pdf", "meta": {"a": 1}}
    db.upsert_document(cur, "docs", "doc-1", entry, sha256_doc="d1")
    assert cur.executed, "deve executar INSERT ... ON CONFLICT"
    params = cur.params[-1]
    assert params[0] == "doc-1"
    # Agora sha256 usa sha256_doc quando disponvel
    assert params[1] == "d1"
    assert params[2] == "d1"


def test_filter_new_chunks_excludes_already_there():
    db = _reload_db()
    cur = _FakeCursor()
    # prepare two chunks A,B; mark A as already existing
    doc_id = "d"
    strategy = "pages"
    chs = [
        {"text": "A", "page_start": 1, "page_end": 1},
        {"text": "B", "page_start": 1, "page_end": 1},
    ]
    # compute existing cid for A using utils.make_chunk_id
    from rag_app.utils import make_chunk_id

    cid_A = make_chunk_id(doc_id, strategy, "A", 1, 1, "file.pdf")
    cur._rows = [(cid_A,)]

    out = db.filter_new_chunks(cur, "chunks_pages", doc_id, strategy, chs, "file.pdf")
    cids = {cid for cid, _ in out}
    assert cid_A not in cids
    assert len(out) == 1


def test_bulk_insert_chunks_uses_execute_values(monkeypatch):
    db = _reload_db()
    cur = _FakeCursor()
    captured = {"calls": 0}

    def fake_execute_values(cur2, sql, rows, page_size=200):
        captured["calls"] += 1
        # record minimal info
        assert "INSERT INTO" in sql
        assert rows
    monkeypatch.setattr(db, "execute_values", fake_execute_values)

    rows = [("cid", "strat", "doc", 1, 1, 10, 50, "text", "{}", "[0.1,0.2]")]
    db.bulk_insert_chunks(cur, "chunks_t", rows, init_batch=2, inner_page=2)
    assert captured["calls"] >= 1
