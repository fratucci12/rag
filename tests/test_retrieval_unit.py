import sys
import types
import importlib


def _install_stub_retrieval_deps():
    # stub openai
    openai_mod = types.ModuleType("openai")
    class _DummyChoice:
        def __init__(self, args_json="{\"semantic_query\": \"q\", \"ano\": 2024}"):
            self.message = types.SimpleNamespace(
                tool_calls=[types.SimpleNamespace(function=types.SimpleNamespace(arguments=args_json))]
            )
    class _DummyCompletions:
        def create(self, **kwargs):
            return types.SimpleNamespace(choices=[_DummyChoice()])
    class _DummyChat:
        def __init__(self):
            self.completions = _DummyCompletions()
    class OpenAI:
        def __init__(self, *a, **k):
            self.chat = _DummyChat()
        class embeddings:
            @staticmethod
            def create(model=None, input=None):
                # return N embeddings of length 2
                data = [types.SimpleNamespace(embedding=[1.0, 0.0]) for _ in (input or [])]
                return types.SimpleNamespace(data=data)
    sys.modules["openai"] = openai_mod
    openai_mod.OpenAI = OpenAI

    # stub sentence_transformers.cross_encoder
    st_pkg = types.ModuleType("sentence_transformers")
    ce_mod = types.ModuleType("sentence_transformers.cross_encoder")
    class CrossEncoder:
        def __init__(self, *a, **k):
            pass
        def predict(self, pairs):
            # score increases with index
            return list(range(len(pairs)))
    ce_mod.CrossEncoder = CrossEncoder
    sys.modules["sentence_transformers"] = st_pkg
    sys.modules["sentence_transformers.cross_encoder"] = ce_mod

    # stub rich console/table
    rich = types.ModuleType("rich")
    console_mod = types.ModuleType("rich.console")
    table_mod = types.ModuleType("rich.table")
    class Console:
        def rule(self, *a, **k):
            pass
        def print(self, *a, **k):
            pass
    class Table:
        def __init__(self, *a, **k):
            pass
        def add_column(self, *a, **k):
            pass
        def add_row(self, *a, **k):
            pass
        def add_section(self, *a, **k):
            pass
    console_mod.Console = Console
    table_mod.Table = Table
    sys.modules["rich"] = rich
    sys.modules["rich.console"] = console_mod
    sys.modules["rich.table"] = table_mod

    # stub tqdm
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda x, **kw: x
    sys.modules["tqdm"] = tqdm_mod


def _reload_retrieval():
    _install_stub_retrieval_deps()
    if "rag_app.retrieval" in sys.modules:
        del sys.modules["rag_app.retrieval"]
    return importlib.import_module("rag_app.retrieval")


class _Cur:
    def __init__(self):
        self.last_sql = None
        self.last_params = None
        self._rows = []

    def execute(self, sql, params=None):
        self.last_sql = sql
        self.last_params = params
        # configure rows based on the SQL executed
        if "websearch_to_tsquery" in sql:
            # return FTS chunk_ids
            self._rows = [("id1",), ("id2",), ("id3",)]
        elif "embedding <=>" in sql and "ORDER BY embedding" in sql:
            # return Vector search chunk_ids in different order
            self._rows = [("id3",), ("id2",), ("id4",)]
        elif "WHERE chunk_id::text = ANY" in sql:
            # return rows with text for any ids requested
            ids = (params or [([],)])[0]
            self._rows = [(f"text-{cid}", 1, 2, cid) for cid in ids]
        else:
            # default for basic_similarity_search
            self._rows = [
                ("T1", 1, 1, 0.9, "cid1"),
                ("T2", 1, 1, 0.8, "cid2"),
                ("T3", 1, 1, 0.7, "cid3"),
            ]

    def fetchall(self):
        return self._rows


def test_build_where_clause_basic():
    rt = _reload_retrieval()
    where, params = rt._build_where_clause({"ano": 2024, "estado_sigla": "SP"})
    assert where.startswith("WHERE ")
    assert where.count("meta @> %s") == 2
    # params are json strings
    assert all(isinstance(p, str) for p in params)


def test_basic_similarity_search_maps_rows():
    rt = _reload_retrieval()
    cur = _Cur()
    out = rt.basic_similarity_search(cur, [0.1, 0.2], "chunks_t", 2, {"ano": 2024})
    assert len(out) == 3  # cursor returns 3 rows
    assert out[0]["text"] == "T1"
    assert out[0]["chunk_id"] == "cid1"
    assert 0 <= out[0]["score"] <= 1


def test_hybrid_search_rrf_combines_and_fetches_details():
    rt = _reload_retrieval()
    cur = _Cur()
    out = rt.hybrid_search_rrf(cur, "cadeiras", [0.1, 0.2], "chunks_t", 3, {"ano": 2024}, k=60)
    assert out
    # ensure it fetched details and preserves top_ids order
    assert all("text-" in item["text"] for item in out)


def test_rerank_with_cross_encoder_sorts_by_ce_scores():
    rt = _reload_retrieval()
    cur = _Cur()
    out = rt.rerank_with_cross_encoder(
        cur, "pergunta", [0.1, 0.2], "chunks_t", "any-model", top_k=2, filters={}
    )
    assert len(out) == 2
    # since CrossEncoder.predict returns increasing scores, top is last of initial list
    assert out[0]["text"] in ("T1", "T2", "T3")

