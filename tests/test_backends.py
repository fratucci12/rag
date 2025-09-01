import os
import types

import importlib


def test_localbackend_raises_without_sentence_transformers(monkeypatch):
    from rag_app import backends

    # Force absence
    monkeypatch.setattr(backends, "SentenceTransformer", None)
    try:
        backends.LocalBackend("any-model")
        assert False, "Deveria ter lançado RuntimeError"
    except RuntimeError:
        pass


def test_openaibackend_batches_and_embeds(monkeypatch):
    from rag_app import backends

    os.environ["OPENAI_API_KEY"] = "dummy"

    class _EmbeddingsClient:
        def __init__(self):
            self.calls = []
        def create(self, model=None, input=None):
            self.calls.append(list(input or []))
            # echo embeddings as simple vectors of len=1 for each input
            data = [types.SimpleNamespace(embedding=[float(i)]) for i, _ in enumerate(input or [])]
            return types.SimpleNamespace(data=data)

    class _DummyOpenAI:
        def __init__(self, *a, **k):
            self.embeddings = _EmbeddingsClient()

    monkeypatch.setattr(backends, "OpenAI", _DummyOpenAI)

    b = backends.OpenAIBackend(model="m", batch_size=3)
    out = b.embed(["a", "b", "c", "d", "e"])  # 5 items => 2 batches (3 + 2)
    assert len(out) == 5
    # first embedding is [0.0], second [1.0], etc. per our dummy
    assert out[0] == [0.0] and out[-1] == [4.0]


def test_openai_batch_processor_basic(monkeypatch, tmp_path):
    from rag_app import backends

    class _Files:
        def create(self, file=None, purpose=None):
            return types.SimpleNamespace(id="file_123")
        def content(self, file_id):
            # simple 2-line jsonl
            text = "{\"custom_id\": \"c1\", \"response\": {\"body\": {\"data\": [{\"embedding\": [0.1]}]}}}\n" \
                   "{\"custom_id\": \"c2\", \"response\": {\"body\": {\"data\": [{\"embedding\": [0.2]}]}}}"
            return types.SimpleNamespace(text=text)

    class _Batches:
        def create(self, input_file_id=None, endpoint=None, completion_window=None, metadata=None):
            return types.SimpleNamespace(id="batch_1")
        def retrieve(self, batch_id):
            return types.SimpleNamespace(
                id=batch_id,
                status="completed",
                request_counts=types.SimpleNamespace(total=2, completed=2, failed=0),
                output_file_id="out_file_1",
                error_file_id=None,
            )

    class _DummyOpenAI:
        def __init__(self, *a, **k):
            self.files = _Files()
            self.batches = _Batches()

    monkeypatch.setattr(backends, "OpenAI", _DummyOpenAI)
    os.environ["OPENAI_API_KEY"] = "dummy"

    proc = backends.OpenAIBatchProcessor()

    # upload_file
    f = tmp_path / "requests.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    fid = proc.upload_file(str(f))
    assert fid == "file_123"

    # create_batch
    bid = proc.create_batch(fid, embedding_model="text-embedding-3-small")
    assert bid == "batch_1"

    # check_status
    st = proc.check_status(bid)
    assert st["status"] == "completed" and st["output_file_id"] == "out_file_1"

    # get_results
    content = proc.get_results(st["output_file_id"])
    assert "c1" in content and "c2" in content
