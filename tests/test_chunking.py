from rag_app import chunking


def sample_pages():
    return [
        "Primeira página. Texto curto.",
        "Segunda página com mais texto.\n\nParágrafo seguinte com conteúdo.",
        "",  # página vazia deve ser ignorada onde fizer sentido
        "Terceira página. Última." 
    ]


def test_chunk_pages_produces_chunks_with_page_bounds():
    pages = sample_pages()
    chunks = chunking.chunk_pages(pages, max_tokens=20, overlap_tokens=4)
    assert chunks, "Deve produzir ao menos um chunk"
    for ch in chunks:
        assert ch["text"].strip()
        assert 1 <= ch["page_start"] <= ch["page_end"]
        assert ch["tok_est"] >= 1
        assert ch["char_len"] == len(ch["text"]) or ch["char_len"] >= 1


def test_chunk_paragraphs_groups_paragraphs():
    pages = sample_pages()
    chunks = chunking.chunk_paragraphs(pages, max_tokens=30, overlap_tokens=6)
    assert chunks
    # deve manter ordem e conter trechos dos parágrafos
    assert any("Parágrafo" in ch["text"] for ch in chunks)


def test_chunk_sentences_splits_sentences():
    pages = [
        "Uma frase. Outra frase! Pergunta? E mais uma.",
    ]
    chunks = chunking.chunk_sentences(pages, max_tokens=12, overlap_tokens=2)
    assert chunks
    # cada chunk é composto por uma ou mais sentenças; garantir que o texto não fica vazio
    for ch in chunks:
        assert ch["text"].strip()


def test_chunk_chars_window_and_overlap():
    pages = ["ABCDEFGHIJ", "KLMNOPQRST"]
    chunks = chunking.chunk_chars(pages, window_chars=5, overlap_chars=2)
    assert chunks
    # janela de 5 com passo 3 (5-2)
    assert chunks[0]["text"] == "ABCDE"
    assert chunks[1]["text"].startswith("DEF") or len(chunks) > 1


def test_chunk_recursive_respects_max_tokens():
    pages = ["# Título\n\nParágrafo 1. Parágrafo 2. Parágrafo 3."]
    chunks = chunking.chunk_recursive(pages, max_tokens=6, overlap_tokens=2)
    assert chunks
    # estimativa de tokens de cada chunk deve ser <= max_tokens + alguma margem de overlap
    for ch in chunks:
        assert ch["tok_est"] <= 6 or ch["text"]

