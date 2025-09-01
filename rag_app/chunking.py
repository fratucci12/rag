# chunking.py - Estratégias para dividir texto em pedaços (chunks)
# (Este arquivo permanece o mesmo do seu projeto original)

import re
from typing import List, Dict, Any
from .utils import approx_token_len, count_tokens


# --- Funções Auxiliares de Mapeamento de Página ---
def _build_page_index(pages: List[str]):
    spans = []
    pos = 0
    for i, p in enumerate(pages, start=1):
        s = p or ""
        spans.append((pos, pos + len(s), i))
        pos += len(s) + 1
    return spans


def _range_to_pages(spans, start_c, end_c):
    p_start = p_end = None
    for s, e, pg in spans:
        if e <= start_c:
            continue
        if s >= end_c:
            break
        p_start = pg if p_start is None else p_start
        p_end = pg
    return (p_start or 1, p_end or (spans[-1][2] if spans else 1))


# --- Estratégias de Chunking ---
def chunk_pages(
    pages: List[str], max_tokens: int, overlap_tokens: int, *, use_tiktoken: bool = False, model: str | None = None
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    cur_text: List[str] = []
    cur_pages: List[int] = []
    cur_tokens = 0

    def tok_len(s: str) -> int:
        return count_tokens(s, model=model) if use_tiktoken else approx_token_len(s)

    def flush():
        nonlocal cur_text, cur_pages, cur_tokens
        if not cur_text:
            return
        txt = "\n".join(cur_text).strip()
        if not txt:
            cur_text, cur_pages, cur_tokens = [], [], 0
            return
        t_tok = tok_len(txt)
        chunks.append(
            {
                "text": txt,
                "page_start": cur_pages[0],
                "page_end": cur_pages[-1],
                "char_len": len(txt),
                "tok_est": t_tok,
            }
        )
        cur_text, cur_pages, cur_tokens = [], [], 0

    for i, p in enumerate(pages, start=1):
        t = (p or "").strip()
        if not t:
            continue
        t_tok = tok_len(t)

        if cur_tokens + t_tok > max_tokens and cur_text:
            flush()

        if t_tok > max_tokens:
            paras = re.split(r"\n\s*\n", t)
            cur, cur_tok = "", 0
            for para in paras:
                pt = tok_len(para)
                if (cur_tok + pt > max_tokens) and cur:
                    chunks.append(
                        {
                            "text": cur.strip(),
                            "page_start": i,
                            "page_end": i,
                            "char_len": len(cur),
                            "tok_est": cur_tok,
                        }
                    )
                    if overlap_tokens > 0 and len(cur) > 0:
                        tail_chars = max(0, len(cur) - overlap_tokens * 4)
                        cur = cur[tail_chars:]
                        cur_tok = tok_len(cur)
                cur += ("\n\n" if cur else "") + para
                cur_tok += pt
            if cur.strip():
                chunks.append(
                    {
                        "text": cur.strip(),
                        "page_start": i,
                        "page_end": i,
                        "char_len": len(cur),
                        "tok_est": cur_tok,
                    }
                )
        else:
            cur_text.append(t)
            cur_pages.append(i)
            cur_tokens += t_tok

    flush()

    if overlap_tokens > 0 and chunks:
        oc = overlap_tokens * 4
        for j in range(1, len(chunks)):
            tail = chunks[j - 1]["text"][-oc:]
            chunks[j]["text"] = (tail + "\n" + chunks[j]["text"]).strip()

    return chunks


def chunk_paragraphs(
    pages: List[str], max_tokens: int, overlap_tokens: int, *, use_tiktoken: bool = False, model: str | None = None
) -> List[Dict[str, Any]]:
    doc_text = "\n\n".join((p or "") for p in pages)
    page_spans = _build_page_index(pages)
    paras = re.split(r"\n\s*\n", doc_text)
    chunks, cur_chunk_text, cur_tok = [], "", 0
    cur_pos = 0

    def tok_len(s: str) -> int:
        return count_tokens(s, model=model) if use_tiktoken else approx_token_len(s)

    for para in paras:
        para_with_sep = para + "\n\n"
        pt = tok_len(para_with_sep)
        if cur_tok + pt > max_tokens and cur_chunk_text:
            start_char = doc_text.find(cur_chunk_text, cur_pos)
            end_char = start_char + len(cur_chunk_text)
            p_start, p_end = _range_to_pages(page_spans, start_char, end_char)
            chunks.append(
                {
                    "text": cur_chunk_text.strip(),
                    "page_start": p_start,
                    "page_end": p_end,
                    "char_len": len(cur_chunk_text),
                    "tok_est": cur_tok,
                }
            )
            if overlap_tokens > 0 and len(cur_chunk_text) > 0:
                tail_chars = max(0, len(cur_chunk_text) - overlap_tokens * 4)
                cur_pos = start_char + tail_chars
                cur_chunk_text = cur_chunk_text[tail_chars:]
                cur_tok = tok_len(cur_chunk_text)
            else:
                cur_chunk_text, cur_tok = "", 0
                cur_pos = end_char
        cur_chunk_text += para_with_sep
        cur_tok += pt

    if cur_chunk_text.strip():
        start_char = doc_text.find(cur_chunk_text, cur_pos)
        end_char = start_char + len(cur_chunk_text)
        p_start, p_end = _range_to_pages(page_spans, start_char, end_char)
        chunks.append(
            {
                "text": cur_chunk_text.strip(),
                "page_start": p_start,
                "page_end": p_end,
                "char_len": len(cur_chunk_text),
                "tok_est": cur_tok,
            }
        )
    return chunks


def chunk_sentences(
    pages: List[str], max_tokens: int, overlap_tokens: int, *, use_tiktoken: bool = False, model: str | None = None
) -> List[Dict[str, Any]]:
    doc_text = " ".join((p or "") for p in pages)
    page_spans = _build_page_index(pages)
    sents = re.split(r"(?<=[\.\!\?])\s+", doc_text)
    chunks, cur_chunk_text, cur_tok = [], "", 0
    cur_pos = 0

    def tok_len(s: str) -> int:
        return count_tokens(s, model=model) if use_tiktoken else approx_token_len(s)

    for s in sents:
        sent_with_sep = s + " "
        st = tok_len(sent_with_sep)
        if cur_tok + st > max_tokens and cur_chunk_text:
            start_char = doc_text.find(cur_chunk_text, cur_pos)
            end_char = start_char + len(cur_chunk_text)
            p_start, p_end = _range_to_pages(page_spans, start_char, end_char)
            chunks.append(
                {
                    "text": cur_chunk_text.strip(),
                    "page_start": p_start,
                    "page_end": p_end,
                    "char_len": len(cur_chunk_text),
                    "tok_est": cur_tok,
                }
            )
            if overlap_tokens > 0 and len(cur_chunk_text) > 0:
                tail_chars = max(0, len(cur_chunk_text) - overlap_tokens * 4)
                cur_pos = start_char + tail_chars
                cur_chunk_text = cur_chunk_text[tail_chars:]
                cur_tok = tok_len(cur_chunk_text)
            else:
                cur_chunk_text, cur_tok = "", 0
                cur_pos = end_char
        cur_chunk_text += sent_with_sep
        cur_tok += st

    if cur_chunk_text.strip():
        start_char = doc_text.find(cur_chunk_text, cur_pos)
        end_char = start_char + len(cur_chunk_text)
        p_start, p_end = _range_to_pages(page_spans, start_char, end_char)
        chunks.append(
            {
                "text": cur_chunk_text.strip(),
                "page_start": p_start,
                "page_end": p_end,
                "char_len": len(cur_chunk_text),
                "tok_est": cur_tok,
            }
        )
    return chunks


def chunk_chars(
    pages: List[str], window_chars: int, overlap_chars: int, *, use_tiktoken: bool = False, model: str | None = None
) -> List[Dict[str, Any]]:
    doc_text = "\n".join((p or "") for p in pages)
    page_spans = _build_page_index(pages)
    n = len(doc_text)
    chunks = []
    step = max(1, window_chars - overlap_chars)

    def tok_len(s: str) -> int:
        return count_tokens(s, model=model) if use_tiktoken else approx_token_len(s)

    for i in range(0, n, step):
        seg = doc_text[i : i + window_chars]
        if not seg.strip():
            continue
        start_char = i
        end_char = i + len(seg)
        p_start, p_end = _range_to_pages(page_spans, start_char, end_char)
        chunks.append(
            {
                "text": seg,
                "page_start": p_start,
                "page_end": p_end,
                "char_len": len(seg),
                "tok_est": tok_len(seg),
            }
        )
    return chunks


def _recursive_split(text: str, max_tokens: int, seps: list[str], tok_len) -> list[str]:
    if tok_len(text) <= max_tokens or not seps:
        if tok_len(text) <= max_tokens:
            return [text]
        win = max_tokens * 4
        return [text[i : i + win] for i in range(0, len(text), win)]
    sep = seps[0]
    parts = text.split(sep) if sep != "" else list(text)
    if len(parts) == 1:
        return _recursive_split(text, max_tokens, seps[1:], tok_len)
    out = []
    for i, p in enumerate(parts):
        seg = p if sep == "" else (p if i == len(parts) - 1 else p + sep)
        out.extend(_recursive_split(seg, max_tokens, seps[1:], tok_len))
    return out


def chunk_recursive(
    pages: List[str], max_tokens: int, overlap_tokens: int, *, use_tiktoken: bool = False, model: str | None = None
) -> List[dict]:
    doc = "\n".join((p or "") for p in pages)
    seps = ["\n\n## ", "\n\n", "\n", ". ", " ", ""]
    spans = _build_page_index(pages)

    def tok_len(s: str) -> int:
        return count_tokens(s, model=model) if use_tiktoken else approx_token_len(s)

    raw_chunks = _recursive_split(doc, max_tokens, seps, tok_len)
    chunks = []
    cur_pos = 0
    for rc in raw_chunks:
        start_c = doc.find(rc, cur_pos)
        if start_c == -1:
            start_c = cur_pos
        end_c = start_c + len(rc)
        p1, p2 = _range_to_pages(spans, start_c, end_c)
        txt = rc.strip()
        if txt:
            chunks.append(
                {
                    "text": txt,
                    "page_start": p1,
                    "page_end": p2,
                    "char_len": len(txt),
                    "tok_est": tok_len(txt),
                }
            )
        cur_pos = end_c

    if overlap_tokens > 0 and chunks:
        tail_chars = overlap_tokens * 4
        for i in range(1, len(chunks)):
            tail = chunks[i - 1]["text"][-tail_chars:]
            chunks[i]["text"] = (tail + "\n" + chunks[i]["text"]).strip()
    return chunks
