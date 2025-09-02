# rag_app/answer.py - Síntese de respostas com citações multi-documento

from typing import List, Dict, Any
from collections import defaultdict
from openai import OpenAI
from .utils import log


def select_diverse_contexts(
    contexts: List[Dict[str, Any]],
    *,
    per_doc_quota: int = 2,
    max_chunks: int = 12,
    max_chars_per_chunk: int = 1200,
) -> List[Dict[str, Any]]:
    """Seleciona trechos maximizando diversidade por documento e qualidade (score).

    - Limita quantidade por doc_id
    - Ordena por score decrescente (se existir), mantendo variedade
    - Trunca o texto de cada chunk
    """
    # ordena global por score desc (se não houver, mantém ordem original)
    ranked = sorted(
        contexts,
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )

    picked: List[Dict[str, Any]] = []
    per_doc_count = defaultdict(int)

    for c in ranked:
        doc_id = c.get("doc_id") or "__unknown__"
        if per_doc_count[doc_id] >= per_doc_quota:
            continue
        # copia raso e trunca
        cc = dict(c)
        txt = (cc.get("text") or "").strip()
        if max_chars_per_chunk > 0 and len(txt) > max_chars_per_chunk:
            cc["text"] = txt[: max_chars_per_chunk - 3] + "..."
        picked.append(cc)
        per_doc_count[doc_id] += 1
        if len(picked) >= max_chunks:
            break

    return picked


def synthesize_answer(
    question: str,
    contexts: List[Dict[str, Any]],
    *,
    model: str = "gpt-4o-mini",
    language: str = "pt-BR",
) -> Dict[str, Any]:
    """Gera resposta final com citações a partir de múltiplos trechos.

    Retorna: {"answer": str, "citations": [{doc_id, pages, chunk_id}], "model": model}
    """
    try:
        client = OpenAI()
    except Exception as e:
        raise RuntimeError("Falha ao inicializar OpenAI: defina OPENAI_API_KEY") from e

    cit_lines = []
    for i, c in enumerate(contexts, start=1):
        doc = c.get("doc_id") or "?"
        p1 = c.get("page_start")
        p2 = c.get("page_end")
        cid = c.get("chunk_id")
        txt = (c.get("text") or "").strip()
        cit_lines.append(
            f"[CTX {i}] doc_id={doc} pages={p1}-{p2} chunk_id={cid}\n{txt}"
        )
    joined = "\n\n".join(cit_lines)

    system = (
        "Você é um analista de mercado especializado em licitações públicas brasileiras. "
        "Responda de forma objetiva e fundamentada nos CONTEXTOS fornecidos. "
        "Cite as fontes entre colchetes no fim de cada afirmação relevante no formato [doc_id:páginas]. "
        "Se houver incerteza ou falta de dados completos, deixe isso explícito."
    )

    user = (
        f"Pergunta: {question}\n\n"
        f"CONTEXTOS (vários documentos):\n{joined}\n\n"
        "Instruções de saída:\n"
        "- Responda em português claro.\n"
        "- Liste números agregados (soma, médias) apenas se houver base suficiente nos contextos.\n"
        "- Inclua uma seção final 'Fontes' com a lista distinta de doc_id e páginas citadas."
    )

    log("synth.start", model=model, contexts=len(contexts))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    answer = (resp.choices[0].message.content or "").strip()
    log("synth.done")
    # não tentamos parsear automaticamente as citações; retornamos o texto e o conjunto de docs usados
    cited_docs = []
    seen = set()
    for c in contexts:
        doc = c.get("doc_id")
        if doc and doc not in seen:
            seen.add(doc)
            cited_docs.append({"doc_id": doc})

    return {"answer": answer, "citations": cited_docs, "model": model}
