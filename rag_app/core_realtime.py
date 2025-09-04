# rag_app/core_realtime.py - Lógica de indexação em tempo real (documento a documento)

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

# --- Dependências do Projeto ---
from .utils import log, to_vec
from .db import (
    connect_pg,
    bulk_insert_chunks,
)
from .backends import LocalBackend, OpenAIBackend
from .core import (  # Reutiliza funções do core principal
    extract_pdf_pages_text,
    iter_docs_from_entry,
    _aplainar_e_mapear_meta,
    CHUNK_FUNCTIONS,
    initialize_database,
    upsert_document,
    filter_new_chunks,
)

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kw):
        return x


def process_documents_realtime(config: dict):
    """
    Processa documentos um a um, gerando e inserindo embeddings em tempo real.
    """
    dsn = os.getenv("PG_DSN")
    if not dsn:
        log(fatal="A variável de ambiente PG_DSN não está definida.")
        sys.exit(1)

    # --- Inicialização do Backend ---
    model_cfg = config["models"]
    backend_name = model_cfg["default_backend"]
    log("backend.init.start", mode="realtime", model=backend_name)
    if backend_name == "local":
        backend = LocalBackend(model_cfg["local"]["embedding_model"])
    elif backend_name == "openai":
        backend = OpenAIBackend(model_cfg["openai"]["embedding_model"])
    else:
        log(fatal=f"Backend '{backend_name}' desconhecido.")
        sys.exit(1)
    log("backend.init.done")

    # --- Configurações ---
    db_cfg = config["database"]
    idx_cfg = config["indexing"]
    man_path = Path(idx_cfg["manifest_path"])

    conn = connect_pg(dsn)
    try:
        with conn.cursor() as cur:
            if idx_cfg.get("reset_strategies", False):
                for tag in config["strategies"]:
                    tbl = f"{db_cfg['chunks_prefix']}_{tag}"
                    log("db.reset", table=tbl)
                    cur.execute(f"TRUNCATE TABLE {tbl} CASCADE;")
                conn.commit()

            with man_path.open("r", encoding="utf-8") as fp:
                manifest_lines = fp.readlines()
                limit = idx_cfg.get("limit_entries") or len(manifest_lines)

                pbar = tqdm(manifest_lines[:limit], desc="Indexação em Tempo Real")
                for line in pbar:
                    entry = json.loads(line)
                    meta_original = entry.get("meta", {})
                    doc_meta_aplainado = _aplainar_e_mapear_meta(meta_original)
                    # Guarda o meta original completo junto com os campos aplainados
                    meta_para_chunk = {
                        **(doc_meta_aplainado or {}),
                        "_doc_meta": meta_original or {},
                    }

                    for doc_id, pdf_path, is_tmp, pdf_sha in iter_docs_from_entry(
                        entry
                    ):
                        upsert_document(
                            cur, db_cfg["docs_table"], doc_id, entry, sha256_doc=pdf_sha
                        )

                        pages = extract_pdf_pages_text(pdf_path)
                        if not any(p.strip() for p in pages):
                            if is_tmp:
                                pdf_path.unlink(missing_ok=True)
                            continue

                        EMBED_BATCH = (config.get("indexing", {}) or {}).get("embed_batch", 64)
                        source_filename = pdf_path.name

                        # 1) Agrupa pendências por tabela como uma lista ordenada (sem usar texto como chave)
                        pendentes_por_tabela = defaultdict(list)  # {tbl: [(cid, tag, ch_obj, meta), ...]}

                        for tag, cfg in config["strategies"].items():
                            tbl = f"{db_cfg['chunks_prefix']}_{tag}"
                            kind = cfg.get("kind")
                            chunk_func = CHUNK_FUNCTIONS.get(kind)
                            if not chunk_func:
                                continue

                            params = {}
                            if kind in ["pages", "paras", "sent", "rec"]:
                                params = {
                                    "max_tokens": cfg.get("max_tokens"),
                                    "overlap_tokens": cfg.get("overlap"),
                                }
                                if cfg.get("use_tiktoken"):
                                    params["use_tiktoken"] = True
                                    token_model = (
                                        (config.get("models", {}).get("openai", {}) or {}).get(
                                            "embedding_model"
                                        )
                                    )
                                    if token_model:
                                        params["model"] = token_model
                            elif kind == "chars":
                                params = {"window_chars": cfg.get("window"), "overlap_chars": cfg.get("overlap_chars")}
                            params = {k: v for k, v in params.items() if v is not None}
                            if not params:
                                continue

                            chunks = chunk_func(pages, **params)
                            novos = filter_new_chunks(cur, tbl, doc_id, tag, chunks, source_filename)
                            for cid, ch_obj in novos:
                                pendentes_por_tabela[tbl].append((cid, tag, ch_obj, meta_para_chunk))

                        # Nada novo? segue o baile
                        if not pendentes_por_tabela:
                            if is_tmp:
                                pdf_path.unlink(missing_ok=True)
                            continue

                        # 2) Para cada tabela, embeda e insere em micro-batches
                        for tabela, pendentes in pendentes_por_tabela.items():
                            n = len(pendentes)
                            if n == 0:
                                continue

                            # loop em lotes de EMBED_BATCH
                            for start in range(0, n, EMBED_BATCH):
                                lote = pendentes[start : start + EMBED_BATCH]

                                # prepara textos mantendo ordem
                                textos = [p[2]["text"] for p in lote]
                                # chama backend (que pode fazer subchunk interno se precisar)
                                embeddings = backend.embed(textos)

                                linhas = []
                                for (cid, tag, ch_obj, meta), emb in zip(lote, embeddings):
                                    if not emb:
                                        log("embedding.skip", reason="failed_embedding_generation", cid=cid)
                                        continue
                                    linhas.append((
                                        cid,
                                        tag,
                                        doc_id,
                                        ch_obj.get("page_start"),
                                        ch_obj.get("page_end"),
                                        ch_obj.get("tok_est"),
                                        ch_obj.get("char_len"),
                                        ch_obj["text"],
                                        json.dumps(meta, ensure_ascii=False),
                                        to_vec(emb),
                                    ))

                                if not linhas:
                                    continue

                                # Insere com função resiliente (lota em sublotes internos e comita)
                                bulk_insert_chunks(cur, tabela, linhas)

                                # Commit explícito POR TABELA / POR SUBLOTE: isola transações e libera memória no servidor
                                conn.commit()

                        # remove PDF temporário, se aplicável
                        if is_tmp:
                            pdf_path.unlink(missing_ok=True)
    finally:
        conn.close()
