# rag_app/db.py - Funções para interação com o banco de dados PostgreSQL

import json
import time
import psycopg2 as psycopg
from psycopg2 import OperationalError
from psycopg2.extras import register_uuid
from typing import List, Dict, Any
from psycopg2.extensions import connection as PgConnection

# Importa funções auxiliares do nosso módulo de utilidades
from .utils import log, make_chunk_id, sanitize_text, to_vec

register_uuid()

try:
    from psycopg2.extras import execute_values
except ImportError:
    def execute_values(cur, sql_with_values_placeholder, rows, page_size=200):
        """Fallback quando psycopg2.extras.execute_values não está disponível."""
        raise ImportError(
            "psycopg2.extras.execute_values não disponível; instale psycopg2-binary"
        )


# --- Definições de Schema (DDL) ---
DDL_DOCS = """
CREATE TABLE IF NOT EXISTS {docs} (
  doc_id text PRIMARY KEY, sha256 text, sha256_doc text, file_path text,
  source_url text, content_type text, meta jsonb
);
"""
DDL_DOCS_INDEXES = """
CREATE INDEX IF NOT EXISTS {docs}_sha256_idx ON {docs}(sha256);
CREATE INDEX IF NOT EXISTS {docs}_sha256_doc_idx ON {docs}(sha256_doc);
CREATE INDEX IF NOT EXISTS {docs}_meta_gin ON {docs} USING GIN (meta);
"""
DDL_CHUNKS = """
CREATE TABLE IF NOT EXISTS {table} (
  chunk_id uuid PRIMARY KEY, strategy text NOT NULL, doc_id text REFERENCES {docs}(doc_id) ON DELETE CASCADE,
  page_start int, page_end int, tok_est int, char_len int, text text,
  meta jsonb,
  embedding vector({dim}), created_at timestamptz DEFAULT now()
);
"""
IDX_CHUNKS_META = (
    "CREATE INDEX IF NOT EXISTS {table}_meta_gin ON {table} USING GIN (meta);"
)
IDX_HNSW = "CREATE INDEX IF NOT EXISTS {table}_hnsw ON {table} USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);"


# --- Funções de Conexão e Schema ---
def connect_pg(dsn: str) -> PgConnection:
    extras_str = "keepalives=1 keepalives_idle=30 keepalives_interval=10 keepalives_count=5 target_session_attrs=read-write"
    extra_kwargs = {
        k.strip(): v.strip()
        for part in extras_str.split()
        if (k_v := part.split("=", 1)) and len(k_v) == 2
        for k, v in [k_v]
    }
    conn = psycopg.connect(dsn, **extra_kwargs, prepare_threshold=None)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '0'")
        cur.execute("SET idle_in_transaction_session_timeout = '0'")
    return conn


def ensure_schema(conn, docs_table: str, chunk_table: str, dim: int, use_hnsw: bool):
    with conn.cursor() as cur:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception:
            log("db.vector_ext.warn", reason="Falha ao criar extensão.")
        cur.execute(DDL_DOCS.format(docs=docs_table))
        cur.execute(DDL_DOCS_INDEXES.format(docs=docs_table))
        cur.execute(DDL_CHUNKS.format(table=chunk_table, docs=docs_table, dim=dim))
        cur.execute(IDX_CHUNKS_META.format(table=chunk_table))
        if use_hnsw:
            cur.execute(IDX_HNSW.format(table=chunk_table))
    conn.commit()


# --- Funções de Manipulação de Dados ---
def upsert_document(cur, docs_table, doc_id, entry, sha256_doc=None):
    # Preenche 'sha256' com o hash do PDF (sha256_doc) quando disponvel
    sha_for_row = sha256_doc or entry.get("sha256")
    cur.execute(
        f"""
        INSERT INTO {docs_table} (doc_id, sha256, sha256_doc, file_path, source_url, content_type, meta)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (doc_id) DO UPDATE SET
          sha256 = EXCLUDED.sha256, sha256_doc = EXCLUDED.sha256_doc, file_path = EXCLUDED.file_path,
          source_url = EXCLUDED.source_url, content_type = EXCLUDED.content_type, meta = EXCLUDED.meta
        """,
        (
            doc_id,
            sha_for_row,
            sha256_doc or entry.get("sha256_doc"),
            entry.get("file_path"),
            entry.get("source_url"),
            entry.get("content_type"),
            json.dumps(entry.get("meta") or {}, ensure_ascii=False),
        ),
    )


# [CORREÇÃO] Adicionado `source_filename` para passar para make_chunk_id
def filter_new_chunks(
    cur,
    table: str,
    doc_id: str,
    strategy: str,
    chunks: list[dict],
    source_filename: str,
):
    prepared = []
    for ch in chunks:
        txt = sanitize_text(ch.get("text") or "")
        if not txt:
            continue
        cid = make_chunk_id(
            doc_id,
            strategy,
            txt,
            ch.get("page_start"),
            ch.get("page_end"),
            source_filename,
        )
        ch2 = dict(ch)
        ch2["text"] = txt
        prepared.append((cid, ch2))

    # A verificação de duplicatas na BD continua a ser útil para re-execuções
    cur.execute(
        f"SELECT chunk_id::text FROM {table} WHERE doc_id=%s AND strategy=%s",
        (doc_id, strategy),
    )
    already = {r[0] for r in cur.fetchall()}
    return [(cid, ch) for (cid, ch) in prepared if cid not in already]


def _flush_batch_with_retry(
    cur,
    sql: str,
    rows,
    *,
    max_retries: int = 4,
    sleep_base: float = 0.7,
    initial_batch_size: int = 100,  # tamanho inicial do lote externo
    inner_page_size: int = 50,  # page_size dentro do execute_values
    commit_each_subbatch: bool = True,  # commit por sublote
) -> None:
    """
    Envia 'rows' em sublotes menores e reexecuta em caso de falha.
    - Divide rows em blocos (initial_batch_size), cada bloco vira um execute_values com page_size menor (inner_page_size).
    - Em erro, reduz o tamanho do sublote pela metade; em último caso, insere 1 a 1.
    - Faz commit por sublote para aliviar a transação e memória no servidor.
    """
    n = len(rows)
    if n == 0:
        return

    i = 0
    batch_size = max(1, min(initial_batch_size, n))

    while i < n:
        sub = rows[i : i + batch_size]
        attempts = 0

        while True:
            try:
                # IMPORTANTE: não usar page_size=len(sub)!
                execute_values(cur, sql, sub, page_size=min(inner_page_size, len(sub)))
                if commit_each_subbatch:
                    cur.connection.commit()
                break  # sublote OK
            except OperationalError as e:
                # rollback se possível
                try:
                    cur.connection.rollback()
                except Exception:
                    pass

                attempts += 1
                # log opcional (seu utils.log): tamanho do sublote ajuda a diagnosticar
                try:
                    from .utils import log

                    log(
                        "db.flush.retry",
                        attempt=attempts,
                        max=max_retries,
                        batch_size=len(sub),
                        error=str(e)[:200],
                    )
                except Exception:
                    pass

                if attempts >= max_retries:
                    # última cartada: dividir o sublote em dois (ou 1 a 1)
                    if len(sub) > 1:
                        mid = len(sub) // 2
                        _flush_batch_with_retry(
                            cur,
                            sql,
                            sub[:mid],
                            max_retries=max_retries,
                            sleep_base=sleep_base,
                            initial_batch_size=max(1, mid // 2),
                            inner_page_size=min(inner_page_size, mid),
                            commit_each_subbatch=commit_each_subbatch,
                        )
                        _flush_batch_with_retry(
                            cur,
                            sql,
                            sub[mid:],
                            max_retries=max_retries,
                            sleep_base=sleep_base,
                            initial_batch_size=max(1, (len(sub) - mid) // 2),
                            inner_page_size=min(inner_page_size, len(sub) - mid),
                            commit_each_subbatch=commit_each_subbatch,
                        )
                        break
                    else:
                        # até 1 linha falhou: propaga
                        raise

                # backoff exponencial leve
                time.sleep(min(8.0, sleep_base * (2 ** (attempts - 1))))

            except Exception as e:
                # erros não-Operacionais (ex.: tamanho/valor inválido)
                try:
                    cur.connection.rollback()
                except Exception:
                    pass

                if len(sub) > 1:
                    mid = len(sub) // 2
                    _flush_batch_with_retry(
                        cur,
                        sql,
                        sub[:mid],
                        max_retries=max_retries,
                        sleep_base=sleep_base,
                        initial_batch_size=max(1, mid // 2),
                        inner_page_size=min(inner_page_size, mid),
                        commit_each_subbatch=commit_each_subbatch,
                    )
                    _flush_batch_with_retry(
                        cur,
                        sql,
                        sub[mid:],
                        max_retries=max_retries,
                        sleep_base=sleep_base,
                        initial_batch_size=max(1, (len(sub) - mid) // 2),
                        inner_page_size=min(inner_page_size, len(sub) - mid),
                        commit_each_subbatch=commit_each_subbatch,
                    )
                    break
                else:
                    raise

        i += len(sub)

def bulk_insert_chunks(
    cur,
    table: str,
    rows,
    *,
    init_batch: int = 100,  # você pode ajustar p/ 200 se estiver estável
    inner_page: int = 50,  # 50 geralmente é um bom equilíbrio
):
    """Insere um lote de linhas de chunks na tabela especificada."""
    if not rows:
        return

    sql = f"""
        INSERT INTO {table}
        (chunk_id, strategy, doc_id, page_start, page_end, tok_est, char_len, text, meta, embedding)
        VALUES %s
        ON CONFLICT DO NOTHING
    """

    _flush_batch_with_retry(
        cur,
        sql,
        rows,
        max_retries=4,
        sleep_base=0.7,
        initial_batch_size=init_batch,
        inner_page_size=inner_page,
        commit_each_subbatch=True,
    )
