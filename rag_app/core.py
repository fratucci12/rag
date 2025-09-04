# rag_app/core.py - Lógica principal de indexação de documentos

import os
import sys
import json
import uuid
import tempfile
import hashlib
from typing import List, Dict, Any, Optional
import zipfile
from pathlib import Path

# --- Dependências do Projeto ---
from .utils import log, sha256_file
from .db import (
    connect_pg,
    ensure_schema,
    upsert_document,
    filter_new_chunks,
)
from . import chunking

try:
    import fitz  # PyMuPDF
except ImportError:
    log(fatal="PyMuPDF não encontrado. Por favor, instale com: pip install PyMuPDF")
    sys.exit(1)
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kw):
        return x


# --- Funções de Processamento de Arquivos ---
def extract_pdf_pages_text(pdf_path: Path) -> List[str]:
    """Extrai o texto de todas as páginas de um ficheiro PDF."""
    pages = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pages.append(page.get_text("text") or "")
    except Exception as e:
        log("pdf.extract.error", path=str(pdf_path), error=str(e))
    return pages


def iter_docs_from_entry(entry: Dict) -> List[tuple]:
    """
    Itera sobre uma entrada do manifesto e retorna os caminhos para os PDFs,
    extraindo-os de ZIPs se necessário.
    """
    out = []
    p = Path(entry.get("file_path", ""))
    if not p.exists():
        log("manifest.entry.skip", reason="file_not_found", path=str(p))
        return out

    zip_sha = entry.get("sha256", "nohash")
    if p.suffix.lower() == ".pdf":
        pdf_sha = sha256_file(p)
        out.append((f"{zip_sha}::root", p, False, pdf_sha))
    elif p.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(p, "r") as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".pdf"):
                        with zf.open(name) as pdf_file:
                            data = pdf_file.read()
                            pdf_sha = hashlib.sha256(data).hexdigest()
                            tmp_dir = Path(tempfile.gettempdir()) / "rag_temp_pdfs"
                            tmp_dir.mkdir(exist_ok=True)
                            tmp_path = tmp_dir / f"{pdf_sha}.pdf"
                            tmp_path.write_bytes(data)
                            out.append((f"{zip_sha}::{name}", tmp_path, True, pdf_sha))
        except zipfile.BadZipFile:
            log("zip.extract.error", reason="bad_zip_file", path=str(p))
        except Exception as e:
            log("zip.extract.error", reason=str(e), path=str(p))
    return out


# --- Mapeamento e Transformação ---
CHUNK_FUNCTIONS = {
    "pages": chunking.chunk_pages,
    "paras": chunking.chunk_paragraphs,
    "sent": chunking.chunk_sentences,
    "chars": chunking.chunk_chars,
    "rec": chunking.chunk_recursive,
}


def _aplainar_e_mapear_meta(meta_original: Dict) -> Dict:
    """Pega a estrutura de metadados aninhada e a transforma num dicionário simples e limpo."""
    if not meta_original:
        return {}
    compra_info = meta_original.get("compra", {})
    meta_aplainado = {
        "cnpj": compra_info.get("cnpj"),
        "orgao_nome": compra_info.get("orgao", {}).get("nome"),
        "ano": compra_info.get("ano"),
        "municipio_nome": meta_original.get("unidadeOrgao", {}).get("municipioNome"),
        "estado_sigla": meta_original.get("unidadeOrgao", {}).get("ufSigla"),
        "objeto_compra": compra_info.get("objeto"),
        "sequencial_compra": compra_info.get("sequencial"),
        "esfera_compra": compra_info.get("esfera"),
        "poder_compra": compra_info.get("poder"),
        "data_publicacao": compra_info.get("data_publicacao"),
    }
    filtrado = {k: v for k, v in meta_aplainado.items() if v is not None}
    if "orgao_nome" not in filtrado:
        filtrado["orgao_nome"] = meta_aplainado.get("orgao_nome")
    return filtrado


# --- Lógica Principal de Indexação ---
def initialize_database(config: Dict[str, Any]):
    """Cria o schema do banco de dados para todas as estratégias configuradas."""
    dsn = os.getenv("PG_DSN")
    if not dsn:
        log(fatal="A variável de ambiente PG_DSN não está definida.")
        sys.exit(1)
    conn = connect_pg(dsn)
    try:
        model_cfg = config["models"]
        # Seleciona dimensão conforme backend padrão
        backend_name = model_cfg.get("default_backend", "openai")
        if backend_name == "local":
            embedding_dim = model_cfg.get("local", {}).get("embedding_dim")
        else:
            embedding_dim = model_cfg.get("openai", {}).get("embedding_dim")
        if not embedding_dim:
            # fallback seguro
            embedding_dim = 1536
        docs_table = config["database"]["docs_table"]
        chunks_prefix = config["database"]["chunks_prefix"]
        use_hnsw = config["database"]["index_type"] == "hnsw"
        log("db.schema.init.start")
        for tag in config["strategies"]:
            tbl = f"{chunks_prefix}_{tag}"
            ensure_schema(conn, docs_table, tbl, embedding_dim, use_hnsw)
        log("db.schema.init.done")
    finally:
        conn.close()


def split_manifest_into_batches(
    config: Dict[str, Any], output_dir: Path, token_limit: int
) -> List[Dict]:
    """Divide o manifesto em vários lotes de acordo com um limite de tokens."""
    dsn = os.getenv("PG_DSN")
    if not dsn:
        log(fatal="A variável de ambiente PG_DSN não está definida.")
        return []

    db_cfg = config["database"]
    idx_cfg = config["indexing"]
    man_path = Path(idx_cfg["manifest_path"])

    conn = connect_pg(dsn)
    all_new_chunks = []

    # [CORREÇÃO] Adicionado conjunto para rastrear IDs já processados nesta execução
    processed_cids_in_this_batch = set()

    with man_path.open("r", encoding="utf-8") as man_fp, conn.cursor() as cur:
        if idx_cfg.get("reset_strategies", False):
            for tag in config["strategies"]:
                tbl = f"{db_cfg['chunks_prefix']}_{tag}"
                log("db.reset", table=tbl)
                cur.execute(f"TRUNCATE TABLE {tbl} CASCADE;")
            conn.commit()

        manifest_lines = man_fp.readlines()

        pbar = tqdm(manifest_lines, desc="A verificar chunks novos")
        for line in pbar:
            try:
                entry = json.loads(line)
                meta_original = entry.get("meta", {})
                    doc_meta_aplainado = _aplainar_e_mapear_meta(meta_original)
                    # Meta a ser gravado em chunks: flatten + original completo
                    meta_para_chunk = {**(doc_meta_aplainado or {}), "_doc_meta": meta_original or {}}

                for doc_id, pdf_path, is_tmp, pdf_sha in iter_docs_from_entry(entry):
                    upsert_document(
                        cur, db_cfg["docs_table"], doc_id, entry, sha256_doc=pdf_sha
                    )
                    pages = extract_pdf_pages_text(pdf_path)
                    if not any(p.strip() for p in pages):
                        if is_tmp:
                            pdf_path.unlink(missing_ok=True)
                        continue

                    # [CORREÇÃO] O nome do ficheiro PDF original é extraído para ser usado no ID do chunk
                    source_filename = pdf_path.name

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
                            # opcional: usar tiktoken para contagem de tokens no chunking
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
                            params = {
                                "window_chars": cfg.get("window"),
                                "overlap_chars": cfg.get("overlap_chars"),
                            }

                        params = {k: v for k, v in params.items() if v is not None}
                        if not params:
                            continue

                        chunks = chunk_func(pages, **params)
                        # [CORREÇÃO] Passa o nome do ficheiro para garantir a unicidade do ID
                        new_chunks_with_cid = filter_new_chunks(
                            cur, tbl, doc_id, tag, chunks, source_filename
                        )

                        for cid, chunk_obj in new_chunks_with_cid:
                            # [CORREÇÃO] Dupla verificação para garantir que o ID não foi adicionado nesta execução
                            if cid in processed_cids_in_this_batch:
                                continue

                            all_new_chunks.append(
                                {
                                    "cid": cid,
                                    "doc_id": doc_id,
                                    "tag": tag,
                                    "tbl": tbl,
                                    "doc_meta": meta_para_chunk,
                                    "chunk_obj": chunk_obj,
                                }
                            )
                            processed_cids_in_this_batch.add(cid)
                    if is_tmp:
                        pdf_path.unlink(missing_ok=True)
            except Exception as e:
                # Efetua rollback para limpar estado de transação abortada e prosseguir com a próxima linha
                try:
                    conn.rollback()
                except Exception:
                    pass
                log("core.error", error=str(e), line=line.strip())
    conn.close()

    if not all_new_chunks:
        return []

    embedding_model = config["models"]["openai"]["embedding_model"]
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(embedding_model)
        except (KeyError, ValueError, TypeError):
            enc = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        enc = None
        log("tiktoken.missing", warning="Libraria tiktoken não disponível, usando contagem de caracteres")
    batches = []
    batch_tokens = []  # total de tokens por batch
    batch_tok_lists = []  # tokens por chunk (para estatísticas)
    current_batch_chunks = []
    current_batch_tokens = 0
    current_batch_tok_list = []

    for chunk_data in all_new_chunks:
        text = chunk_data["chunk_obj"]["text"]
        chunk_tokens = len(enc.encode(text or "")) if enc else len(text or "")
        if current_batch_tokens + chunk_tokens > token_limit and current_batch_chunks:
            batches.append(current_batch_chunks)
            batch_tokens.append(current_batch_tokens)
            batch_tok_lists.append(current_batch_tok_list)
            current_batch_chunks = []
            current_batch_tokens = 0
            current_batch_tok_list = []

        current_batch_chunks.append(chunk_data)
        current_batch_tokens += chunk_tokens
        current_batch_tok_list.append(chunk_tokens)

    if current_batch_chunks:
        batches.append(current_batch_chunks)
        batch_tokens.append(current_batch_tokens)
        batch_tok_lists.append(current_batch_tok_list)

    batch_files = []

    for i, batch_chunk_list in enumerate(batches):
        batch_num = i + 1
        input_file = output_dir / f"batch_input_{batch_num}.jsonl"
        metadata_file = output_dir / f"batch_metadata_{batch_num}.jsonl"

        with input_file.open("w", encoding="utf-8") as out_fp:
            for chunk_data in batch_chunk_list:
                cid = chunk_data["cid"]
                chunk_obj = chunk_data["chunk_obj"]
                request = {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {"input": chunk_obj["text"], "model": embedding_model},
                }
                out_fp.write(json.dumps(request, ensure_ascii=False) + "\n")
        # Escreve metadados como JSONL (uma linha por chunk), para casar com o leitor em scripts/batch_processor.py
        with metadata_file.open("w", encoding="utf-8") as meta_fp:
            for chunk_data in batch_chunk_list:
                cid = chunk_data["cid"]
                chunk_obj = chunk_data["chunk_obj"]
                meta_line = {
                    "chunk_id": cid,
                    "strategy": chunk_data.get("tag"),
                    "table": chunk_data.get("tbl"),
                    "doc_id": chunk_data.get("doc_id"),
                    "page_start": chunk_obj.get("page_start"),
                    "page_end": chunk_obj.get("page_end"),
                    "tok_est": chunk_obj.get("tok_est"),
                    "char_len": chunk_obj.get("char_len"),
                    "text": chunk_obj.get("text", ""),
                    "meta": chunk_data.get("doc_meta", {}),
                }
                meta_fp.write(json.dumps(meta_line, ensure_ascii=False) + "\n")

        # Estatísticas e contagem de tokens do batch no log
        try:
            tok_list = batch_tok_lists[i] if i < len(batch_tok_lists) else []
            total_tok = batch_tokens[i] if i < len(batch_tokens) else sum(tok_list)
            cnt = len(tok_list) or len(batch_chunk_list)
            min_tok = min(tok_list) if tok_list else None
            max_tok = max(tok_list) if tok_list else None
            avg_tok = (total_tok / cnt) if cnt else None
            log(
                "batch.pack.stats",
                batch=(i + 1),
                chunks=len(batch_chunk_list),
                tokens=total_tok,
                avg_tokens=avg_tok,
                min_tokens=min_tok,
                max_tokens=max_tok,
                token_limit=token_limit,
                approx=(enc is None),
                input=str(input_file),
                meta=str(metadata_file),
            )
        except Exception:
            pass

        batch_files.append({"input_file": input_file, "metadata_file": metadata_file})

    return batch_files
