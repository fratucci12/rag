# rag_app/utils.py - Funções auxiliares

import json
import os
import math
import uuid
import hashlib
import re
import datetime as dt
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

# --- Constantes ---
NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

# --- Logging ---
LOG_JSON = os.getenv("LOG_JSON", "false").lower() in ("1", "true", "yes")
LOG_REDACT = ("PNCP_BEARER", "AUTHORIZATION", "PG_DSN", "OPENAI_API_KEY")


def _now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")


def _redact(d: dict) -> dict:
    return {k: ("***" if k.upper() in LOG_REDACT else v) for k, v in (d or {}).items()}


def log(event: str, **fields):
    fields = _redact(fields)
    if LOG_JSON:
        print(
            json.dumps({"ts": _now_iso(), "event": event, **fields}, ensure_ascii=False)
        )
    else:
        parts = " ".join(f"{k}={repr(v)}" for k, v in fields.items())
        print(f"[{_now_iso()}] {event} {parts}")


def load_config(path: str) -> Dict[str, Any]:
    """Carrega um ficheiro de configuração YAML."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        log(fatal=f"Ficheiro de configuração não encontrado em: {path}")
        raise
    except Exception as e:
        log(fatal=f"Erro ao ler o ficheiro de configuração: {e}")
        raise


# --- Funções de Texto e Normalização ---
def sanitize_text(s: str) -> str:
    return (s or "").replace("\x00", " ").strip()


def approx_token_len(s: str) -> int:
    return max(1, math.ceil(len(s) / 4))


def _get_token_encoder(
    model: Optional[str] = None, encoding_name: Optional[str] = None
):
    """
    Tenta obter um encoder do tiktoken para um dado modelo ou encoding base.
    - Se tiktoken não estiver instalado, retorna None.
    - Se o modelo não for reconhecido, tenta encoding base (por padrão cl100k_base).
    """
    try:
        import tiktoken  # type: ignore

        enc = None
        if model:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                enc = None
        if enc is None:
            base = encoding_name or "cl100k_base"
            try:
                enc = tiktoken.get_encoding(base)
            except Exception:
                enc = None
        return enc
    except ImportError:
        return None


def count_tokens(
    text: str,
    model: Optional[str] = None,
    *,
    encoding_name: Optional[str] = None,
    encoder=None,
) -> int:
    """
    Conta tokens de um texto.
    - Se um `encoder` (tiktoken) for fornecido, usa-o.
    - Caso contrário, tenta resolver via `model` ou `encoding_name`.
    - Se tiktoken não estiver disponível, usa a estimativa approx (len/4).
    """
    s = text or ""
    enc = encoder or _get_token_encoder(model, encoding_name)
    if enc is not None:
        try:
            return len(enc.encode(s))
        except Exception:
            pass
    return approx_token_len(s)


# --- Funções de Hash e IDs ---
def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# [CORREÇÃO] Adicionado `source_filename` para garantir IDs únicos
def make_chunk_id(
    doc_id: str,
    strategy: str,
    text: str,
    page_start: Any,
    page_end: Any,
    source_filename: str,
) -> str:
    h = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
    # A chave agora inclui o nome do ficheiro para garantir unicidade entre PDFs do mesmo doc_id
    key = (
        f"{doc_id}|{source_filename}|{strategy}|{page_start or ''}|{page_end or ''}|{h}"
    )
    return str(uuid.uuid5(NAMESPACE, key))


# --- Funções de Vetor ---
def to_vec(v: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"
