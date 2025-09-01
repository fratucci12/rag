# rag_app/backends.py - Classes para geração de embeddings

import os
from typing import List
from dotenv import load_dotenv

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .utils import log

load_dotenv()


# --- Classes de Backend ---
class Backend:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class LocalBackend(Backend):
    def __init__(self, model_name: str):
        if SentenceTransformer is None:
            raise RuntimeError("Instale a biblioteca 'sentence-transformers'")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        arr = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in arr]


class OpenAIBackend(Backend):
    def __init__(self, model: str, batch_size: int = 128):
        if OpenAI is None:
            raise RuntimeError("Instale a biblioteca 'openai'")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Defina a variável de ambiente OPENAI_API_KEY")
        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            resp = self.client.embeddings.create(model=self.model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            log("openai.embed.batch_fallback", reason=str(e))

        all_embeddings = []
        # Processa em lotes para não exceder limites de API
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                all_embeddings.extend([d.embedding for d in resp.data])
            except Exception as e:
                log("openai.embed.error", error=str(e))
                # Adiciona vetores vazios para manter a correspondência de tamanho
                all_embeddings.extend([[]] * len(batch))
        return all_embeddings


# [NOVO] Classe dedicada para interagir com o Batch API da OpenAI
class OpenAIBatchProcessor:
    def __init__(self):
        if OpenAI is None:
            raise RuntimeError("Instale a biblioteca 'openai'")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Defina a variável de ambiente OPENAI_API_KEY")
        self.client = OpenAI()

    def upload_file(self, file_path: str) -> str:
        """Envia um ficheiro para a OpenAI e retorna o seu ID."""
        with open(file_path, "rb") as f:
            file_object = self.client.files.create(file=f, purpose="batch")
        return file_object.id

    def create_batch(self, file_id: str, embedding_model: str) -> str:
        """Cria um trabalho em lote a partir de um ID de ficheiro."""
        endpoint = "/v1/embeddings"
        completion_window = "24h"

        result = self.client.batches.create(
            input_file_id=file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata={"embedding_model": embedding_model},
        )
        return result.id

    def check_status(self, batch_id: str) -> dict:
        """Verifica o estado de um trabalho em lote."""
        batch = self.client.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "total_requests": batch.request_counts.total,
            "completed_requests": batch.request_counts.completed,
            "failed_requests": batch.request_counts.failed,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }

    def get_results(self, file_id: str) -> str:
        """Obtém o conteúdo de um ficheiro de resultados."""
        if not file_id:
            return ""
        content = self.client.files.content(file_id)
        return content.text

    # Compatibilidade retroativa
    get_file_content = get_results

    def list_active_batches(self) -> list[dict]:
        """Lista lotes ainda ativos (enqueued/validating/in_progress/finalizing)."""
        try:
            resp = self.client.batches.list(limit=100)
        except Exception:
            return []
        out = []
        data = getattr(resp, "data", []) or []
        for b in data:
            st = getattr(b, "status", None)
            if st in ("enqueued", "validating", "in_progress", "finalizing"):
                out.append({"id": getattr(b, "id", None), "status": st})
        return out
