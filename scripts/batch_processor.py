# scripts/batch_processor.py - Orquestrador do processo de embedding via Batch API da OpenAI

import os
import sys
import argparse
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Adiciona o diretório raiz ao path para permitir importações de 'rag_app'
sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_app.utils import log, load_config, to_vec
from rag_app.core import split_manifest_into_batches
from rag_app.backends import OpenAIBatchProcessor
from rag_app.db import connect_pg, bulk_insert_chunks
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kw):
        return x


try:
    from rich import print
except ImportError:
    pass

# --- adicione perto dos imports no topo (já tem os imports de os e dotenv) ---
def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(
            f"Variável de ambiente {name} não definida. Verifique seu .env."
        )
    return v


def retrieve_and_insert(
    batch_id: str,
    metadata_file: Path,
    processor: OpenAIBatchProcessor,
    config: dict,
):
    """Obtém os resultados de um lote concluído e insere-os na base de dados."""
    log("batch.retrieve.start", batch_id=batch_id)

    if not metadata_file.exists():
        log(
            "batch.retrieve.fatal",
            error=f"Ficheiro de metadados não encontrado em {metadata_file}.",
        )
        return False

    # 1) Verifica status do lote
    batch_info = processor.check_status(batch_id)
    status = batch_info.get("status")
    if status != "completed":
        log("batch.retrieve.wait", batch_id=batch_id, current_status=status)
        return False

    result_file_id = batch_info.get("output_file_id")
    if not result_file_id:
        log(
            "batch.retrieve.error",
            batch_id=batch_id,
            error="ID do ficheiro de resultados em falta.",
        )
        return False

    try:
        # 2) Carrega resultados (embeddings) e metadados
        results_content = processor.get_results(result_file_id)
        results = [json.loads(line) for line in results_content.strip().split("\n")]
        metadata = [
            json.loads(line) for line in metadata_file.read_text().strip().split("\n")
        ]

        if len(results) != len(metadata):
            log(
                "batch.retrieve.warn",
                reason="len_mismatch",
                results=len(results),
                meta=len(metadata),
            )

        # 3) Conecta no Postgres via PG_DSN do .env
        dsn = os.getenv("PG_DSN")
        if not dsn:
            raise RuntimeError(
                "Variável de ambiente PG_DSN não definida. Verifique seu .env."
            )
        conn = connect_pg(dsn)
        cur = conn.cursor()

        # chunks_prefix para compor nome da tabela, se necessário
        db_cfg = config.get("database", {})
        chunks_prefix = db_cfg.get("chunks_prefix", "chunks")

        # 4) Agrupa linhas por tabela
        from collections import defaultdict

        rows_by_table = defaultdict(list)

        for meta, result in zip(metadata, results):
            # extrai embedding do retorno batch
            emb = (
                result.get("response", {})
                .get("body", {})
                .get("data", [{}])[0]
                .get("embedding")
            )
            if not emb:
                log(
                    "embedding.skip",
                    reason="missing_embedding",
                    meta=meta.get("id") or meta.get("chunk_id"),
                )
                continue

            # vetor pgvector
            vec = to_vec(emb)

            # descobre a tabela: usa meta["table"] ou prefixo + strategy
            strategy = meta.get("strategy")
            table = meta.get("table") or (
                f"{chunks_prefix}_{strategy}" if strategy else None
            )
            if not table:
                log("batch.retrieve.skip", reason="missing_table_strategy", meta=meta)
                continue

            # mapeia campos obrigatórios (com fallback para nomes alternativos)
            chunk_id = meta.get("chunk_id") or meta.get("cid")
            doc_id = meta.get("doc_id")
            text = meta.get("text") or (meta.get("body", {}) or {}).get("input")

            if not chunk_id or not strategy or not doc_id or not text:
                log(
                    "batch.retrieve.skip",
                    reason="missing_required_fields",
                    have={
                        "chunk_id": bool(chunk_id),
                        "strategy": bool(strategy),
                        "doc_id": bool(doc_id),
                        "text": bool(text),
                    },
                )
                continue

            row = (
                chunk_id,
                strategy,
                doc_id,
                meta.get("page_start"),
                meta.get("page_end"),
                meta.get("tok_est"),
                meta.get("char_len"),
                text,
                json.dumps(meta.get("meta") or {}, ensure_ascii=False),
                vec,
            )
            rows_by_table[table].append(row)

        # 5) Insere por tabela, usando (cursor, tabela, linhas) e commit
        total = 0
        for tabela, linhas in rows_by_table.items():
            if not linhas:
                continue
            bulk_insert_chunks(cur, tabela, linhas)  # <- assinatura correta
            conn.commit()
            total += len(linhas)
            log("batch.retrieve.inserted", table=tabela, rows=len(linhas))

        cur.close()
        conn.close()

        log("batch.retrieve.success", batch_id=batch_id, count=total)
        print(
            f"[bold green]Resultados do lote {batch_id} inseridos com sucesso! ({total} linhas)[/bold green]"
        )
        return True

    except Exception as e:
        log("batch.retrieve.fatal", batch_id=batch_id, error=str(e))
        print(
            f"[bold red]Erro ao obter ou inserir resultados para o lote {batch_id}: {e}[/bold red]"
        )
        return False

    finally:
        # limpa metadados locais
        try:
            metadata_file.unlink()
        except OSError as e:
            log("cleanup.error", file=str(metadata_file), error=str(e))


def start_process(args, config):
    """Orquestra o processo completo de criação, execução e monitorização de lotes."""
    output_dir = Path("out/batch")
    output_dir.mkdir(exist_ok=True, parents=True)
    processor = OpenAIBatchProcessor()

    log(
        "batch.process.start",
        message="A dividir o manifesto em lotes baseados no limite de tokens...",
    )
    token_limit = config.get("batch_processing", {}).get(
        "token_limit_per_batch", 2000000)

    batch_files = split_manifest_into_batches(config, output_dir, token_limit)

    if not batch_files:
        log("batch.process.end", message="Nenhum chunk novo para processar.")
        print("\n[yellow]Nenhum chunk novo encontrado para processar.[/yellow]")
        return

    log("batch.process.prepare_done", num_batches=len(batch_files))
    print(f"\n[green]Manifesto dividido em {len(batch_files)} lote(s).[/green]")

    active_batches = {}
    pending_batches = list(batch_files)

    # [CORREÇÃO] Define um limite de concorrência para evitar sobrecarregar a API
    MAX_CONCURRENT_BATCHES = 1

    try:
        while pending_batches or active_batches:
            # Inicia novos lotes apenas se houver espaço na fila de concorrência
            while pending_batches and len(active_batches) < MAX_CONCURRENT_BATCHES:
                next_batch = pending_batches.pop(0)
                input_file = next_batch["input_file"]
                metadata_file = next_batch["metadata_file"]

                log("batch.process.uploading", file=str(input_file))
                print(f"\nA enviar o ficheiro do lote: {input_file.name}...")
                file_id = processor.upload_file(input_file)

                log("batch.process.creating_batch", file_id=file_id)
                batch_id = processor.create_batch(
                    file_id, config["models"]["openai"]["embedding_model"]
                )
                print(
                    f"Lote [bold cyan]{batch_id}[/bold cyan] criado e em processamento."
                )
                active_batches[batch_id] = {
                    "metadata_file": metadata_file,
                    "status": "in_progress",
                }

            if not active_batches:
                break

            log("batch.process.monitoring", active_count=len(active_batches))
            print(
                f"\n({time.strftime('%H:%M:%S')}) A monitorizar {len(active_batches)} lote(s) ativo(s)... A aguardar 60 segundos."
            )
            time.sleep(60)

            completed_batches = []
            for batch_id, data in active_batches.items():
                status_info = processor.check_status(batch_id)
                status = status_info.get("status")
                log("batch.process.status_check", batch_id=batch_id, status=status)
                print(f" - Lote [cyan]{batch_id}[/cyan]: {status}")

                if status == "completed":
                    print(
                        f"   [green]Lote concluído! A obter e a inserir resultados...[/green]"
                    )
                    success = retrieve_and_insert(
                        batch_id,
                        data["metadata_file"],
                        processor,
                        config,
                    )
                    if success:
                        print(
                            f"   [green]Resultados do lote {batch_id} inseridos na base de dados com sucesso.[/green]"
                        )
                    else:
                        print(
                            f"   [red]Falha ao obter ou inserir resultados para o lote {batch_id}.[/red]"
                        )
                    completed_batches.append(batch_id)
                elif status in ["failed", "expired", "cancelled"]:
                    log("batch.process.failed", batch_id=batch_id, status=status)
                    print(
                        f"   [bold red]ERRO: O lote {batch_id} falhou com o estado: {status}[/bold red]"
                    )
                    completed_batches.append(batch_id)

            for batch_id in completed_batches:
                del active_batches[batch_id]

    except KeyboardInterrupt:
        print("\nProcesso interrompido pelo utilizador.")

    log("batch.process.end", message="Processo de lote concluído.")
    print("\n[bold green]Processo de todos os lotes concluído![/bold green]")


def main():
    """Função principal para gerir os argumentos da linha de comando."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Orquestrador para o Batch API da OpenAI."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Comando a executar"
    )

    parser_start = subparsers.add_parser(
        "start",
        help="Inicia o processo completo de lote (prepara, executa, monitoriza e obtém).",
    )
    parser_start.set_defaults(func=start_process)

    args = parser.parse_args()
    config = load_config("config/config.yml")

    args.func(args, config)


if __name__ == "__main__":
    main()
