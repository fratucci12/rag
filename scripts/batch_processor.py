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
from rag_app.core import split_manifest_into_batches, initialize_database
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
        results = [json.loads(line) for line in results_content.strip().split("\n") if line.strip()]
        meta_text = metadata_file.read_text(encoding="utf-8", errors="replace")
        metadata = [json.loads(line) for line in meta_text.strip().split("\n") if line.strip()]

        if len(results) != len(metadata):
            log(
                "batch.retrieve.warn",
                reason="len_mismatch",
                results=len(results),
                meta=len(metadata),
            )

        # Indexa metadados por chave estável para casar com os resultados
        def _best_meta_key(m: dict):
            return m.get("id") or m.get("chunk_id") or m.get("cid")
        meta_by_key = {k: m for m in metadata if (k := _best_meta_key(m))}

        # 3) Conecta no Postgres via PG_DSN do .env
        dsn = _require_env("PG_DSN")
        if False and not dsn:
            raise RuntimeError(
                "Variável de ambiente PG_DSN não definida. Verifique seu .env."
            )
        conn = connect_pg(dsn)
        cur = conn.cursor()

        # chunks_prefix para compor nome da tabela, se necessário
        db_cfg = config.get("database", {})
        chunks_prefix = db_cfg.get("chunks_prefix", "chunks")

        # 4) Agrupa linhas por tabela

        rows_by_table = defaultdict(list)
        matched = 0

        for idx, result in enumerate(results):
            # Resolve metadado correspondente priorizando custom_id/id
            custom_id = result.get("custom_id") or result.get("id")
            meta = meta_by_key.get(custom_id)
            if meta is None:
                if idx < len(metadata):
                    meta = metadata[idx]
                    log("batch.retrieve.match_fallback", index=idx)
                else:
                    log("batch.retrieve.unmatched_result", custom_id=custom_id)
                    continue
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
            matched += 1

        if matched == 0:
            log("batch.retrieve.nothing_to_insert", batch_id=batch_id)
            print(f"[yellow]Nenhuma linha válida para inserir do lote {batch_id}.[/yellow]")
            return False

        # 5) Insere por tabela, usando (cursor, tabela, linhas) e commit
        total = 0
        for tabela, linhas in rows_by_table.items():
            if not linhas:
                continue
            bulk_insert_chunks(cur, tabela, linhas)  # <- assinatura correta
            conn.commit()
            total += len(linhas)
            log("batch.retrieve.inserted", table=tabela, rows=len(linhas))

        # conexões serão fechadas no finally

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
        # garante fechamento de recursos de DB
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def start_process(args, config):
    """Orquestra o processo completo de criação, execução e monitorização de lotes."""
    output_dir = Path("out/batch")
    output_dir.mkdir(exist_ok=True, parents=True)
    processor = OpenAIBatchProcessor()

    # diretório para armazenar metadados em caso de falha
    failed_dir = output_dir / "failed"
    failed_dir.mkdir(exist_ok=True, parents=True)

    # validação do modelo de embeddings
    embedding_model = config.get("models", {}).get("openai", {}).get("embedding_model")
    if not embedding_model:
        log("batch.config.error", missing="models.openai.embedding_model")
        print("[bold red]Config ausente: models.openai.embedding_model[/bold red]")
        return

    # Garante schema no Postgres (tabelas/índices) antes de ler o manifesto
    try:
        initialize_database(config)
    except SystemExit:
        return
    except Exception as e:
        log("batch.db.init.warn", error=str(e))

    log(
        "batch.process.start",
        message="A dividir o manifesto em lotes baseados no limite de tokens...",
    )
    bp_cfg = config.get("batch_processing", {}) or {}
    token_limit = bp_cfg.get("token_limit_per_batch", 2000000)
    poll_interval_seconds = bp_cfg.get("poll_interval_seconds", 60)
    create_cooldown_seconds = bp_cfg.get("create_cooldown_seconds", 30)
    enqueued_retry_backoff_seconds = bp_cfg.get("enqueued_retry_backoff_seconds", 90)

    batch_files = split_manifest_into_batches(config, output_dir, token_limit)

    if not batch_files:
        log("batch.process.end", message="Nenhum chunk novo para processar.")
        print("\n[yellow]Nenhum chunk novo encontrado para processar.[/yellow]")
        return

    log("batch.process.prepare_done", num_batches=len(batch_files))
    print(f"\n[green]Manifesto dividido em {len(batch_files)} lote(s).[/green]")

    active_batches = {}
    pending_batches = list(batch_files)
    enqueue_errors = defaultdict(int)

    # [CORREÇÃO] Define um limite de concorrência para evitar sobrecarregar a API
    MAX_CONCURRENT_BATCHES = 1
    last_create_ts = 0.0

    try:
        while pending_batches or active_batches:
            # Inicia novos lotes apenas se houver espaço na fila de concorrência
            while pending_batches and len(active_batches) < MAX_CONCURRENT_BATCHES:
                next_batch = pending_batches.pop(0)
                input_file = next_batch["input_file"]
                metadata_file = next_batch["metadata_file"]

                # Check remote active batches to avoid enqueued limit
                try:
                    remote_active = processor.list_active_batches()
                    if len(remote_active) >= 1:
                        print(
                            f"Ainda ha lote(s) ativos na API ({len(remote_active)}). Aguardando {poll_interval_seconds}s..."
                        )
                        # Re-enqueue and break to wait before attempting creation again
                        pending_batches.insert(0, {"input_file": input_file, "metadata_file": metadata_file})
                        break
                except Exception:
                    pass

                try:
                    # Respeita cooldown entre criações, se configurado
                    since_last = time.time() - last_create_ts
                    if since_last < create_cooldown_seconds:
                        wait = int(create_cooldown_seconds - since_last)
                        if wait > 0:
                            print(f"Aguardando {wait}s antes de criar próximo lote...")
                            time.sleep(wait)
                    log("batch.process.uploading", file=str(input_file))
                    print(f"\nA enviar o ficheiro do lote: {input_file.name}...")
                    file_id = processor.upload_file(input_file)

                    log("batch.process.creating_batch", file_id=file_id)
                    batch_id = processor.create_batch(file_id, embedding_model)
                    print(
                        f"Lote [bold cyan]{batch_id}[/bold cyan] criado e em processamento."
                    )
                    active_batches[batch_id] = {
                        "metadata_file": metadata_file,
                        "input_file": input_file,
                        "status": "in_progress",
                    }
                    last_create_ts = time.time()
                except Exception as e:
                    # controla tentativas de enfileirar
                    key = str(metadata_file)
                    enqueue_errors[key] += 1
                    attempts = enqueue_errors[key]
                    log("batch.process.enqueue_error", file=key, error=str(e), attempts=attempts)
                    print(
                        f"[yellow]Falha ao enfileirar lote: {e}. Tentativa {attempts}/3[/yellow]"
                    )
                    # Se atingir limite de 'enqueued', aplica backoff explícito
                    emsg = str(e).lower()
                    if "enqueued" in emsg and "limit" in emsg:
                        print(
                            f"[yellow]Limite de lotes enfileirados atingido. Aguardando {enqueued_retry_backoff_seconds}s antes de re-tentar...[/yellow]"
                        )
                        time.sleep(enqueued_retry_backoff_seconds)
                    if attempts < 3:
                        pending_batches.append(next_batch)
                    else:
                        try:
                            target = failed_dir / f"{metadata_file.stem}__enqueue_failed.jsonl"
                            if metadata_file.exists():
                                metadata_file.replace(target)
                            print(f"[red]Metadados movidos para falhas: {target.name}[/red]")
                        except Exception as me:
                            log("batch.process.move_failed_meta_error", file=str(metadata_file), error=str(me))
                    # continue to next pending
                    continue

            if not active_batches:
                continue

            log("batch.process.monitoring", active_count=len(active_batches))
            print(
                f"\n({time.strftime('%H:%M:%S')}) A monitorizar {len(active_batches)} lote(s) ativo(s)... A aguardar {poll_interval_seconds} segundos."
            )
            time.sleep(poll_interval_seconds)

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
                        # cleanup metadata on success
                        try:
                            data["metadata_file"].unlink()
                        except Exception as ce:
                            log("cleanup.error", file=str(data["metadata_file"]), error=str(ce))
                        # Cooldown para garantir que a API reconheça liberação antes da próxima criação
                        if create_cooldown_seconds > 0:
                            print(
                                f"Aguardando {create_cooldown_seconds}s antes de criar o próximo lote..."
                            )
                            time.sleep(create_cooldown_seconds)
                    else:
                        print(
                            f"   [red]Falha ao obter ou inserir resultados para o lote {batch_id}.[/red]"
                        )
                        # move metadata and input to failed folder on insert failure
                        try:
                            mf = data.get("metadata_file")
                            if mf:
                                target = Path("out/batch/failed") / f"{mf.stem}__insert_failed.jsonl"
                                if mf.exists():
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    mf.replace(target)
                                    log("batch.process.meta_moved_failed", file=str(target))
                        except Exception as me:
                            log("batch.process.move_failed_meta_error", file=str(data.get("metadata_file")), error=str(me))
                        try:
                            inf = data.get("input_file")
                            if inf:
                                infp = Path(inf)
                                target_in = Path("out/batch/failed") / f"{infp.stem}__insert_failed.jsonl"
                                if infp.exists():
                                    target_in.parent.mkdir(parents=True, exist_ok=True)
                                    infp.replace(target_in)
                                    log("batch.process.input_moved_failed", file=str(target_in))
                        except Exception as ie:
                            log("batch.process.move_failed_input_error", file=str(data.get("input_file")), error=str(ie))
                    completed_batches.append(batch_id)
                elif status in ["failed", "expired", "cancelled"]:
                    log("batch.process.failed", batch_id=batch_id, status=status)
                    print(
                        f"   [bold red]ERRO: O lote {batch_id} falhou com o estado: {status}[/bold red]"
                    )
                    # move metadata and input to failed folder when batch fails in API
                    try:
                        mf = data.get("metadata_file")
                        if mf:
                            target = Path("out/batch/failed") / f"{mf.stem}__batch_{status}.jsonl"
                            if mf.exists():
                                target.parent.mkdir(parents=True, exist_ok=True)
                                mf.replace(target)
                                log("batch.process.meta_moved_failed", file=str(target))
                    except Exception as me:
                        log("batch.process.move_failed_meta_error", file=str(data.get("metadata_file")), error=str(me))
                    try:
                        inf = data.get("input_file")
                        if inf:
                            infp = Path(inf)
                            target_in = Path("out/batch/failed") / f"{infp.stem}__batch_{status}.jsonl"
                            if infp.exists():
                                target_in.parent.mkdir(parents=True, exist_ok=True)
                                infp.replace(target_in)
                                log("batch.process.input_moved_failed", file=str(target_in))
                    except Exception as ie:
                        log("batch.process.move_failed_input_error", file=str(data.get("input_file")), error=str(ie))
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
