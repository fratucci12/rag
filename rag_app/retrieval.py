# rag_app/retrieval.py - Lógica com filtragem de metadados

import os
import sys
import json
import re
from typing import List, Dict, Any, Optional

from .db import connect_pg
from .backends import LocalBackend, OpenAIBackend
from .utils import log
from .agent import QueryPlanner

from rich.console import Console
from rich.table import Table
from sentence_transformers.cross_encoder import CrossEncoder
from openai import OpenAI
from tqdm import tqdm


class LLMEvaluator:
    """Usa um LLM generativo para avaliar a relevância de um contexto para uma pergunta."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        try:
            self.client = OpenAI()
        except Exception as e:
            raise RuntimeError(
                "Erro ao inicializar cliente OpenAI. Verifique a sua chave de API."
            ) from e
        self.model = model
        self.system_prompt = """
        Você é um assistente especialista em avaliar a relevância de documentos.
        A sua tarefa é avaliar, numa escala de 1 a 5, quão útil o [CONTEXTO] fornecido é para responder à [PERGUNTA].

        A escala de notas é:
        1: Totalmente irrelevante.
        2: Pouco relevante.
        3: Relevante.
        4: Muito relevante.
        5: Extremamente relevante (resposta direta e completa).

        Responda APENAS com o número da sua nota, sem nenhum texto adicional.
        """

    def evaluate_chunk(self, query: str, context: str) -> Optional[int]:
        log("llm_judge.evaluate.start", model=self.model)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"[PERGUNTA]: {query}\n\n[CONTEXTO]: {context}",
                    },
                ],
                temperature=0,
                max_tokens=5,
            )
            score_text = response.choices[0].message.content.strip()
            score = int(re.search(r"\d+", score_text).group())
            log("llm_judge.evaluate.done", score=score)
            return score
        except Exception as e:
            log("llm_judge.evaluate.error", error=str(e))
            return None


def hyde_generate_text(query: str, model: str = "gpt-3.5-turbo") -> Optional[str]:
    """Gera um texto hipotético (HyDE) a partir da pergunta para melhorar a busca vetorial.
    Se houver falha, retorna None e o chamador pode fazer fallback para a própria query.
    """
    try:
        client = OpenAI()
        system = (
            "Gere um parágrafo conciso, informativo e factual que pareça uma resposta a "
            "uma pergunta de busca. Evite opiniões; foque em termos e entidades concretas."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log("hyde.generate.error", error=str(e))
        return None


def _build_where_clause(filters: Dict[str, Any]) -> (str, List[Any]):
    """Constrói a cláusula WHERE e os parâmetros para a consulta SQL a partir dos filtros."""
    if not filters:
        return "", []

    clauses = []
    params = []
    for key, value in filters.items():
        # Usa o operador @> do JSONB para checar se o metadado contém o par chave/valor
        clauses.append("meta @> %s")
        params.append(json.dumps({key: value}))

    return f"WHERE {' AND '.join(clauses)}", params


def basic_similarity_search(
    cur, query_embedding: list[float], table: str, top_k: int, filters: Dict
) -> List[Dict[str, Any]]:
    where_clause, params = _build_where_clause(filters)
    query_embedding_str = str(query_embedding)

    sql = f"""
        SELECT text, page_start, page_end, 1 - (embedding <=> %s::vector) AS similarity, chunk_id::text
        FROM {table}
        {where_clause}
        ORDER BY similarity DESC LIMIT %s
    """

    cur.execute(sql, [query_embedding_str] + params + [top_k])
    return [
        {
            "text": r[0],
            "page_start": r[1],
            "page_end": r[2],
            "score": float(r[3]),
            "chunk_id": r[4],
        }
        for r in cur.fetchall()
    ]


def hybrid_search_rrf(
    cur,
    query: str,
    query_embedding: list[float],
    table: str,
    top_k: int,
    filters: Dict,
    k: int = 60,
) -> List[Dict[str, Any]]:
    where_clause, params = _build_where_clause(filters)
    where_clause_fts = where_clause.replace("WHERE", "AND") if where_clause else ""

    # tenta via coluna ts/unaccent (mais rápido e robusto)
    try:
        fts_sql = f"""
            SELECT chunk_id::text FROM {table}
            WHERE ts @@ websearch_to_tsquery('portuguese', unaccent(%s)) {where_clause_fts}
            ORDER BY ts_rank(ts, websearch_to_tsquery('portuguese', unaccent(%s))) DESC
            LIMIT 20
        """
        cur.execute(fts_sql, [query, query] + params)
    except Exception:
        # fallback: computa tsvector on-the-fly sem unaccent
        fts_sql = f"""
            SELECT chunk_id::text FROM {table}
            WHERE to_tsvector('portuguese', text) @@ websearch_to_tsquery('portuguese', %s) {where_clause_fts}
            ORDER BY ts_rank(to_tsvector('portuguese', text), websearch_to_tsquery('portuguese', %s)) DESC
            LIMIT 20
        """
        cur.execute(fts_sql, [query, query] + params)
    keyword_results = [(row[0], i + 1) for i, row in enumerate(cur.fetchall())]

    vec_sql = f"SELECT chunk_id::text FROM {table} {where_clause} ORDER BY embedding <=> %s::vector LIMIT 20"
    cur.execute(vec_sql, params + [str(query_embedding)])
    vector_results = [(row[0], i + 1) for i, row in enumerate(cur.fetchall())]

    rrf_scores = {}
    for chunk_id, rank in keyword_results:
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank)
    for chunk_id, rank in vector_results:
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank)

    if not rrf_scores:
        return []

    sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    top_ids = [cid for cid, score in sorted_chunks[:top_k]]

    if not top_ids:
        return []

    cur.execute(
        f"SELECT text, page_start, page_end, chunk_id::text FROM {table} WHERE chunk_id::text = ANY(%s)",
        (top_ids,),
    )
    results_map = {
        row[3]: {
            "text": row[0],
            "page_start": row[1],
            "page_end": row[2],
            "chunk_id": row[3],
        }
        for row in cur.fetchall()
    }

    return [results_map[cid] for cid in top_ids if cid in results_map]


def rerank_with_cross_encoder(
    cur,
    query: str,
    query_embedding: list[float],
    table: str,
    reranker_model: str,
    top_k: int,
    filters: Dict,
    candidates: int = 25,
) -> List[Dict[str, Any]]:
    initial_results = basic_similarity_search(
        cur, query_embedding, table, candidates, filters
    )
    if not initial_results:
        return []

    MAX_QUERY_CHARS = 300
    MAX_DOC_CHARS = 1400

    truncated_query = query[:MAX_QUERY_CHARS]
    pairs = [
        [truncated_query, (res.get("text") or "")[:MAX_DOC_CHARS]]
        for res in initial_results
    ]

    cross_encoder = CrossEncoder(reranker_model)
    scores = cross_encoder.predict(pairs)

    for res, score in zip(initial_results, scores):
        res["score"] = float(score)

    return sorted(initial_results, key=lambda x: x["score"], reverse=True)[:top_k]


def hyde_search(
    cur,
    query: str,
    backend,
    table: str,
    top_k: int,
    filters: Dict,
    hyde_model: str,
) -> List[Dict[str, Any]]:
    """Aplica HyDE: gera um texto hipotético com LLM, embeda e busca por similaridade."""
    hyde_text = hyde_generate_text(query, model=hyde_model) or query
    try:
        emb = backend.embed([hyde_text])[0]
    except Exception as e:
        log("hyde.embed.error", error=str(e))
        return []
    return basic_similarity_search(cur, emb, table, top_k, filters)


def hybrid_then_rerank(
    cur,
    query: str,
    query_embedding: list[float],
    table: str,
    reranker_model: str,
    top_k: int,
    filters: Dict,
    *,
    candidates: int = 25,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Combina a busca Híbrida (RRF entre FTS e Vetorial) para obter candidatos
    e aplica re-ranking com CrossEncoder.
    """
    initial_results = hybrid_search_rrf(
        cur, query, query_embedding, table, candidates, filters, k=rrf_k
    )
    if not initial_results:
        return []

    MAX_QUERY_CHARS = 300
    MAX_DOC_CHARS = 1400

    truncated_query = query[:MAX_QUERY_CHARS]
    pairs = [
        [truncated_query, (res.get("text") or "")[:MAX_DOC_CHARS]]
        for res in initial_results
    ]

    cross_encoder = CrossEncoder(reranker_model)
    scores = cross_encoder.predict(pairs)

    for res, score in zip(initial_results, scores):
        res["score"] = float(score)

    return sorted(initial_results, key=lambda x: x["score"], reverse=True)[:top_k]


def display_batch_results_table(
    console: Console, all_results: List[Dict], judge_threshold: int = 4
):
    """Exibe os resultados compilados de TODAS as queries (usado no modo batch)."""
    console.rule("[bold green]RELATÓRIO FINAL COMPARATIVO DE ESTRATÉGIAS[/]")
    console.print("Resultados Compilados de Todas as Perguntas", justify="center")

    table = Table(show_header=True, header_style="bold cyan", min_width=120)
    table.add_column("Query ID", style="dim", width=12)
    table.add_column("Query", width=32)
    table.add_column("Estratégia (Chunking)", style="magenta", width=27)
    table.add_column("Método (Busca)", style="yellow", width=17)
    table.add_column("Hit (Top-1)", style="bold", justify="center")
    table.add_column("Score (Top-1)", justify="right")
    table.add_column("Avg Score (Top-K)", justify="right")
    table.add_column("MRR@K", justify="right")

    last_query_id = None
    for result_item in all_results:
        query_id = result_item["query_id"]
        if last_query_id is not None and query_id != last_query_id:
            table.add_section()

        for table_name, methods in result_item["results_by_strategy_table"].items():
            strategy_name = table_name.replace("chunks_", "")
            for method_name, results in methods.items():
                if not results:
                    table.add_row(
                        query_id,
                        result_item["query_text"][:30] + "...",
                        strategy_name,
                        method_name,
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                    )
                    continue

                top_1_hit = (
                    "[green]True[/]"
                    if results[0].get("llm_judge_score", 0) >= judge_threshold
                    else "[red]False[/]"
                )
                top_1_score = results[0].get("llm_judge_score", "N/A")
                valid_scores = [
                    r.get("llm_judge_score")
                    for r in results
                    if r.get("llm_judge_score") is not None
                ]
                avg_score_top_k = (
                    sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                )
                mrr = 0.0
                for i, res in enumerate(results):
                    if res.get("llm_judge_score", 0) >= judge_threshold:
                        mrr = 1.0 / (i + 1)
                        break

                table.add_row(
                    query_id,
                    result_item["query_text"][:30] + "...",
                    strategy_name,
                    method_name,
                    top_1_hit,
                    f"{top_1_score}",
                    f"{avg_score_top_k:.2f}",
                    f"{mrr:.2f}",
                )
        last_query_id = query_id

    console.print(table)


def run_tests(
    config: Dict[str, Any],
    queries_to_process: List[Dict],
    interactive_mode: bool,
    output_path_str: Optional[str] = None,
):
    dsn = os.getenv("PG_DSN")
    if not dsn:
        log(fatal="A variável de ambiente PG_DSN não está definida.")
        sys.exit(1)

    console = Console()
    model_cfg = config["models"]
    test_cfg = config["retrieval_testing"]
    backend_name = model_cfg["default_backend"]
    backend = (
        LocalBackend(model_cfg["local"]["embedding_model"])
        if backend_name == "local"
        else OpenAIBackend(model_cfg["openai"]["embedding_model"])
    )
    conn = connect_pg(dsn)

    evaluator = LLMEvaluator(model=test_cfg["judge_model"])
    planner = QueryPlanner(
        model=config.get("agent", {}).get("planner_model", "gpt-4-turbo")
    )

    reranker_model = test_cfg.get(
        "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    log("reranker.init", model=reranker_model)
    # Config HyDE (defaults); pode ser sobrescrita por estratégia
    hyde_defaults = test_cfg.get("hyde", {}) or {}
    hyde_model_default = (
        hyde_defaults.get("model")
        or test_cfg.get("hyde_model")
        or "gpt-3.5-turbo"
    )
    # Métodos padrão (sem HyDE como método separado)
    default_methods = ["Similarity", "Hybrid", "Re-Ranking", "Hybrid+Re-Ranking"]
    # Threshold global para considerar "hit"
    judge_threshold = int(test_cfg.get("judge_threshold", 4))

    strategy_tables = [
        f"{config['database']['chunks_prefix']}_{tag}" for tag in config["strategies"]
    ]

    all_results_for_all_queries = []

    output_fp = (
        open(output_path_str, "w", encoding="utf-8") if output_path_str else None
    )

    try:
        pbar = tqdm(queries_to_process, desc="A processar perguntas")
        for query_item in pbar:
            user_query = query_item["text"]
            query_id = query_item.get("query_id", "N/A")
            pbar.set_description(f"Query: {query_id}")

            plan = (
                planner.analyze_query(user_query)
                if not interactive_mode
                else query_item
            )
            semantic_query = plan["semantic_query"]
            filters = plan["filters"]

            results_for_all_strategies = {}

            for table_name in strategy_tables:
                # Config por estratégia (baseado no nome da tabela atual)
                strategy_name = table_name.replace(
                    f"{config['database']['chunks_prefix']}_", ""
                )
                per_strategy = (test_cfg.get("per_strategy", {}) or {}).get(
                    strategy_name, {}
                )
                methods_cfg = (
                    per_strategy.get("methods")
                    or test_cfg.get("methods")
                    or default_methods
                )
                top_k_cfg = int(
                    per_strategy.get("top_k", test_cfg.get("top_k", 5))
                )
                candidates_cfg = int(
                    per_strategy.get("candidates", test_cfg.get("candidates", 25))
                )
                rrf_k_cfg = int(
                    per_strategy.get("rrf_k", test_cfg.get("rrf_k", 60))
                )

                # HyDE por estratégia
                hyde_cfg = (per_strategy.get("hyde") or {})
                hyde_enabled = bool(
                    hyde_cfg.get("enabled", (hyde_defaults.get("enabled", False)))
                )
                hyde_model = hyde_cfg.get("model", hyde_model_default)

                # Gera embedding com ou sem HyDE
                semantic_for_embedding = (
                    hyde_generate_text(semantic_query, model=hyde_model)
                    if hyde_enabled
                    else semantic_query
                ) or semantic_query
                query_embedding = backend.embed([semantic_for_embedding])[0]
                with conn.cursor() as cur:
                    methods = {}
                    for name in methods_cfg:
                        try:
                            if name == "Similarity":
                                methods[name] = basic_similarity_search(
                                    cur,
                                    query_embedding,
                                    table_name,
                                    top_k_cfg,
                                    filters,
                                )
                            elif name == "Hybrid":
                                methods[name] = hybrid_search_rrf(
                                    cur,
                                    semantic_query,
                                    query_embedding,
                                    table_name,
                                    top_k_cfg,
                                    filters,
                                    k=rrf_k_cfg,
                                )
                            elif name == "Re-Ranking":
                                methods[name] = rerank_with_cross_encoder(
                                    cur,
                                    semantic_query,
                                    query_embedding,
                                    table_name,
                                    reranker_model,
                                    top_k_cfg,
                                    filters,
                                    candidates=candidates_cfg,
                                )
                            elif name == "Hybrid+Re-Ranking":
                                methods[name] = hybrid_then_rerank(
                                    cur,
                                    semantic_query,
                                    query_embedding,
                                    table_name,
                                    reranker_model,
                                    top_k_cfg,
                                    filters,
                                    candidates=candidates_cfg,
                                    rrf_k=rrf_k_cfg,
                                )
                            elif name == "HyDE":
                                # Compatibilidade retroativa: trata "HyDE" como Similarity com HyDE forçado
                                _emb_text = hyde_generate_text(
                                    semantic_query, model=hyde_model
                                ) or semantic_query
                                _emb = backend.embed([_emb_text])[0]
                                methods[name] = basic_similarity_search(
                                    cur, _emb, table_name, top_k_cfg, filters
                                )
                        except Exception as e:
                            # Log e recupera a conex3o de um poss77vel estado abortado
                            log(
                                "retrieval.method.error",
                                method=name,
                                table=table_name,
                                error=str(e),
                            )
                            try:
                                cur.connection.rollback()
                            except Exception:
                                pass

                    for method_name, results_list in methods.items():
                        for res in results_list:
                            score = evaluator.evaluate_chunk(
                                semantic_query, res["text"]
                            )
                            res["llm_judge_score"] = score

                    results_for_all_strategies[table_name] = methods

            final_result_item = {
                "query_id": query_id,
                "query_text": user_query,
                "semantic_query": semantic_query,
                "filters": filters,
                "results_by_strategy_table": results_for_all_strategies,
            }
            all_results_for_all_queries.append(final_result_item)

            if output_fp:
                output_fp.write(
                    json.dumps(final_result_item, ensure_ascii=False) + "\n"
                )

        if interactive_mode and all_results_for_all_queries:
            # No modo interativo, mostramos a tabela de apenas uma pergunta
            display_batch_results_table(
                console, all_results_for_all_queries, judge_threshold
            )
        elif not interactive_mode:
            # No modo batch, mostramos a tabela consolidada no final
            display_batch_results_table(
                console, all_results_for_all_queries, judge_threshold
            )

    finally:
        conn.close()
        if output_fp:
            output_fp.close()
            log(
                "run.finish",
                message=f"Resultados detalhados do teste salvos em: {output_path_str}",
            )


# --- Helpers adicionais para Q&A com síntese ---
def get_doc_ids_for_chunks(cur, table: str, chunk_ids: list[str]) -> dict[str, str]:
    """Resolve doc_id para uma lista de chunk_ids numa tabela específica."""
    if not chunk_ids:
        return {}
    cur.execute(
        f"SELECT chunk_id::text, doc_id FROM {table} WHERE chunk_id::text = ANY(%s)",
        (chunk_ids,),
    )
    return {row[0]: row[1] for row in cur.fetchall()}


def retrieve_contexts(
    config: Dict[str, Any],
    plan: Dict[str, Any],
    *,
    per_strategy_top_k: int = 5,
    method: str = "Hybrid+Re-Ranking",
) -> list[Dict[str, Any]]:
    """Executa a recuperação multi-estratégia e retorna contextos com doc_id e metadados.

    Retorna uma lista com itens: {text, page_start, page_end, chunk_id, doc_id, strategy, table, score}
    """
    dsn = os.getenv("PG_DSN")
    if not dsn:
        raise RuntimeError("PG_DSN não definido")

    model_cfg = config["models"]
    backend_name = model_cfg["default_backend"]
    backend = (
        LocalBackend(model_cfg["local"]["embedding_model"]) if backend_name == "local" else OpenAIBackend(model_cfg["openai"]["embedding_model"])
    )

    test_cfg = config.get("retrieval_testing", {}) or {}
    reranker_model = test_cfg.get(
        "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    hyde_defaults = test_cfg.get("hyde", {}) or {}

    strategy_tables = [
        (tag, f"{config['database']['chunks_prefix']}_{tag}") for tag in config["strategies"]
    ]

    semantic_query = plan.get("semantic_query") or plan.get("text") or ""
    filters = plan.get("filters") or {}

    out_contexts: list[Dict[str, Any]] = []
    conn = connect_pg(dsn)
    try:
        with conn.cursor() as cur:
            for tag, table_name in strategy_tables:
                per_strategy = (test_cfg.get("per_strategy", {}) or {}).get(tag, {})
                hyde_cfg = (per_strategy.get("hyde") or {})
                hyde_enabled = bool(
                    hyde_cfg.get("enabled", (hyde_defaults.get("enabled", False)))
                )
                hyde_model = hyde_cfg.get("model", hyde_defaults.get("model", "gpt-3.5-turbo"))

                semantic_for_embedding = (
                    hyde_generate_text(semantic_query, model=hyde_model)
                    if hyde_enabled
                    else semantic_query
                ) or semantic_query

                query_embedding = backend.embed([semantic_for_embedding])[0]

                # Executa método escolhido
                results: list[Dict[str, Any]] = []
                if method == "Similarity":
                    results = basic_similarity_search(
                        cur, query_embedding, table_name, per_strategy_top_k, filters
                    )
                elif method == "Hybrid":
                    results = hybrid_search_rrf(
                        cur,
                        semantic_query,
                        query_embedding,
                        table_name,
                        per_strategy_top_k,
                        filters,
                        k=int(test_cfg.get("rrf_k", 60)),
                    )
                elif method == "Re-Ranking":
                    results = rerank_with_cross_encoder(
                        cur,
                        semantic_query,
                        query_embedding,
                        table_name,
                        reranker_model,
                        per_strategy_top_k,
                        filters,
                        candidates=int(test_cfg.get("candidates", 25)),
                    )
                else:  # Hybrid+Re-Ranking
                    results = hybrid_then_rerank(
                        cur,
                        semantic_query,
                        query_embedding,
                        table_name,
                        reranker_model,
                        per_strategy_top_k,
                        filters,
                        candidates=int(test_cfg.get("candidates", 25)),
                        rrf_k=int(test_cfg.get("rrf_k", 60)),
                    )

                # anexa doc_id e metadados úteis
                chunk_ids = [r["chunk_id"] for r in results]
                doc_map = get_doc_ids_for_chunks(cur, table_name, chunk_ids)
                for r in results:
                    out_contexts.append(
                        {
                            **r,
                            "doc_id": doc_map.get(r["chunk_id"]),
                            "table": table_name,
                            "strategy": tag,
                        }
                    )
    finally:
        conn.close()

    return out_contexts
