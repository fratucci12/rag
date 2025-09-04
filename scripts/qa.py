# scripts/qa.py - CLI simples para perguntas e respostas com síntese e citações

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Carrega .env do projeto ANTES dos imports do app
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# path para importar rag_app
sys.path.append(str(PROJECT_ROOT))

from rag_app.utils import load_config, log
from rag_app.agent import QueryPlanner
from rag_app.retrieval import retrieve_contexts
from rag_app.answer import select_diverse_contexts, synthesize_answer


def main():
    # Reforço caso seja executado fora do diretório do projeto
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Perguntas e Respostas (RAG) com síntese e citações")
    parser.add_argument("question", nargs="?", help="Pergunta em linguagem natural")
    parser.add_argument(
        "--config", default="config/config.yml", help="Caminho do config YAML"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Top-K por estratégia para formar contexto"
    )
    parser.add_argument(
        "--quota", type=int, default=2, help="Máximo de trechos por documento"
    )
    parser.add_argument(
        "--model", default=None, help="Modelo LLM para síntese (override)"
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        sys.exit(1)

    question = args.question
    if not question:
        try:
            question = input("Digite sua pergunta: ")
        except KeyboardInterrupt:
            print("\nA sair.")
            return
    if not question:
        return

    planner_model = config.get("agent", {}).get("planner_model", "gpt-4-turbo")
    qp = QueryPlanner(model=planner_model)
    plan = qp.analyze_query(question)

    contexts = retrieve_contexts(config, plan, per_strategy_top_k=args.top_k)
    diverse = select_diverse_contexts(contexts, per_doc_quota=args.quota)

    synth_model = args.model or config.get("agent", {}).get("synthesizer_model", "gpt-4o-mini")
    result = synthesize_answer(question, diverse, model=synth_model)

    print("\n=== RESPOSTA ===\n")
    print(result.get("answer", ""))
    if result.get("citations"):
        print("\n=== FONTES (doc_id) ===")
        for c in result["citations"]:
            print(f"- {c.get('doc_id')}")


if __name__ == "__main__":
    main()
