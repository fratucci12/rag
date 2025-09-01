# scripts/test_retrieval.py - Ponto de entrada para o teste de recuperação

import os
import sys
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

# Adiciona o diretório raiz ao path para permitir importações de 'rag_app'
sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_app.retrieval import run_tests
from rag_app.utils import load_config, log
from rag_app.agent import QueryPlanner


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Script para testar estratégias de recuperação."
    )
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Caminho para o ficheiro de configuração YAML.",
    )

    # Grupo de argumentos mutuamente exclusivos: ou é interativo, ou usa um ficheiro
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--interactive", action="store_true", help="Inicia em modo interativo."
    )
    query_group.add_argument(
        "--query-file",
        help="Caminho para o ficheiro JSON com as perguntas do teste em lote.",
    )

    parser.add_argument(
        "--output-file",
        help="[Apenas para modo batch] Caminho para o ficheiro JSONL onde os resultados detalhados serão salvos.",
    )
    args = parser.parse_args()

    log("run.start", script="test_retrieval", config_path=args.config)

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        sys.exit(1)

    if args.interactive:
        # --- MODO INTERATIVO ---
        planner = QueryPlanner(
            model=config.get("agent", {}).get("planner_model", "gpt-4-turbo")
        )
        try:
            user_query = input(
                "Digite a sua pergunta de teste (ex: preço de cadeiras no estado de SP): "
            )
            if not user_query:
                return

            plan = planner.analyze_query(user_query)
            run_tests(config, [plan], interactive_mode=True)
        except KeyboardInterrupt:
            print("\nA sair.")

    elif args.query_file:
        # --- MODO BATCH ---
        query_path = Path(args.query_file)
        if not query_path.exists():
            log(fatal="Ficheiro de perguntas não encontrado.", path=str(query_path))
            sys.exit(1)

        with query_path.open("r", encoding="utf-8") as f:
            queries_to_process = json.load(f)

        run_tests(
            config,
            queries_to_process,
            interactive_mode=False,
            output_path_str=args.output_file,
        )


if __name__ == "__main__":
    main()
