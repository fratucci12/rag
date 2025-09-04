# scripts/index_realtime.py - Ponto de entrada para a indexação em tempo real (documento a documento)

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Carrega .env do projeto ANTES dos imports do app
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Adiciona o diretório raiz ao path para permitir importações de 'rag_app'
sys.path.append(str(PROJECT_ROOT))

from rag_app.core_realtime import process_documents_realtime, initialize_database
from rag_app.utils import load_config, log


def main():
    """Função principal que orquestra a indexação em tempo real."""
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Script para indexar documentos em tempo real."
    )
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Caminho para o ficheiro de configuração YAML.",
    )
    parser.add_argument(
        "--init-db-only",
        action="store_true",
        help="Apenas inicializa o schema da base de dados e sai.",
    )
    args = parser.parse_args()

    log("run.start", script="index_realtime", config_path=args.config)

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        sys.exit(1)

    if args.init_db_only:
        initialize_database(config)
    else:
        process_documents_realtime(config)


if __name__ == "__main__":
    main()
