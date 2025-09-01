# Makefile para orquestrar o projeto de RAG

.PHONY: help install db-init index-realtime index-batch test-interactive test pre-commit-install pre-commit pre-commit-update

help:
	@echo "---------------------------------------------------------------------"
	@echo " Comandos Disponiveis para o Projeto RAG"
	@echo "---------------------------------------------------------------------"
	@echo ""
	@echo "Uso Geral:"
	@echo "  make install         -> Instala as dependencias do requirements.txt"
	@echo "  make db-init         -> Cria as tabelas e extensoes na base de dados"
	@echo "  make test            -> Executa a suite de testes (pytest)"
	@echo "  make pre-commit-install -> Instala e configura os hooks do pre-commit"
	@echo "  make pre-commit      -> Executa os hooks do pre-commit em todos os ficheiros"
	@echo "  make pre-commit-update -> Atualiza as versoes dos hooks do pre-commit"
	@echo ""
	@echo "Metodos de Indexacao:"
	@echo "  make index-realtime  -> (MAIS RAPIDO para poucos ficheiros) Processa e envia os embeddings um a um."
	@echo "  make index-batch     -> (MAIS BARATO para muitos ficheiros) Processa tudo de forma automatica e otimizada com o Batch API."
	@echo ""
	@echo "Testes:"
	@echo "  make test-interactive-> Testa a busca com uma pergunta interativa no terminal."
	@echo "---------------------------------------------------------------------"

# --- Comandos de Setup ---
install:
	@echo "--> A instalar dependencias..."
	pip install -r requirements.txt

test:
	@echo "--> A executar testes (pytest)..."
	python -c "import os, sys, subprocess; p='requirements-dev.txt'; sys.exit(0) if not os.path.exists(p) else sys.exit(subprocess.call([sys.executable,'-m','pip','install','-r',p]))"
	pytest -q

pre-commit-install:
	@echo "--> A instalar e configurando pre-commit..."
	pip install pre-commit
	pre-commit install

pre-commit:
	@echo "--> A executar pre-commit em todos os ficheiros..."
	pre-commit run --all-files

pre-commit-update:
	@echo "--> A atualizar os hooks do pre-commit..."
	pre-commit autoupdate

db-init:
	@echo "--> A inicializar o schema da base de dados..."
	python scripts/index_realtime.py --init-db-only

# --- Comandos de Indexacao ---
index-realtime:
	@echo "--> A iniciar indexacao em tempo real..."
	python scripts/index_realtime.py

index-batch:
	@echo "--> A iniciar o processo de indexacao em lote automatico..."
	python scripts/batch_processor.py start

# --- Comandos de Teste ---
test-interactive:
	@echo "--> A iniciar teste de recuperacao interativo..."
	python scripts/test_retrieval.py --interactive
