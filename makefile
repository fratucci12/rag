# Makefile para orquestrar o projeto de RAG

.PHONY: help install db-init index-realtime index-batch test-interactive

help:
	@echo "---------------------------------------------------------------------"
	@echo " Comandos Disponiveis para o Projeto RAG"
	@echo "---------------------------------------------------------------------"
	@echo ""
	@echo "Uso Geral:"
	@echo "  make install         -> Instala as dependencias do requirements.txt"
	@echo "  make db-init         -> Cria as tabelas e extensoes na base de dados"
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

db-init:
	@echo "--> A inicializar o schema da base de dados..."
	PYTHONPATH=. python scripts/index_realtime.py --init-db-only

# --- Comandos de Indexacao ---
index-realtime:
	@echo "--> A iniciar indexacao em tempo real..."
	PYTHONPATH=. python scripts/index_realtime.py

index-batch:
	@echo "--> A iniciar o processo de indexacao em lote automatico..."
	PYTHONPATH=. python scripts/batch_processor.py start

# --- Comandos de Teste ---
test-interactive:
	@echo "--> A iniciar teste de recuperacao interativo..."
	PYTHONPATH=. python scripts/test_retrieval.py --interactive