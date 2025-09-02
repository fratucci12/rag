RAG – Indexação e Recuperação com PostgreSQL + pgvector

Visão Geral
- Pipeline de indexação de documentos (PDFs avulsos ou dentro de ZIPs) em PostgreSQL com pgvector, com chunking configurável, embeddings via OpenAI (ou local) e buscas híbridas (vetorial + full‑text) com reranking.
- Testes de recuperação interativo e em lote, com avaliador LLM opcional e relatório em tabela (Rich) no terminal.

Requisitos
- Python 3.10+
- PostgreSQL 14+ com extensão pgvector instalada no cluster (o código também executa CREATE EXTENSION IF NOT EXISTS vector)
- Variáveis de ambiente:
  - `PG_DSN`: DSN de conexão com Postgres. Ex.: `postgresql://user:pass@localhost:5432/mydb` ou `dbname=... user=... password=... host=... port=...`
  - `OPENAI_API_KEY`: chave para uso do backend OpenAI, do agente/planner e do avaliador LLM (quando habilitados)
- Dependências Python: `pip install -r requirements.txt`

Instalação Rápida
- Instalar dependências: `make install`
- Instalar hooks do pre-commit (opcional): `make pre-commit-install`
- Rodar testes: `make test`

Configuração (config/config.yml)
- Principais blocos:
  - `database`:
    - `docs_table`: tabela de documentos (ex.: `documents`)
    - `chunks_prefix`: prefixo das tabelas de chunks por estratégia (ex.: `chunks` => `chunks_pages_1500x400`)
    - `index_type`: `ivfflat` ou `hnsw` para o índice vetorial
  - `indexing`:
    - `manifest_path`: NDJSON com as entradas a indexar
    - `limit_entries`: 0 para todas as linhas do manifesto (ou um número)
    - `reset_strategies`: true/false para truncar tabelas antes de reindexar
    - `embed_batch` (opcional): tamanho do micro‑lote de embedding no modo realtime
  - `models`:
    - `default_backend`: `openai` ou `local`
    - Modelos e dimensões do embedding conforme backend
  - `batch_processing` (modo batch de embeddings): limites e intervalos do Batch API
  - `strategies` (chunking): define as estratégias (páginas, parágrafos, recursivo, etc.) com parâmetros por estratégia
  - `retrieval_testing` (testes de recuperação):
    - `query_file`, `output_file`
    - `top_k` (default), `candidates` (para re‑ranking), `rrf_k` (para híbrida)
    - `judge_model`, `evaluate_with_llm`, `judge_threshold` (para marcar “Hit” e MRR na tabela)
    - `methods`: lista e ordem dos métodos executados (ex.: ["Similarity", "Hybrid", "Re-Ranking", "Hybrid+Re-Ranking"])
    - `hyde`: toggle e modelo padrão do HyDE global (aplicado a todas as estratégias)
    - `per_strategy`: overrides por nome da estratégia (ex.: `pages_1500x400`) para `top_k`, `candidates`, `rrf_k`, `methods` e `hyde.enabled/model`

Manifesto (NDJSON)
- Cada linha descreve um arquivo a indexar. Campos importantes: `file_path` (PDF ou ZIP contendo PDFs) e metadados opcionais (armazenados em `docs.meta`). Exemplo mínimo:
  {"file_path": "C:/caminho/para/arquivo.zip", "sha256": "...", "meta": {"compra": {"cnpj": "..."}}}
- Para ZIPs, apenas membros `.pdf` são extraídos e processados.

Fluxo de Indexação
- Preparar schema: `make db-init` (cria `docs`/chunks por estratégia e índices)
- Realtime (documento a documento): `make index-realtime`
  - Lê o `manifest_path`, extrai texto de PDFs (diretos ou de dentro de ZIPs), aplica chunking por estratégia e insere embeddings.
  - Deduplicação:
    - Documentos: `upsert` por `doc_id` (formado a partir do ZIP e membro) – o campo `sha256` é preenchido com o `sha256_doc` (hash do PDF) quando disponível.
    - Chunks: `chunk_id` determinístico inclui `doc_id`, estratégia, páginas, texto e nome do arquivo fonte (evita colisões entre PDFs diferentes com o mesmo texto).
- Batch (opcional): `make index-batch`
  - Usa scripts com o OpenAI Batch API para grandes volumes (ver `scripts/batch_processor.py`).

Testes de Recuperação
- Interativo: `make test-interactive`
  - Usa o agente `QueryPlanner` (OpenAI) para extrair consulta semântica e filtros.
- Em lote (com tabela Rich e JSONL):
  - `make test-batch QUERIES_FILE=tests/queries.json OUTPUT_FILE=data/retrieval_results.jsonl`
  - Métodos disponíveis (configuráveis via `retrieval_testing.methods`):
    - Similarity: busca vetorial
    - Hybrid: combinação RRF entre full‑text (Postgres) e vetorial
    - Re‑Ranking: vetorial seguido de reranker CrossEncoder
    - Hybrid+Re‑Ranking: híbrida (RRF) seguida de reranker CrossEncoder
  - HyDE (toggle): quando habilitado, o texto de embedding da query é uma resposta hipotética gerada por LLM (por estratégia através de `per_strategy.<nome>.hyde.enabled` ou global em `retrieval_testing.hyde.enabled`).
  - Avaliação com LLM: se `evaluate_with_llm` estiver on, o script pede uma nota 1–5 por contexto, calcula média e MRR@K e imprime a tabela final.

Variáveis de Ambiente e Execução
- Windows (PowerShell):
  - `setx PG_DSN "postgresql://user:pass@localhost:5432/mydb"`
  - `setx OPENAI_API_KEY "sk-..."`
- Linux/macOS:
  - `export PG_DSN=postgresql://user:pass@localhost:5432/mydb`
  - `export OPENAI_API_KEY=sk-...`
- Logs:
  - Texto legível por padrão. Para JSON: `LOG_JSON=1 make test-batch ...`
  - Desligar cores: `NO_COLOR=1 make test-batch ...`

Comandos Make Disponíveis
- `make install` – instala dependências
- `make test` – roda a suíte de testes
- `make pre-commit-install` / `make pre-commit` / `make pre-commit-update`
- `make db-init` – prepara schema e índices
- `make index-realtime` – indexação em tempo real
- `make index-batch` – indexação com OpenAI Batch API
- `make test-interactive` – teste interativo
- `make test-batch` – teste em lote a partir de um JSON

Estrutura do Projeto (principais arquivos)
- `config/config.yml` – configuração principal (DB, modelos, estratégias, retrieval)
- `data/manifest.ndjson` – manifesto de arquivos a indexar
- `rag_app/core.py` – lógica principal de indexação e utilitários de batch
- `rag_app/core_realtime.py` – indexação em tempo real
- `rag_app/chunking.py` – funções de chunking (páginas, parágrafos, recursivo, etc.)
- `rag_app/db.py` – schema, inserções e utilitários de BD
- `rag_app/backends.py` – backends de embedding (OpenAI, local)
- `rag_app/retrieval.py` – buscas (Similarity, Hybrid, Re‑Ranking, Hybrid+Re‑Ranking), HyDE e relatório
- `rag_app/agent.py` – planejador de consultas (QueryPlanner via OpenAI)
- `rag_app/utils.py` – logging, hashes, tokens, helpers
- `scripts/index_realtime.py` – driver de indexação em tempo real
- `scripts/test_retrieval.py` — driver dos testes de recuperação
- `scripts/qa.py` — CLI para perguntas e respostas com síntese e citações
- `tests/**` – suíte de testes

Manutenção e Dicas
- Harmonizar SHA dos documentos já carregados (se necessário):
  - `UPDATE documents SET sha256 = sha256_doc WHERE sha256_doc IS NOT NULL AND (sha256 IS DISTINCT FROM sha256_doc);`
  - Ajuste `documents` para o nome definido em `database.docs_table`.
- Performance/tuning:
  - Ajuste `indexing.embed_batch` para balancear latência/uso de API
  - `retrieval_testing.top_k`, `candidates` e `rrf_k` podem ser afinados globalmente ou por estratégia
- Dependências ausentes:
  - PyMuPDF: `pip install PyMuPDF`
  - sentence-transformers (backend local ou CrossEncoder): `pip install sentence-transformers`

Exemplos Rápidos
- Indexar a base (realtime) após configurar o manifesto e variáveis de ambiente:
  - `make db-init`
  - `make index-realtime`
- Testar recuperação em lote com HyDE global ligado (padrão) e JSON de consultas:
  - `make test-batch QUERIES_FILE=tests/queries.json OUTPUT_FILE=data/retrieval_results.jsonl`

Perguntas e Respostas (Síntese com Citações)
- Pré-requisitos: base indexada e variáveis `PG_DSN` e `OPENAI_API_KEY` definidas.
- Execução:
  - `python scripts/qa.py "Qual o volume total de cadeiras giratórias licitadas no último trimestre?"`
  - Opcional: `--top-k 5 --quota 2 --model gpt-4o-mini`
  - O script usa o planejador para extrair filtros, recupera contextos multi-estratégia com diversidade por documento e gera uma resposta final com citações `[doc_id:páginas]`.

Suporte
- Ajustes finos podem ser feitos diretamente em `config/config.yml`. Caso queira, habilite/disable HyDE por estratégia em `retrieval_testing.per_strategy.<nome>.hyde.enabled`.


Executar teste de recuperação em lote

- Pré-requisitos: definir `OPENAI_API_KEY` e `PG_DSN` no ambiente e garantir que a base já está indexada (via `make index-realtime` ou `make index-batch`).
- Com seu ficheiro de consultas (JSON), execute:

```
make test-batch QUERIES_FILE=tests/queries.json OUTPUT_FILE=data/retrieval_results.jsonl
```

Parâmetros
- `QUERIES_FILE`: caminho para um JSON com uma lista de objetos `{ "query_id": "...", "text": "..." }`.
- `OUTPUT_FILE`: caminho para salvar resultados JSONL detalhados.
