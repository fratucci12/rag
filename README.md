# rag

Executar teste de recuperação em lote

- Pré-requisitos: definir `OPENAI_API_KEY` e `PG_DSN` no ambiente e garantir que a base já está indexada (via `make index-realtime` ou `make index-batch`).
- Com seu ficheiro de consultas (JSON), execute:

```
make test-batch QUERIES_FILE=tests/queries.json OUTPUT_FILE=data/retrieval_results.jsonl
```

Parâmetros
- `QUERIES_FILE`: caminho para um JSON com uma lista de objetos `{ "query_id": "...", "text": "..." }`.
- `OUTPUT_FILE`: caminho para salvar resultados JSONL detalhados.
