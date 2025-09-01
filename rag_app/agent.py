# rag_app/agent.py - Módulo para o "Agente Planejador de Consultas"

import os
from typing import Dict, Any, Optional, List
from openai import OpenAI
from .utils import log
import json


class QueryPlanner:
    """
    Usa um LLM para analisar a pergunta do usuário e dividi-la em uma
    consulta semântica e filtros de metadados estruturados.
    """

    def __init__(self, model: str = "gpt-4-turbo"):
        try:
            self.client = (
                OpenAI()
            )  # Pega a chave da variável de ambiente OPENAI_API_KEY
        except Exception as e:
            raise RuntimeError(
                "Erro ao inicializar cliente OpenAI. Verifique sua chave de API."
            ) from e

        self.model = model
        self.system_prompt = "Você é um especialista em analisar perguntas e extrair parâmetros para busca em um banco de dados de compras públicas."

    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analisa a consulta do usuário e retorna a consulta semântica e os filtros.

        Returns:
            Um dicionário como:
            {
                "semantic_query": "preço médio de cadeiras de escritório",
                "filters": {
                    "orgao_nome": "Ministério da Educação",
                    "ano": 2024
                }
            }
        """
        log("query_planner.start", query=user_query)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query},
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "search_documents",
                            "description": "Executa uma busca em documentos de compras públicas.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "semantic_query": {
                                        "type": "string",
                                        "description": "A parte da pergunta do usuário que requer busca por significado. Reformule para ser uma boa consulta de busca vetorial.",
                                    },
                                    "orgao_nome": {
                                        "type": "string",
                                        "description": "O nome exato da entidade ou órgão público mencionado.",
                                    },
                                    "cnpj": {
                                        "type": "string",
                                        "description": "O CNPJ do órgão, se mencionado.",
                                    },
                                    "estado_sigla": {
                                        "type": "string",
                                        "description": "A sigla do estado (ex: SP, RJ, BA), se mencionada.",
                                    },
                                    "municipio_nome": {
                                        "type": "string",
                                        "description": "O nome do município, se mencionado.",
                                    },
                                    "ano": {
                                        "type": "integer",
                                        "description": "O ano da compra, se mencionado.",
                                    },
                                },
                                "required": ["semantic_query"],
                            },
                        },
                    }
                ],
                tool_choice={
                    "type": "function",
                    "function": {"name": "search_documents"},
                },
            )

            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)

            semantic_query = args.pop("semantic_query", user_query)
            filters = {k: v for k, v in args.items() if v}  # Remove filtros vazios

            log("query_planner.done", semantic_query=semantic_query, filters=filters)
            return {"semantic_query": semantic_query, "filters": filters}

        except Exception as e:
            log("query_planner.error", error=str(e))
            # Fallback: se o LLM falhar, faz uma busca normal sem filtros
            return {"semantic_query": user_query, "filters": {}}
