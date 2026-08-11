"""
Módulo de Pipeline RAG (Retrieval-Augmented Generation).

Responsável por integrar a recuperação semântica, o construtor de prompt e o LLM
(via LangChain e ChatGoogleGenerativeAI) para gerar respostas fundamentadas
com citação de fontes oficiais e tratamento resiliente de erros com retry exponential backoff.
"""

import os
import time
from typing import Any, Dict, List, Optional

from app.config import settings
from app.generation.prompts import FALLBACK_INSUFFICIENT_CONTEXT, RAGPromptBuilder
from app.retrieval.retriever import SemanticRetriever

# Mensagem amigável padrão para indisponibilidade temporária do serviço de IA
SERVICE_UNAVAILABLE_MESSAGE = (
    "O serviço de geração de respostas está temporariamente indisponível. "
    "Tente novamente em alguns instantes."
)


class RAGPipeline:
    """Pipeline RAG completo do ANS-Assistant utilizando LangChain e Gemini."""

    def __init__(
        self,
        retriever: Optional[SemanticRetriever] = None,
        prompt_builder: Optional[RAGPromptBuilder] = None,
        llm_provider: Optional[str] = None,
    ) -> None:
        self.retriever = retriever or SemanticRetriever()
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.llm_provider = (llm_provider or settings.llm_provider).lower()

    @staticmethod
    def _is_transient_error(e: Exception) -> bool:
        """Determina se uma exceção é um erro temporário elegível para retry."""
        err_msg = str(e).lower()

        # Erros HTTP 503 e 429 (Resource Exhausted / Rate Limit) são SEMPRE temporários
        if (
            "503" in err_msg
            or "429" in err_msg
            or "unavailable" in err_msg
            or "resource_exhausted" in err_msg
        ):
            return True

        # Exceções nativas de rede/timeout
        if isinstance(e, (TimeoutError, ConnectionError)):
            return True

        # Indicadores de erro permanente (não deve fazer retry)
        permanent_indicators = [
            "400",
            "invalid_argument",
            "401",
            "403",
            "unauthenticated",
            "404",
            "not_found",
        ]
        for perm in permanent_indicators:
            if perm in err_msg:
                return False

        transient_indicators = [
            "high demand",
            "overloaded",
            "quota",
            "rate limit",
            "timeout",
            "timed out",
            "connection",
            "serviceunavailable",
        ]
        for trans in transient_indicators:
            if trans in err_msg:
                return True

        return False


    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Invoca o LLM via LangChain (ChatGoogleGenerativeAI) com suporte a retry
        com exponential backoff para falhas temporárias (503, 429, timeout).

        Retorna um dicionário:
        {"text": str, "success": bool, "error": Optional[str]}
        """
        gemini_key = os.environ.get("GEMINI_API_KEY") or settings.gemini_api_key
        openai_key = os.environ.get("OPENAI_API_KEY") or settings.openai_api_key

        # 1. Provedor Padrão: Google Gemini via LangChain
        if self.llm_provider == "gemini":
            if not gemini_key:
                return {
                    "text": (
                        "[CONFIGURAÇÃO PENDENTE] A variável GEMINI_API_KEY não foi configurada. "
                        "Para obter respostas geradas por IA, adicione sua chave GEMINI_API_KEY no arquivo .env."
                    ),
                    "success": False,
                    "error": "GEMINI_API_KEY ausente",
                }

            max_attempts = 3
            backoff_delays = [2, 4]  # Delays entre tentativas (em segundos)

            for attempt in range(1, max_attempts + 1):
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI

                    llm = ChatGoogleGenerativeAI(
                        model=settings.llm_model_name,
                        google_api_key=gemini_key,
                        temperature=settings.llm_temperature,
                        max_retries=0,  # Desativado retry interno do LangChain para ter controle estrito
                    )
                    res = llm.invoke(prompt)
                    content = res.content

                    if isinstance(content, list):
                        text_parts = [
                            item.get("text", "") if isinstance(item, dict) else str(item)
                            for item in content
                        ]
                        extracted_text = "".join(text_parts).strip()
                    else:
                        extracted_text = str(content).strip()

                    return {"text": extracted_text, "success": True, "error": None}

                except Exception as e:
                    if self._is_transient_error(e):
                        if attempt < max_attempts:
                            delay = backoff_delays[attempt - 1]
                            time.sleep(delay)
                            continue
                        else:
                            return {
                                "text": SERVICE_UNAVAILABLE_MESSAGE,
                                "success": False,
                                "error": f"Tentativas esgotadas (503/429/Timeout): {e}",
                            }
                    else:
                        # Erro permanente
                        err_str = str(e)
                        if "401" in err_str or "403" in err_str or "API_KEY" in err_str.upper():
                            msg = "[ERRO DE CONFIGURAÇÃO] A GEMINI_API_KEY informada é inválida ou sem permissão."
                        elif "404" in err_str or "NOT_FOUND" in err_str.upper():
                            msg = f"[ERRO DE CONFIGURAÇÃO] O modelo '{settings.llm_model_name}' não foi encontrado."
                        else:
                            msg = f"[Erro de API do Gemini: {e}]"

                        return {"text": msg, "success": False, "error": str(e)}

            return {
                "text": SERVICE_UNAVAILABLE_MESSAGE,
                "success": False,
                "error": "Serviço indisponível após retries",
            }

        # 2. Provedor Opcional: OpenAI (Mantido como fallback alternativo)
        if self.llm_provider == "openai":
            if not openai_key:
                return {
                    "text": (
                        "[CONFIGURAÇÃO PENDENTE] A variável OPENAI_API_KEY não foi configurada. "
                        "Por favor, adicione sua chave OPENAI_API_KEY no arquivo .env."
                    ),
                    "success": False,
                    "error": "OPENAI_API_KEY ausente",
                }
            try:
                import openai

                client = openai.OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.llm_temperature,
                )
                return {
                    "text": response.choices[0].message.content.strip(),
                    "success": True,
                    "error": None,
                }
            except Exception as e:
                return {"text": f"[Erro na API da OpenAI: {e}]", "success": False, "error": str(e)}

        return {
            "text": f"[ERRO DE CONFIGURAÇÃO] Provedor LLM '{self.llm_provider}' desconhecido.",
            "success": False,
            "error": "Provedor desconhecido",
        }

    def extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extrai e consolida as fontes normativas dos chunks recuperados."""
        sources: List[Dict[str, Any]] = []
        seen_pages = set()

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            page = meta.get("page_number")
            norm_number = meta.get("norm_number", "566/2022")
            doc_id = meta.get("document_id", "RN-566-2022")
            title = meta.get("title", "RN nº 566/2022")
            official_url = meta.get("official_url", "")

            key = (doc_id, page)
            if key not in seen_pages:
                seen_pages.add(key)
                sources.append(
                    {
                        "document_id": doc_id,
                        "norm_number": norm_number,
                        "title": title,
                        "page_number": page,
                        "official_url": official_url,
                    }
                )

        return sources

    def generate_answer(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Executa o pipeline RAG completo:
        1. Recupera chunks via SemanticRetriever.
        2. Constrói o prompt RAG via RAGPromptBuilder.
        3. Chama o LLM via ChatGoogleGenerativeAI (LangChain) com suporte a retry.
        4. Retorna resposta estruturada e lista de fontes.
        """
        if not query or not query.strip():
            return {
                "query": query,
                "answer": "Por favor, digite uma pergunta válida sobre a regulamentação.",
                "sources": [],
                "chunks_count": 0,
                "success": False,
            }

        # 1. Recuperar chunks
        chunks = self.retriever.retrieve(query=query, top_k=top_k)

        # 2. Se nenhum chunk for encontrado
        if not chunks:
            return {
                "query": query,
                "answer": FALLBACK_INSUFFICIENT_CONTEXT,
                "sources": [],
                "chunks_count": 0,
                "success": True,
            }

        # 3. Construir prompt RAG
        prompt = self.prompt_builder.build_prompt(query=query, chunks=chunks)

        # 4. Chamar LLM com suporte a retry
        llm_result = self._call_llm(prompt)

        # 5. Consolidar fontes
        sources = self.extract_sources(chunks)

        return {
            "query": query,
            "answer": llm_result["text"],
            "sources": sources,
            "chunks_count": len(chunks),
            "raw_chunks": chunks,
            "success": llm_result["success"],
            "error": llm_result.get("error"),
        }


class RAGAnswerGenerator(RAGPipeline):
    """Alias mantido para retrocompatibilidade."""

    pass
