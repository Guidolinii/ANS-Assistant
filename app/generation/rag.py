"""
Módulo de Pipeline RAG (Retrieval-Augmented Generation).

Responsável por integrar a recuperação semântica, o construtor de prompt e o LLM
para gerar respostas fundamentadas com citação de fontes oficiais.
"""

import os
from typing import Any, Dict, List, Optional

from app.config import settings
from app.generation.prompts import FALLBACK_INSUFFICIENT_CONTEXT, RAGPromptBuilder
from app.retrieval.retriever import SemanticRetriever


class RAGPipeline:
    """Pipeline RAG completo do ANS-Assistant."""

    def __init__(
        self,
        retriever: Optional[SemanticRetriever] = None,
        prompt_builder: Optional[RAGPromptBuilder] = None,
        llm_provider: Optional[str] = None,
    ) -> None:
        self.retriever = retriever or SemanticRetriever()
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.llm_provider = (llm_provider or settings.llm_provider).lower()

    def _call_llm(self, prompt: str) -> str:
        """Invoca o Gemini LLM configurado ou informa sobre a falta da GEMINI_API_KEY."""
        gemini_key = os.environ.get("GEMINI_API_KEY") or settings.gemini_api_key
        openai_key = os.environ.get("OPENAI_API_KEY") or settings.openai_api_key

        # 1. Provedor Padrão: Google Gemini
        if self.llm_provider == "gemini":
            if not gemini_key:
                return (
                    "[CONFIGURAÇÃO PENDENTE] A variável GEMINI_API_KEY não foi configurada. "
                    "Para obter respostas geradas por IA, adicione sua chave GEMINI_API_KEY no arquivo .env."
                )
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                response = client.models.generate_content(
                    model=settings.llm_model_name,
                    contents=prompt,
                )
                return response.text.strip()
            except Exception as e:
                return f"[Erro ao conectar à API do Gemini: {e}]"

        # 2. Provedor Opcional: OpenAI
        if self.llm_provider == "openai":
            if not openai_key:
                return (
                    "[CONFIGURAÇÃO PENDENTE] A variável OPENAI_API_KEY não foi configurada. "
                    "Por favor, adicione sua chave OPENAI_API_KEY no arquivo .env."
                )
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.llm_temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"[Erro ao conectar à API da OpenAI: {e}]"

        return f"[ERRO DE CONFIGURAÇÃO] Provedor LLM '{self.llm_provider}' desconhecido."


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
        Executa o pipeline RAG completo.

        1. Recupera os chunks relevantes via SemanticRetriever.
        2. Verifica se o contexto retornado é suficiente.
        3. Formata o prompt RAG.
        4. Invoca o LLM.
        5. Retorna a resposta estruturada com as fontes normativas citadas.
        """
        if not query or not query.strip():
            return {
                "query": query,
                "answer": "Por favor, digite uma pergunta válida sobre a regulamentação.",
                "sources": [],
                "chunks_count": 0,
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
            }

        # 3. Construir prompt
        prompt = self.prompt_builder.build_prompt(query=query, chunks=chunks)

        # 4. Chamar LLM
        answer_text = self._call_llm(prompt)

        # 5. Consolidar fontes
        sources = self.extract_sources(chunks)

        return {
            "query": query,
            "answer": answer_text,
            "sources": sources,
            "chunks_count": len(chunks),
            "raw_chunks": chunks,
        }


class RAGAnswerGenerator(RAGPipeline):
    """Alias mantido para retrocompatibilidade."""

    pass

