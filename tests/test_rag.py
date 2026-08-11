"""
Suíte de testes automatizados para os módulos de Geração RAG (Etapa 4).
"""

import pytest

from app.generation.prompts import (
    FALLBACK_INSUFFICIENT_CONTEXT,
    RAGPromptBuilder,
    SYSTEM_PROMPT,
)
from app.generation.rag import RAGPipeline


@pytest.fixture
def sample_chunks():
    return [
        {
            "text": "Art. 3º A operadora deverá garantir o atendimento em até 7 dias úteis para consulta básica de pediatria.",
            "metadata": {
                "document_id": "RN-566-2022",
                "norm_number": "566/2022",
                "title": "Resolução Normativa RN nº 566/2022",
                "page_number": 2,
                "chunk_index": 5,
                "official_url": "https://www.ans.gov.br/legislacao",
            },
        }
    ]


def test_rag_prompt_builder_format_context(sample_chunks):
    """Valida a formatação legível do bloco de contexto."""
    builder = RAGPromptBuilder()
    context_str = builder.format_context(sample_chunks)

    assert "RN-566-2022" in context_str
    assert "Página: 2" in context_str
    assert "consulta básica de pediatria" in context_str


def test_rag_prompt_builder_build_prompt(sample_chunks):
    """Valida que o prompt gerado inclui o sistema, contexto e a pergunta."""
    builder = RAGPromptBuilder()
    prompt = builder.build_prompt("qual o prazo de pediatria?", sample_chunks)

    assert SYSTEM_PROMPT in prompt
    assert "qual o prazo de pediatria?" in prompt
    assert "CONTEXTO REGULATÓRIO" in prompt


def test_rag_pipeline_extract_sources(sample_chunks):
    """Valida a extração e consolidação de fontes dos chunks."""
    pipeline = RAGPipeline()
    sources = pipeline.extract_sources(sample_chunks)

    assert len(sources) == 1
    assert sources[0]["document_id"] == "RN-566-2022"
    assert sources[0]["page_number"] == 2
    assert sources[0]["official_url"] == "https://www.ans.gov.br/legislacao"


def test_rag_pipeline_generate_answer_fallback_context_vazio():
    """Valida que se o retriever não encontrar chunks, o pipeline retorna o fallback de contexto insuficiente."""
    class EmptyRetriever:
        def retrieve(self, query: str, top_k: int = 5):
            return []

    pipeline = RAGPipeline(retriever=EmptyRetriever())
    result = pipeline.generate_answer("pergunta sem contexto")

    assert result["answer"] == FALLBACK_INSUFFICIENT_CONTEXT
    assert result["chunks_count"] == 0
    assert len(result["sources"]) == 0


def test_rag_pipeline_generate_answer_fluxo_completo(sample_chunks):
    """Valida a execução do fluxo RAG completo com retriever mockado."""
    class DummyRetriever:
        def retrieve(self, query: str, top_k: int = 5):
            return sample_chunks

    pipeline = RAGPipeline(retriever=DummyRetriever())
    result = pipeline.generate_answer("qual o prazo de pediatria?")

    assert "answer" in result
    assert result["chunks_count"] == 1
    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_id"] == "RN-566-2022"
