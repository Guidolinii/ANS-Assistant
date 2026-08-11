"""
Suíte de testes automatizados para os módulos de Geração RAG (Etapa 4).

Cobre:
- Formatação de prompt e contexto
- Extração de fontes
- Suporte a retries com exponential backoff para falhas 503/429/Timeout
- Ausência de retries para erros permanentes (401/400)
- Resposta de fallback quando o contexto é insuficiente
- Integração com ChatGoogleGenerativeAI (LangChain) via mocks
"""

from unittest.mock import MagicMock, patch
import pytest

from app.generation.prompts import (
    FALLBACK_INSUFFICIENT_CONTEXT,
    RAGPromptBuilder,
    SYSTEM_PROMPT,
)
from app.generation.rag import SERVICE_UNAVAILABLE_MESSAGE, RAGPipeline


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
        def retrieve(self, query: str, top_k: int = 10):
            return []

    pipeline = RAGPipeline(retriever=EmptyRetriever())
    result = pipeline.generate_answer("pergunta sem contexto")

    assert result["answer"] == FALLBACK_INSUFFICIENT_CONTEXT
    assert result["chunks_count"] == 0
    assert len(result["sources"]) == 0
    assert result["success"] is True


@patch("app.generation.rag.os.environ.get", return_value="fake_api_key")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_rag_pipeline_langchain_integration(mock_chat_cls, mock_env_get, sample_chunks):
    """Valida que o RAGPipeline utiliza ChatGoogleGenerativeAI do LangChain."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = MagicMock(
        content="Prazo de 7 dias úteis segundo Art. 3º."
    )
    mock_chat_cls.return_value = mock_llm_instance

    class DummyRetriever:
        def retrieve(self, query: str, top_k: int = 10):
            return sample_chunks

    pipeline = RAGPipeline(retriever=DummyRetriever())
    result = pipeline.generate_answer("qual o prazo de pediatria?")

    assert result["success"] is True
    assert "7 dias úteis" in result["answer"]
    mock_chat_cls.assert_called_once()
    mock_llm_instance.invoke.assert_called_once()


@patch("app.generation.rag.os.environ.get", return_value="fake_api_key")
@patch("time.sleep")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_rag_pipeline_retry_on_503_success(
    mock_chat_cls, mock_sleep, mock_env_get, sample_chunks
):
    """Valida que em erro 503 temporário, o pipeline realiza retry e tem sucesso na 2ª tentativa."""
    mock_llm_instance = MagicMock()
    # 1ª chamada lança 503 UNAVAILABLE, 2ª chamada sucede
    mock_llm_instance.invoke.side_effect = [
        Exception("503 UNAVAILABLE: Model is overloaded"),
        MagicMock(content="Resposta gerada após retry."),
    ]
    mock_chat_cls.return_value = mock_llm_instance

    class DummyRetriever:
        def retrieve(self, query: str, top_k: int = 10):
            return sample_chunks

    pipeline = RAGPipeline(retriever=DummyRetriever())
    result = pipeline.generate_answer("pergunta teste")

    assert result["success"] is True
    assert result["answer"] == "Resposta gerada após retry."
    assert mock_llm_instance.invoke.call_count == 2
    mock_sleep.assert_called_once_with(2)  # Primeiro delay do backoff


@patch("app.generation.rag.os.environ.get", return_value="fake_api_key")
@patch("time.sleep")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_rag_pipeline_retry_on_429_exhausted(
    mock_chat_cls, mock_sleep, mock_env_get, sample_chunks
):
    """Valida que em erro 429 persistente por 3 tentativas, esgota retries e retorna mensagem amigável."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")
    mock_chat_cls.return_value = mock_llm_instance

    class DummyRetriever:
        def retrieve(self, query: str, top_k: int = 10):
            return sample_chunks

    pipeline = RAGPipeline(retriever=DummyRetriever())
    result = pipeline.generate_answer("pergunta teste")

    assert result["success"] is False
    assert result["answer"] == SERVICE_UNAVAILABLE_MESSAGE
    assert mock_llm_instance.invoke.call_count == 3
    assert mock_sleep.call_count == 2


@patch("app.generation.rag.os.environ.get", return_value="fake_api_key")
@patch("time.sleep")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_rag_pipeline_no_retry_on_permanent_error(
    mock_chat_cls, mock_sleep, mock_env_get, sample_chunks
):
    """Valida que para erros permanentes (ex: 401 API_KEY_INVALID), NÃO é feito retry."""
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("401 API_KEY_INVALID: Invalid key provided")
    mock_chat_cls.return_value = mock_llm_instance

    class DummyRetriever:
        def retrieve(self, query: str, top_k: int = 10):
            return sample_chunks

    pipeline = RAGPipeline(retriever=DummyRetriever())
    result = pipeline.generate_answer("pergunta teste")

    assert result["success"] is False
    assert "[ERRO DE CONFIGURAÇÃO]" in result["answer"]
    assert mock_llm_instance.invoke.call_count == 1
    mock_sleep.assert_not_called()
