"""
Suíte de testes automatizados para os módulos de Recuperação Semântica (Etapa 3).
"""

from pathlib import Path
import pytest

from app.retrieval.embeddings import EmbeddingGenerator
from app.retrieval.vectorstore import VectorStoreManager
from app.retrieval.retriever import SemanticRetriever


@pytest.fixture
def sample_chunks():
    """Retorna lista de chunks sintéticos para teste de indexação e recuperação."""
    return [
        {
            "text": "Art. 3º A operadora deverá garantir o atendimento nos seguintes prazos máximos: consulta médica em pediatria até 7 dias úteis.",
            "metadata": {
                "document_id": "RN-566-2022",
                "norm_number": "566/2022",
                "title": "Resolução Normativa RN nº 566/2022",
                "thematic_domain": "Garantia de Atendimento",
                "source_type": "Resolução Normativa",
                "document_type": "Resolução Normativa",
                "official_url": "https://www.ans.gov.br/legislacao",
                "page_number": 2,
                "chunk_index": 1,
            },
        },
        {
            "text": "Art. 4º Na hipótese de indisponibilidade de prestador integrante da rede assistencial no município, a operadora deverá garantir o atendimento em município limítrofe.",
            "metadata": {
                "document_id": "RN-566-2022",
                "norm_number": "566/2022",
                "title": "Resolução Normativa RN nº 566/2022",
                "thematic_domain": "Garantia de Atendimento",
                "source_type": "Resolução Normativa",
                "document_type": "Resolução Normativa",
                "official_url": "https://www.ans.gov.br/legislacao",
                "page_number": 3,
                "chunk_index": 2,
            },
        },
    ]


def test_embedding_generator_dimensao_e_retorno():
    """Valida a geração de vetores de embedding locais com SentenceTransformers."""
    generator = EmbeddingGenerator()
    embedding = generator.generate_embedding("Qual o prazo de atendimento?")

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert isinstance(embedding[0], float)

    batch_embeddings = generator.generate_embeddings(["Texto um", "Texto dois"])
    assert isinstance(batch_embeddings, list)
    assert len(batch_embeddings) == 2


def test_vectorstore_construcao_e_persistencia(tmp_path: Path, sample_chunks):
    """Valida criação, indexação, busca e persistência do índice FAISS em disco."""
    generator = EmbeddingGenerator()
    manager = VectorStoreManager(embedding_generator=generator, store_dir=tmp_path)

    # Adicionar chunks e verificar indexação em memória
    manager.add_chunks(sample_chunks)
    assert manager.index is not None
    assert manager.index.ntotal == 2

    # Salvar índice em diretório temporário
    manager.save(tmp_path)
    assert (tmp_path / "index.faiss").is_file()
    assert (tmp_path / "payloads.json").is_file()

    # Carregar índice em novo manager
    new_manager = VectorStoreManager(embedding_generator=generator, store_dir=tmp_path)
    loaded = new_manager.load(tmp_path)
    assert loaded is True
    assert new_manager.index.ntotal == 2
    assert len(new_manager.payloads) == 2


def test_vectorstore_busca_semantica_e_metadados(tmp_path: Path, sample_chunks):
    """Valida a busca por similaridade e integridade dos metadados retornados."""
    generator = EmbeddingGenerator()
    manager = VectorStoreManager(embedding_generator=generator, store_dir=tmp_path)
    manager.add_chunks(sample_chunks)

    results = manager.search_similarity(query="prazo máximo consulta médica pediatria", top_k=2)

    assert len(results) > 0
    top_result = results[0]

    assert "text" in top_result
    assert "metadata" in top_result
    assert "score" in top_result
    assert top_result["score"] > 0.0

    meta = top_result["metadata"]
    assert meta["document_id"] == "RN-566-2022"
    assert meta["page_number"] == 2
    assert meta["chunk_index"] == 1
    assert "pediatria" in top_result["text"]


def test_semantic_retriever_interface(tmp_path: Path, sample_chunks):
    """Valida a interface do SemanticRetriever."""
    generator = EmbeddingGenerator()
    manager = VectorStoreManager(embedding_generator=generator, store_dir=tmp_path)
    manager.add_chunks(sample_chunks)

    retriever = SemanticRetriever(vectorstore_manager=manager)
    results = retriever.retrieve("indisponibilidade de rede de prestadores", top_k=1)

    assert len(results) == 1
    assert "indisponibilidade" in results[0]["text"]
    assert results[0]["metadata"]["page_number"] == 3
