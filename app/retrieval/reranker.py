"""
Módulo de Reordenação (Reranking).

Responsável por reordenar os fragmentos recuperados utilizando modelos de reranking
para aumentar a precisão dos contextos enviados ao LLM.
"""


class ContextReranker:
    """Reordenador de contexto (futura implementação)."""

    def rerank(self, query: str, retrieved_documents: list) -> list:
        """Reordena a lista de documentos recuperados para otimizar a relevância."""
        pass
