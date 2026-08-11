"""
Módulo do Recuperador Semântico (Retriever).

Responsável por coordenar a busca de fragmentos relevantes no banco vetorial
a partir de perguntas do usuário sobre a saúde suplementar.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.retrieval.vectorstore import VectorStoreManager


class SemanticRetriever:
    """Recuperador semântico de documentos para o ANS-Assistant."""

    def __init__(
        self,
        vectorstore_manager: Optional[VectorStoreManager] = None,
    ) -> None:
        """
        Inicializa o recuperador semântico.

        Args:
            vectorstore_manager: Instância do gerenciador do banco vetorial.
        """
        self.vectorstore = vectorstore_manager or VectorStoreManager()
        self._ensure_index_loaded()

    def _ensure_index_loaded(self) -> None:
        """Garante que o índice FAISS esteja carregado ou seja construído a partir dos chunks salvos."""
        if self.vectorstore.index is not None and len(self.vectorstore.payloads) > 0:
            return

        # Tentar carregar índice do disco
        if self.vectorstore.load():
            return

        # Se o índice não existir no disco, construir a partir do arquivo de chunks ingeridos
        chunks_file = settings.metadata_dir / "RN_566_2022_chunks.json"
        if not chunks_file.is_file():
            json_files = list(settings.metadata_dir.glob("*_chunks.json"))
            if json_files:
                chunks_file = json_files[0]

        if chunks_file.is_file():
            with open(chunks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = data.get("chunks", [])
            if chunks:
                self.vectorstore.add_chunks(chunks)
                self.vectorstore.save()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera os chunks mais relevantes para a consulta do usuário.

        Args:
            query: Pergunta do usuário em linguagem natural.
            top_k: Número de fragmentos relevantes a retornar.

        Returns:
            Lista de dicionários com 'text', 'metadata' e 'score'.
        """
        if not query or not query.strip():
            return []

        self._ensure_index_loaded()
        return self.vectorstore.search_similarity(query=query, top_k=top_k)

