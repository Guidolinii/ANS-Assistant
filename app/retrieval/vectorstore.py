"""
Módulo de Gerenciamento do Banco Vetorial.

Responsável pela criação, persistência, busca por similaridade e atualização
do índice de vetores utilizando FAISS (Facebook AI Similarity Search).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

from app.config import settings
from app.retrieval.embeddings import EmbeddingGenerator


class VectorStoreManager:
    """Gerenciador do banco vetorial FAISS local."""

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        store_dir: Optional[Path] = None,
    ) -> None:
        """
        Inicializa o gerenciador vetorial.

        Args:
            embedding_generator: Instância do gerador de embeddings (opcional).
            store_dir: Diretório para salvar/carregar o índice FAISS.
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.store_dir = Path(store_dir or settings.vectorstore_dir)
        self.index: Optional[faiss.Index] = None
        self.payloads: List[Dict[str, Any]] = []

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normaliza os vetores para norma L2 unitária para busca por cosseno."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Gera embeddings e adiciona uma lista de chunks com seus metadados ao índice FAISS.

        Args:
            chunks: Lista de dicionários contendo 'text' e 'metadata'.
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings_list = self.embedding_generator.generate_embeddings(texts)
        vectors = np.array(embeddings_list, dtype=np.float32)
        normalized_vectors = self._normalize(vectors)

        dimension = normalized_vectors.shape[1]

        # Criar índice de Produto Interno (IP) que equivale a Cosine Similarity após normalização L2
        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(normalized_vectors)
        self.payloads.extend(chunks)

    def save(self, target_dir: Optional[Path] = None) -> Path:
        """
        Persiste o índice FAISS e os payloads (metadados + texto) em disco.

        Returns:
            Caminho do diretório onde os arquivos foram salvos.
        """
        dir_path = Path(target_dir or self.store_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            index_path = dir_path / "index.faiss"
            faiss.write_index(self.index, str(index_path))

        payloads_path = dir_path / "payloads.json"
        with open(payloads_path, "w", encoding="utf-8") as f:
            json.dump(self.payloads, f, indent=2, ensure_ascii=False)

        return dir_path

    def load(self, source_dir: Optional[Path] = None) -> bool:
        """
        Carrega o índice FAISS e os payloads salvos em disco.

        Returns:
            True se carregado com sucesso, False caso os arquivos não existam.
        """
        dir_path = Path(source_dir or self.store_dir)
        index_path = dir_path / "index.faiss"
        payloads_path = dir_path / "payloads.json"

        if not index_path.is_file() or not payloads_path.is_file():
            return False

        self.index = faiss.read_index(str(index_path))
        with open(payloads_path, "r", encoding="utf-8") as f:
            self.payloads = json.load(f)

        return True

    def search_similarity(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Realiza busca vetorial por similaridade de cosseno a partir de uma consulta em texto.

        Args:
            query: Pergunta ou texto de busca.
            top_k: Quantidade de resultados mais relevantes a retornar.

        Returns:
            Lista de dicionários com 'text', 'metadata' e 'score'.
        """
        if self.index is None or not self.payloads:
            return []

        query_vector = self.embedding_generator.generate_embedding(query)
        vec_np = np.array([query_vector], dtype=np.float32)
        norm_query = self._normalize(vec_np)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(norm_query, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.payloads):
                continue
            chunk_copy = dict(self.payloads[idx])
            chunk_copy["score"] = float(score)
            results.append(chunk_copy)

        return results

    def add_documents(self, documents: list) -> None:
        """Método de compatibilidade."""
        self.add_chunks(documents)

