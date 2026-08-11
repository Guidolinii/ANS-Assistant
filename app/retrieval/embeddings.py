"""
Módulo de Geração de Embeddings.

Responsável por converter textos e fragmentos em vetores densos de representação semântica
utilizando modelos locais e gratuitos (SentenceTransformers).
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingGenerator:
    """Gerador de vetores de embedding com SentenceTransformers local."""

    def __init__(self, model_name: str = settings.embedding_model_name) -> None:
        """
        Inicializa o modelo local de embeddings.

        Args:
            model_name: Nome do modelo no HuggingFace/SentenceTransformers.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Gera o vetor de embedding para um único texto.

        Args:
            text: Texto de entrada.

        Returns:
            Lista de floats representando o vetor denso.
        """
        if not text:
            return []
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Gera os vetores de embedding para um texto ou lista de textos.

        Args:
            texts: Texto individual ou lista de textos.

        Returns:
            Lista de vetores densos.
        """
        if isinstance(texts, str):
            return [self.generate_embedding(texts)]

        if not texts:
            return []

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

