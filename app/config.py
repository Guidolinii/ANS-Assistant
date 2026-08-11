"""
Módulo de Configuração do ANS-Assistant.

Gerencia as configurações centrais da aplicação, caminhos de arquivos,
diretórios e parâmetros de fragmentação (chunking).
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações centrais da aplicação."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Diretórios e caminhos
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    documents_dir: Path = data_dir / "documents"
    metadata_dir: Path = data_dir / "metadata"
    catalog_file: Path = documents_dir / "documents.json"

    # Parâmetros de Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Configurações de Embeddings e Banco Vetorial
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vectorstore_dir: Path = metadata_dir / "vectorstore"


settings = Settings()



