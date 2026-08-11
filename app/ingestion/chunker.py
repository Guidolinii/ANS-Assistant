"""
Módulo de Fragmentação (Chunking) e Metadados.

Responsável por dividir os textos higienizados em fragmentos menores (chunks)
e associar metadados estruturados (ex.: norma, órgão emissor, tema, página, seção).
"""

from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Divisor de documentos com suporte a metadados regulatórios."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        """
        Inicializa o fragmentador de texto.

        Args:
            chunk_size: Tamanho máximo do chunk em caracteres.
            chunk_overlap: Sobreposição entre chunks consecutivos.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )

    def create_chunks(
        self,
        pages_data: List[Dict[str, Any]],
        document_metadata: Dict[str, Any],
        source_file: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Divide o texto de cada página em chunks mantendo a página de origem e os metadados completos.

        Args:
            pages_data: Lista de dicionários por página (com 'page_number' e 'text').
            document_metadata: Dicionário contendo os metadados do documento oriundos do catálogo.
            source_file: Nome do arquivo PDF de origem (ex.: 'RN_566_2022.pdf').

        Returns:
            Lista de chunks contendo o texto e dicionário completo de metadados.
        """
        chunks: List[Dict[str, Any]] = []
        global_chunk_index = 0

        # Resolver source_file se fornecido no parâmetro ou no metadata
        resolved_source_file = source_file or document_metadata.get("source_file", "")

        for page in pages_data:
            page_text = page.get("text", "")
            page_number = page.get("page_number", 1)

            if not page_text or not page_text.strip():
                continue

            page_split_texts = self.splitter.split_text(page_text)

            for chunk_text in page_split_texts:
                global_chunk_index += 1
                chunk_metadata = {
                    "document_id": document_metadata.get("id", ""),
                    "title": document_metadata.get("title", ""),
                    "document_type": document_metadata.get("document_type", ""),
                    "norm_number": document_metadata.get("norm_number", ""),
                    "issuing_body": document_metadata.get("issuing_body", ""),
                    "thematic_domain": document_metadata.get("thematic_domain", ""),
                    "effective_date": document_metadata.get("effective_date", ""),
                    "status": document_metadata.get("status", ""),
                    "official_url": document_metadata.get("official_url", ""),
                    "source_file": resolved_source_file,
                    "page_number": page_number,
                    "chunk_index": global_chunk_index,
                    "source_type": document_metadata.get("source_type", document_metadata.get("document_type", "")),
                }

                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": chunk_metadata,
                    }
                )

        return chunks


