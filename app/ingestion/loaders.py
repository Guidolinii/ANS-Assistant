"""
Módulo de Carregamento de Documentos.

Responsável por ler e extrair texto de diferentes formatos de arquivo
(PDF, DOCX, XLSX, PPTX, Markdown, CSV, JSON, HTML) contendo normas da ANS.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union
import pypdf

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Carregador genérico de documentos oficiais da ANS."""

    def load_pdf(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Carrega um arquivo PDF e extrai seu texto página a página.

        Args:
            file_path: Caminho para o arquivo PDF.

        Returns:
            Lista de dicionários contendo page_number (1-indexed), text e has_text.

        Raises:
            FileNotFoundError: Se o arquivo especificado não existir.
            ValueError: Se o arquivo não for um PDF válido.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo PDF não encontrado no caminho: '{path.resolve()}'")

        pages_data: List[Dict[str, Any]] = []

        try:
            reader = pypdf.PdfReader(str(path))
            total_pages = len(reader.pages)

            for index, page in enumerate(reader.pages):
                page_number = index + 1
                extracted_text = page.extract_text() or ""
                has_text = len(extracted_text.strip()) > 0

                if not has_text:
                    logger.warning(
                        f"[Diagnóstico Loader] A página {page_number}/{total_pages} do arquivo '{path.name}' "
                        "não possui texto extraível."
                    )

                pages_data.append(
                    {
                        "page_number": page_number,
                        "text": extracted_text,
                        "has_text": has_text,
                    }
                )

        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            raise ValueError(f"Erro ao ler arquivo PDF '{path.name}': {str(e)}") from e

        return pages_data

    def load_document(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Carrega um documento a partir do caminho fornecido."""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            return self.load_pdf(path)
        raise NotImplementedError(f"Suporte para arquivos '{path.suffix}' ainda não foi implementado.")

