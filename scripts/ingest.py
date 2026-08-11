"""
Script standalone de Ingestão de Documentos.

Executa o pipeline completo de carregamento, limpeza, fragmentação e indexação
dos documentos oficiais da ANS.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Adicionar diretório raiz ao PYTHONPATH para permitir imports do pacote app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion.chunker import DocumentChunker
from app.ingestion.cleaner import TextCleaner
from app.ingestion.loaders import DocumentLoader


def load_catalog(catalog_path: Path) -> List[Dict[str, Any]]:
    """Carrega o catálogo de documentos do arquivo JSON."""
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catálogo de documentos não encontrado em: '{catalog_path.resolve()}'")
    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("documents", [])


def find_document_in_catalog(doc_id: str, catalog: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Busca um documento no catálogo por ID ou por norm_number."""
    doc_id_clean = doc_id.strip().upper().replace("_", "-")
    for doc in catalog:
        current_id = doc.get("id", "").upper().replace("_", "-")
        norm_number = doc.get("norm_number", "").upper().replace("_", "-")
        if doc_id_clean == current_id or doc_id_clean in norm_number:
            return doc
    return None


def resolve_pdf_path(doc_metadata: Dict[str, Any], documents_dir: Path) -> Path:
    """Tenta localizar o arquivo PDF correspondente ao documento no diretório."""
    doc_id = doc_metadata.get("id", "")
    norm_num = doc_metadata.get("norm_number", "").replace("/", "_")

    possible_names = [
        f"{doc_id}.pdf",
        f"{doc_id.replace('-', '_')}.pdf",
        f"{norm_num}.pdf",
        f"RN_{doc_id.replace('-', '_')}.pdf",
    ]

    for name in possible_names:
        candidate = documents_dir / name
        if candidate.is_file():
            return candidate

    return documents_dir / f"{doc_id.replace('-', '_')}.pdf"


def run_ingestion(doc_id_argument: Optional[str] = None) -> None:
    """Executa o fluxo de ingestão para o documento especificado."""
    print("=" * 70)
    print("      ANS-ASSISTANT — PIPELINE DE INGESTÃO DE DOCUMENTOS (ETAPA 1)")
    print("=" * 70)

    try:
        catalog = load_catalog(settings.catalog_file)
    except Exception as e:
        print(f"\n[ERRO CRÍTICO] Falha ao carregar o catálogo de documentos: {e}")
        sys.exit(1)

    if not doc_id_argument:
        print("\n[AVISO] Nenhum documento foi especificado para ingestão.")
        print("\nUso do comando:")
        print("  python scripts/ingest.py --document <ID_DO_DOCUMENTO>\n")
        print("Documentos disponíveis no catálogo:")
        for doc in catalog:
            print(f"  - [{doc.get('id')}] {doc.get('title')} ({doc.get('thematic_domain')})")
        print("\nExemplo:")
        print("  python scripts/ingest.py --document RN-566-2022")
        print("=" * 70)
        return

    doc_metadata = find_document_in_catalog(doc_id_argument, catalog)
    if not doc_metadata:
        print(f"\n[ERRO] Documento '{doc_id_argument}' não encontrado no catálogo 'documents.json'.")
        print("\nIDs disponíveis:")
        for doc in catalog:
            print(f"  - {doc.get('id')}")
        sys.exit(1)

    pdf_path = resolve_pdf_path(doc_metadata, settings.documents_dir)
    print(f"\n[1/4] Documento localizado no catálogo: {doc_metadata.get('title')}")
    print(f"[2/4] Verificando arquivo PDF: '{pdf_path.name}'...")

    if not pdf_path.is_file():
        print(f"\n[ERRO DE ARQUIVO] Arquivo PDF não encontrado em: '{pdf_path.resolve()}'")
        print(f"Por favor, insira o arquivo PDF do documento na pasta '{settings.documents_dir.resolve()}'")
        print(f"Nome sugerido para o arquivo: '{pdf_path.name}'")
        sys.exit(1)

    # 1. Carregar páginas do PDF
    loader = DocumentLoader()
    try:
        pages_data = loader.load_pdf(pdf_path)
    except Exception as e:
        print(f"\n[ERRO NO LOADER] Falha ao ler PDF '{pdf_path.name}': {e}")
        sys.exit(1)

    # 2. Limpar texto conservadoramente
    cleaner = TextCleaner()
    total_raw_chars = 0
    total_cleaned_chars = 0
    pages_without_text = 0

    for page in pages_data:
        raw_text = page["text"]
        total_raw_chars += len(raw_text)
        cleaned_text = cleaner.clean_text(raw_text)
        page["text"] = cleaned_text
        total_cleaned_chars += len(cleaned_text)

        if not page["has_text"]:
            pages_without_text += 1

    # 3. Gerar chunks com metadados completos
    chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = chunker.create_chunks(
        pages_data=pages_data,
        document_metadata=doc_metadata,
        source_file=pdf_path.name,
    )

    total_chunks = len(chunks)
    avg_chunk_size = (
        sum(len(c["text"]) for c in chunks) / total_chunks if total_chunks > 0 else 0.0
    )

    # 4. Persistência genérica dos chunks em arquivo JSON
    doc_id_slug = doc_metadata.get("id", "doc").replace("-", "_")
    output_filename = f"{doc_id_slug}_chunks.json"
    output_path = settings.metadata_dir / output_filename

    output_payload = {
        "document_id": doc_metadata.get("id", ""),
        "source_file": pdf_path.name,
        "chunk_count": total_chunks,
        "chunks": chunks,
    }

    settings.metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    # 5. Relatório no Terminal
    print("\n" + "=" * 70)
    print("                    RELATÓRIO DE INGESTÃO DO PDF")
    print("=" * 70)
    print(f" ID do Documento      : {doc_metadata.get('id')}")
    print(f" Título               : {doc_metadata.get('title')}")
    print(f" Domínio Temático     : {doc_metadata.get('thematic_domain')}")
    print(f" Arquivo PDF          : {pdf_path.name}")
    print(f" Caminho Completo     : {pdf_path.resolve()}")
    print("-" * 70)
    print(f" Total de Páginas     : {len(pages_data)}")
    print(f" Páginas Sem Texto    : {pages_without_text}")
    print(f" Caracteres (Bruto)   : {total_raw_chars}")
    print(f" Caracteres (Limpo)   : {total_cleaned_chars}")
    print(f" Chunks Gerados       : {total_chunks}")
    print(f" Tamanho Médio Chunk  : {avg_chunk_size:.1f} caracteres")
    print(f" Configuração Chunking: size={settings.chunk_size}, overlap={settings.chunk_overlap}")
    print(f" Arquivo de Saída     : {output_path.resolve()}")
    print("=" * 70)
    print(" STATUS: INGESTÃO E CHUNKING CONCLUÍDOS COM SUCESSO!")
    print("=" * 70 + "\n")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script de Ingestão de Documentos PDF para o ANS-Assistant."
    )
    parser.add_argument(
        "--document",
        "-d",
        type=str,
        help="ID ou número da norma do documento a ser ingerido (ex.: RN-566-2022).",
    )
    args = parser.parse_args()
    run_ingestion(args.document)


if __name__ == "__main__":
    main()

