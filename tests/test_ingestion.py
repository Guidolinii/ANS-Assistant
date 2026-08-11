"""
Suíte de testes automatizados para os módulos de Ingestão de Documentos (Etapa 1).
"""

from pathlib import Path
import pytest
import pypdf

from app.ingestion.loaders import DocumentLoader
from app.ingestion.cleaner import TextCleaner
from app.ingestion.chunker import DocumentChunker


@pytest.fixture
def dummy_pdf_path(tmp_path: Path) -> Path:
    """Cria um arquivo PDF válido sintético em diretório temporário para testes."""
    pdf_file = tmp_path / "test_document.pdf"
    writer = pypdf.PdfWriter()

    # Adicionar página em branco sintética
    writer.add_blank_page(width=612, height=792)
    with open(pdf_file, "wb") as f:
        writer.write(f)

    return pdf_file


def test_pdf_inexistente_raises_file_not_found():
    """Valida que o DocumentLoader lança FileNotFoundError se o PDF não existir."""
    loader = DocumentLoader()
    non_existent_path = Path("caminho/invalido/inexistente.pdf")

    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load_pdf(non_existent_path)

    assert "não encontrado" in str(exc_info.value)


def test_pdf_valido_carregamento(dummy_pdf_path: Path):
    """Valida a leitura de um arquivo PDF sintético existente."""
    loader = DocumentLoader()
    pages_data = loader.load_pdf(dummy_pdf_path)

    assert isinstance(pages_data, list)
    assert len(pages_data) == 1
    assert pages_data[0]["page_number"] == 1
    assert "text" in pages_data[0]
    assert "has_text" in pages_data[0]


def test_text_cleaner_limpeza_conservadora():
    """Valida que o TextCleaner limpa formatação sem remover texto jurídico."""
    cleaner = TextCleaner()
    raw_text = (
        "Art. 1  Esta  Resolução   Normativa - RN  dispõe sobre a garantia...\r\n\r\n"
        "\xa0\xa0\t§ 1 O atendimento deverá ser garantido.\n\n\n\n"
        "Inciso I - prazo de 7 dias.   "
    )

    cleaned = cleaner.clean_text(raw_text)

    assert "Art. 1 Esta Resolução Normativa - RN dispõe sobre a garantia..." in cleaned
    assert "§ 1 O atendimento deverá ser garantido." in cleaned
    assert "Inciso I - prazo de 7 dias." in cleaned
    assert "\n\n\n" not in cleaned


def test_text_cleaner_unicode_nfkc_ligatures():
    """Valida que o TextCleaner normaliza ligaduras de Unicode (ex.: ﬁ -> fi, ﬂ -> fl)."""
    cleaner = TextCleaner()
    text_with_ligatures = "Garantia dos bene\ufb01ciários na área geográ\ufb01ca e a\ufb02uência de atendimento certi\ufb01cada."

    cleaned = cleaner.clean_text(text_with_ligatures)

    assert "beneficiários" in cleaned
    assert "geográfica" in cleaned
    assert "afluência" in cleaned
    assert "certificada" in cleaned
    assert "\ufb01" not in cleaned


def test_document_chunker_criacao_e_metadados_completos():
    """Valida a divisão em chunks e a vinculação rigorosa de todos os metadados."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

    pages_data = [
        {
            "page_number": 1,
            "text": "Este é o primeiro parágrafo longo do documento regulatório da ANS sobre garantia de atendimento. " * 3,
            "has_text": True,
        },
    ]

    doc_metadata = {
        "id": "RN-566-2022",
        "title": "Resolução Normativa RN nº 566/2022",
        "document_type": "Resolução Normativa",
        "norm_number": "566/2022",
        "issuing_body": "ANS - Agência Nacional de Saúde Suplementar",
        "thematic_domain": "Garantia de Atendimento",
        "effective_date": "2022-12-30",
        "status": "Ativa",
        "official_url": "https://www.ans.gov.br/legislacao",
    }

    chunks = chunker.create_chunks(pages_data, doc_metadata, source_file="RN_566_2022.pdf")

    assert len(chunks) > 0

    first_chunk = chunks[0]
    meta = first_chunk["metadata"]

    assert meta["document_id"] == "RN-566-2022"
    assert meta["title"] == "Resolução Normativa RN nº 566/2022"
    assert meta["document_type"] == "Resolução Normativa"
    assert meta["norm_number"] == "566/2022"
    assert meta["issuing_body"] == "ANS - Agência Nacional de Saúde Suplementar"
    assert meta["thematic_domain"] == "Garantia de Atendimento"
    assert meta["effective_date"] == "2022-12-30"
    assert meta["status"] == "Ativa"
    assert meta["official_url"] == "https://www.ans.gov.br/legislacao"
    assert meta["source_file"] == "RN_566_2022.pdf"
    assert meta["page_number"] == 1
    assert meta["chunk_index"] == 1
