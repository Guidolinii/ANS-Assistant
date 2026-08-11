"""
Script CLI de Consulta e Busca Semântica.

Permite realizar buscas por similaridade de vetor na base de conhecimento da RN 566/2022
e exibir os resultados com trechos, páginas, metadados e score de similaridade.
"""

import argparse
import sys
from pathlib import Path

# Adicionar diretório raiz ao PYTHONPATH para permitir imports do pacote app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.retriever import SemanticRetriever


def run_search(query: str, top_k: int = 5) -> None:
    """Executa a busca semântica e exibe o relatório formatado no terminal."""
    if not query or not query.strip():
        print("[ERRO] Informe uma pergunta para realizar a busca semântica.")
        print("Exemplo: python scripts/search.py \"qual o prazo máximo para atendimento?\"")
        sys.exit(1)

    print("=" * 80)
    print("        ANS-ASSISTANT — BUSCA SEMÂNTICA (RETRIEVAL MVP)")
    print("=" * 80)
    print(f" PERGUNTA REALIZADA : \"{query.strip()}\"")
    print(f" TOP-K RESULTADOS   : {top_k}")
    print("-" * 80)

    try:
        retriever = SemanticRetriever()
        results = retriever.retrieve(query=query, top_k=top_k)
    except Exception as e:
        print(f"\n[ERRO CRÍTICO] Falha ao executar a busca semântica: {e}")
        sys.exit(1)

    if not results:
        print("\n[AVISO] Nenhum resultado relevante foi encontrado para esta consulta.")
        print("=" * 80)
        return

    print(f"\n[ENCONTRADOS {len(results)} CHUNKS RELEVANTES NA RN 566/2022]\n")

    for rank, item in enumerate(results, start=1):
        score = item.get("score", 0.0)
        metadata = item.get("metadata", {})
        text = item.get("text", "")

        doc_id = metadata.get("document_id", "N/A")
        norm_number = metadata.get("norm_number", "N/A")
        title = metadata.get("title", "N/A")
        page_number = metadata.get("page_number", "N/A")
        chunk_index = metadata.get("chunk_index", "N/A")
        official_url = metadata.get("official_url", "N/A")
        source_file = metadata.get("source_file", "N/A")

        print("-" * 80)
        print(f" RANK #{rank} | SCORE DE SIMILARIDADE: {score:.4f}")
        print("-" * 80)
        print(f"  • Documento de Origem : [{doc_id}] {title}")
        print(f"  • Norma / Arquivo     : {norm_number} | Arquivo: {source_file}")
        print(f"  • Posição no Documento: Página {page_number} (Chunk #{chunk_index})")
        print(f"  • Fonte Oficial URL   : {official_url}")
        print(f"\n  [TRECHO RECUPERADO]:")
        print(f"  \"{text}\"")
        print()

    print("=" * 80)
    print(" STATUS: CONSULTA SEMÂNTICA CONCLUÍDA COM SUCESSO!")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script de Busca Semântica na base normativo-regulatória do ANS-Assistant."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="Pergunta ou consulta em linguagem natural (ex.: 'qual o prazo máximo para atendimento?')",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Número de fragmentos mais relevantes a retornar (padrão: 5).",
    )
    args = parser.parse_args()

    if not args.query:
        print("[AVISO] Nenhuma consulta informada.")
        print("Uso: python scripts/search.py \"sua pergunta aqui\" [--top-k 5]\n")
        print("Exemplo de consulta:")
        print("  python scripts/search.py \"qual o prazo máximo para consulta médica de pediatria?\"")
        sys.exit(0)

    run_search(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
