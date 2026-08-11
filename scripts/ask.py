"""
Script CLI de Pergunta e Resposta (RAG).

Permite enviar perguntas em linguagem natural para o ANS-Assistant
e obter respostas fundamentadas estritamente na RN 566/2022 com citação de fontes oficiais.
"""

import argparse
import sys
from pathlib import Path

# Adicionar diretório raiz ao PYTHONPATH para permitir imports do pacote app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.generation.prompts import FALLBACK_INSUFFICIENT_CONTEXT
from app.generation.rag import RAGPipeline


def run_ask(query: str, top_k: int = 10) -> None:
    """Executa o pipeline RAG e exibe a resposta e fontes no terminal."""
    if not query or not query.strip():
        print("[ERRO] Informe uma pergunta para obter a resposta do ANS-Assistant.")
        print("Exemplo: python scripts/ask.py \"qual o prazo máximo para consulta de pediatria?\"")
        sys.exit(1)

    print("=" * 80)
    print("        ANS-ASSISTANT — RESPOSTA FUNDAMENTADA (RAG MVP)")
    print("=" * 80)
    print(f" PERGUNTA REALIZADA : \"{query.strip()}\"")
    print("-" * 80)

    try:
        pipeline = RAGPipeline()
        result = pipeline.generate_answer(query=query, top_k=top_k)
    except Exception as e:
        print(f"\n[ERRO CRÍTICO] Falha ao executar a geração RAG: {e}")
        sys.exit(1)

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    chunks_count = result.get("chunks_count", 0)
    is_success = result.get("success", True)
    is_fallback = answer == FALLBACK_INSUFFICIENT_CONTEXT

    print("\n[RESPOSTA FUNDAMENTADA DO LLM]:\n")
    print(answer)
    print("\n" + "-" * 80)

    if not is_success:
        print(" [STATUS]: Contexto recuperado com sucesso, porém a geração da resposta falhou.")
        print("-" * 80)
    elif is_fallback:
        print(" CONTEXTO RECUPERADO (Sem correspondência suficiente na norma):")
        print("-" * 80)
        if sources:
            for idx, src in enumerate(sources, start=1):
                title = src.get("title", "")
                page = src.get("page_number", "N/A")
                official_url = src.get("official_url", "")
                print(f"  [{idx}] {title} — Página {page}")
                print(f"      URL Oficial: {official_url}")
    else:
        print(" FONTES UTILIZADAS (Contexto da RN 566/2022):")
        print("-" * 80)
        if sources:
            for idx, src in enumerate(sources, start=1):
                title = src.get("title", "")
                page = src.get("page_number", "N/A")
                official_url = src.get("official_url", "")
                print(f"  [{idx}] {title} — Página {page}")
                print(f"      URL Oficial: {official_url}")
        else:
            print("  Nenhuma fonte correspondente encontrada.")

    print(f"\n (Total de fragmentos normativos analisados: {chunks_count})")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script CLI de Pergunta e Resposta RAG do ANS-Assistant."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="Pergunta em linguagem natural (ex.: 'qual o prazo máximo para atendimento?')",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Número de fragmentos recuperados para compor o contexto (padrão: 10).",
    )

    args = parser.parse_args()

    if not args.query:
        print("[AVISO] Nenhuma pergunta informada.")
        print("Uso: python scripts/ask.py \"sua pergunta aqui\" [--top-k 10]\n")
        print("Exemplo de pergunta:")
        print("  python scripts/ask.py \"qual o prazo máximo para consulta médica em pediatria?\"")
        sys.exit(0)

    run_ask(args.query, top_k=args.top_k)


if __name__ == "__main__":
    main()
