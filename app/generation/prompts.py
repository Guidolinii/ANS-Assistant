"""
Módulo de Prompts e Engenharia de Prompt.

Responsável pela definição de templates de prompt estruturados para o LLM,
garantindo respostas baseadas estritamente em regulamentações da ANS e citações de fontes.
"""

from typing import Any, Dict, List


FALLBACK_INSUFFICIENT_CONTEXT = (
    "Não foram encontradas informações suficientes na regulamentação disponível (RN 566/2022) "
    "para responder à sua pergunta."
)

SYSTEM_PROMPT = """Você é o ANS-Assistant, um assistente virtual especializado na regulamentação da Agência Nacional de Saúde Suplementar (ANS) e legislação brasileira de saúde suplementar.

REGRAS DE RESPOSTA (RIGOROSAMENTE OBRIGATÓRIAS):
1. Responda à pergunta do usuário UTILIZANDO EXCLUSIVAMENTE as informações contidas no CONTEXTO REGULATÓRIO fornecido abaixo.
2. NUNCA utilize conhecimentos prévios externos ou especulações que não estejam expressamente escritos nos trechos normativos fornecidos.
3. Se os trechos do CONTEXTO REGULATÓRIO não contiverem dados suficientes para responder à pergunta com precisão, responda EXATAMENTE:
   "Não foram encontradas informações suficientes na regulamentação disponível (RN 566/2022) para responder à sua pergunta."
4. Mantenha tom formal, técnico e normativo.
5. Sempre que citar prazos ou regras, mencione a norma (RN nº 566/2022) e a página correspondente constante nos trechos.
"""


class RAGPromptBuilder:
    """Construtor de prompts instrucionais para RAG com grounding e fallback."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formata os chunks recuperados em um bloco legível de contexto regulatório.

        Args:
            chunks: Lista de dicionários contendo 'text' e 'metadata'.

        Returns:
            String formatada do contexto.
        """
        if not chunks:
            return "NENHUM CONTEXTO DISPONÍVEL."

        formatted_blocks: List[str] = []
        for index, item in enumerate(chunks, start=1):
            meta = item.get("metadata", {})
            text = item.get("text", "").strip()
            doc_id = meta.get("document_id", "RN-566-2022")
            norm_number = meta.get("norm_number", "566/2022")
            page = meta.get("page_number", "N/A")
            chunk_idx = meta.get("chunk_index", "N/A")

            header = f"--- [TRECHO #{index} | Documento: {doc_id} (RN {norm_number}) | Página: {page} | Chunk #{chunk_idx}] ---"
            formatted_blocks.append(f"{header}\n{text}")

        return "\n\n".join(formatted_blocks)

    def build_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Gera o prompt completo estruturado pronto para envio ao LLM.

        Args:
            query: Pergunta do usuário.
            chunks: Chunks recuperados pelo retriever.

        Returns:
            String contendo o prompt montado.
        """
        context_str = self.format_context(chunks)

        user_message = (
            f"CONTEXTO REGULATÓRIO (RN 566/2022):\n"
            f"{context_str}\n\n"
            f"PERGUNTA DO USUÁRIO: {query}\n\n"
            f"RESPOSTA FUNDAMENTADA:"
        )

        return f"{self.system_prompt}\n\n{user_message}"


class PromptBuilder(RAGPromptBuilder):
    """Classe legada mantida para retrocompatibilidade."""

    pass

