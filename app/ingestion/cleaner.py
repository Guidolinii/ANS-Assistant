"""
Módulo de Limpeza de Texto.

Responsável pela higienização conservadora, remoção de ruídos de formatação,
padronização de caracteres e estruturação prévia dos textos extraídos das regulamentações da ANS.
"""

import re
import unicodedata


class TextCleaner:
    """Higienizador conservador de texto para documentos normativos."""

    def clean_text(self, text: str) -> str:
        """
        Aplica limpeza conservadora ao texto extraído.

        - Aplica normalização Unicode NFKC (substitui ligaduras tipográficas como 'ﬁ' por 'fi').
        - Normaliza espaços em branco (substitui \\xa0 e múltiplos espaços).
        - Normaliza quebras de linha (\\r\\n -> \\n).
        - Remove espaços em branco ao final das linhas.
        - Limita linhas em branco consecutivas a no máximo duas (\\n\\n).
        - Preserva integralmente o texto jurídico, pontuação e estruturas de artigos/incisos.
        """
        if not text:
            return ""

        # Normalização Unicode NFKC (converte ligaduras como \ufb01 -> fi, \ufb02 -> fl, etc.)
        cleaned = unicodedata.normalize("NFKC", text)

        # Normalizar quebras de linha
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Substituir caracteres de espaço não-quebrável (\xa0) e tabulações por espaço
        cleaned = cleaned.replace("\xa0", " ").replace("\t", " ")

        # Remover espaços no final de cada linha
        lines = [line.rstrip() for line in cleaned.split("\n")]
        cleaned = "\n".join(lines)

        # Substituir múltiplos espaços consecutivos por um único espaço
        cleaned = re.sub(r"[ ]{2,}", " ", cleaned)

        # Limitar mais de 2 quebras de linha consecutivas a 2 (\n\n)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()


