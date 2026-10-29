"""Divisão de documentos em trechos (chunks) respeitando parágrafos."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Document


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str
    page: int | None = None


def _sections(text: str) -> list[str]:
    """Divide em seções por títulos Markdown; um título sozinho é anexado à seção seguinte."""
    sections: list[str] = []
    carry = ""
    for section in re.split(r"\n(?=#{1,6} )", text.strip()):
        section = section.strip()
        if not section:
            continue
        section = f"{carry}\n\n{section}" if carry else section
        carry = ""
        if "\n" not in section and section.startswith("#"):
            carry = section  # só o título: junta com o próximo
        else:
            sections.append(section)
    if carry:
        sections.append(carry)
    return sections


def chunk_text(text: str, size: int = 700, overlap: int = 100) -> list[str]:
    """Quebra por seções (títulos Markdown) e depois agrupa parágrafos até ~`size` caracteres.

    Cada seção vira ao menos um trecho próprio, para que assuntos diferentes não se misturem
    no mesmo chunk (isso piorava a busca).
    """
    chunks: list[str] = []
    for section in _sections(text):
        chunks.extend(_pack(section, size, overlap))
    return chunks


def _pack(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("size deve ser positivo")
    if not 0 <= overlap < size:
        raise ValueError("overlap deve estar entre 0 e size")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        # Parágrafos maiores que o limite são cortados em espaços.
        while len(paragraph) > size:
            cut = paragraph.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:cut])
            paragraph = paragraph[cut:].lstrip()
        if not paragraph:
            continue

        if current and len(current) + len(paragraph) + 2 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail.split(" ", 1)[-1] if " " in tail else tail  # evita começar no meio da palavra
            current = f"{tail}\n\n{paragraph}" if tail else paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph

    if current:
        chunks.append(current)
    return chunks


def chunk_documents(docs: list[Document], size: int = 700, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for doc in docs:
        for piece in chunk_text(doc.text, size, overlap):
            n = counters.get(doc.source, 0)
            counters[doc.source] = n + 1
            chunks.append(Chunk(id=f"{doc.source}#{n}", source=doc.source, text=piece, page=doc.page))
    return chunks
