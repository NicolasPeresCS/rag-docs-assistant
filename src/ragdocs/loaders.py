"""Leitura de documentos (.txt, .md, .pdf) para uma estrutura comum."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class Document:
    source: str
    text: str
    page: int | None = None


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def load_uploaded(name: str, data: bytes) -> list[Document]:
    """Converte bytes em documentos. PDFs geram um documento por página."""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Formato não suportado: {suffix or name!r}. Use {sorted(SUPPORTED)}.")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(Document(source=name, text=text, page=number))
        return pages
    text = _decode(data).strip()
    return [Document(source=name, text=text)] if text else []


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    return load_uploaded(path.name, path.read_bytes())


def load_directory(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Pasta de documentos não encontrada: {directory}")
    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            docs.extend(load_file(path))
    return docs
