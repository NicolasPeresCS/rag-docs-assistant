"""Persistência simples dos chunks em JSON (o índice BM25 é reconstruído ao carregar)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .chunking import Chunk


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")


def load_chunks(path: str | Path) -> list[Chunk]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Chunk(**item) for item in data]
