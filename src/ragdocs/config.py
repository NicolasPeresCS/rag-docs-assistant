"""Configuração via variáveis de ambiente (lidas na hora do uso, o que facilita os testes)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    docs_dir: str
    index_path: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_coverage: float
    min_dense: float
    model: str
    max_tokens: int
    use_dense: bool
    dense_model: str
    anthropic_api_key: str | None


def get_settings() -> Settings:
    return Settings(
        docs_dir=os.getenv("DOCS_DIR", "data/sample_docs"),
        index_path=os.getenv("INDEX_PATH", ".index/chunks.json"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        top_k=int(os.getenv("TOP_K", "4")),
        min_coverage=float(os.getenv("MIN_COVERAGE", "0.3")),
        min_dense=float(os.getenv("MIN_DENSE", "0.45")),
        model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"),
        max_tokens=int(os.getenv("MAX_TOKENS", "700")),
        use_dense=os.getenv("USE_DENSE", "0") == "1",
        dense_model=os.getenv(
            "DENSE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
    )
