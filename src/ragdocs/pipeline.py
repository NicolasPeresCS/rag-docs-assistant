"""Pipeline RAG: recupera, decide se há base suficiente, monta o prompt e gera a resposta."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_documents
from .config import Settings, get_settings
from .llm import LLM, SYSTEM_PROMPT, build_llm
from .loaders import Document, load_directory
from .retriever import DenseIndex, Hit, HybridRetriever, sentence_transformer_embedder
from .store import load_chunks, save_chunks

NOT_FOUND = "Não encontrei essa informação nos documentos indexados."


@dataclass(frozen=True)
class Source:
    id: int
    source: str
    page: int | None
    score: float
    cited: bool
    snippet: str


@dataclass(frozen=True)
class Answer:
    text: str
    answered: bool
    mode: str
    sources: list[Source]


def format_context(hits: list[Hit]) -> str:
    parts = []
    for n, hit in enumerate(hits, start=1):
        page = f' page="{hit.chunk.page}"' if hit.chunk.page else ""
        parts.append(f'<document index="{n}" source="{hit.chunk.source}"{page}>\n{hit.chunk.text}\n</document>')
    return "\n\n".join(parts)


class RagPipeline:
    def __init__(self, chunks: list[Chunk], llm: LLM, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.chunks = chunks
        self.llm = llm
        dense = None
        if self.settings.use_dense and chunks:
            embedder = sentence_transformer_embedder(self.settings.dense_model)
            dense = DenseIndex([c.text for c in chunks], embedder)
        self.retriever = HybridRetriever(chunks, dense=dense)

    # ---- construção -------------------------------------------------------------------------
    @classmethod
    def from_documents(cls, docs: list[Document], llm: LLM | None = None, settings: Settings | None = None):
        settings = settings or get_settings()
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        return cls(chunks, llm or build_llm(settings), settings)

    @classmethod
    def from_settings(cls, settings: Settings | None = None, rebuild: bool = False, llm: LLM | None = None):
        settings = settings or get_settings()
        index = Path(settings.index_path)
        if index.exists() and not rebuild:
            return cls(load_chunks(index), llm or build_llm(settings), settings)
        pipeline = cls.from_documents(load_directory(settings.docs_dir), llm, settings)
        save_chunks(pipeline.chunks, index)
        return pipeline

    # ---- consulta ---------------------------------------------------------------------------
    def _has_support(self, hits: list[Hit]) -> bool:
        s = self.settings
        return any(
            h.coverage >= s.min_coverage or (h.dense_score is not None and h.dense_score >= s.min_dense)
            for h in hits
        )

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("A pergunta não pode ser vazia.")

        hits = self.retriever.search(question, top_k or self.settings.top_k)
        if not hits or not self._has_support(hits):
            # Sem base nos documentos: nem chama o LLM (economiza custo e evita alucinação).
            return Answer(NOT_FOUND, answered=False, mode=self.llm.name, sources=[])

        user_prompt = f"{format_context(hits)}\n\nPergunta: {question}"
        text = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        sources = [
            Source(
                id=n,
                source=h.chunk.source,
                page=h.chunk.page,
                score=round(h.score, 4),
                cited=n in cited,
                snippet=" ".join(h.chunk.text.split())[:200],
            )
            for n, h in enumerate(hits, start=1)
        ]
        return Answer(text, answered=True, mode=self.llm.name, sources=sources)
