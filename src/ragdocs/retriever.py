"""Recuperação de trechos: BM25 (léxico) + embeddings opcionais, combinados com RRF."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    ["a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "sobre", "entre", "e", "ou", "que", "se", "ao", "aos", "como", "mais", "mas", "qual", "quais", "quem", "quando", "onde", "isso", "isto", "esse", "essa", "este", "esta", "meu", "minha", "seu", "sua", "ser", "sao", "foi", "quanto", "quanta", "posso", "pode", "preciso", "faco", "fazer", "tem", "ter", "the", "of", "and", "to", "in", "is"]
)


def normalize(text: str) -> str:
    """Minúsculas e sem acentos, para 'política' casar com 'politica'."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN_RE.findall(normalize(text)):
        if token in STOPWORDS or len(token) < 2:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]  # plural simples: "dias" -> "dia" (mantém "business")
        if token in STOPWORDS:  # "quantos" -> "quanto"
            continue
        tokens.append(token)
    return tokens


class BM25Index:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._tfs = [Counter(tokenize(d)) for d in docs]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        n = len(docs)
        self._avgdl = (sum(self._lens) / n) if n else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = set(tokenize(query))
        out = []
        for tf, dl in zip(self._tfs, self._lens):
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                    score += self._idf[term] * f * (self.k1 + 1) / denom
            out.append(score)
        return out

    def coverage(self, query: str) -> list[float]:
        """Fração dos termos únicos da pergunta presentes em cada documento."""
        terms = set(tokenize(query))
        if not terms:
            return [0.0] * len(self._tfs)
        return [sum(1 for t in terms if t in tf) / len(terms) for tf in self._tfs]


class DenseIndex:
    """Busca por similaridade de cosseno. O `embed` é injetado, então é fácil de testar."""

    def __init__(self, texts: Sequence[str], embed: Callable[[list[str]], object]):
        import numpy as np

        self._np = np
        self._embed = embed
        matrix = np.asarray(embed(list(texts)), dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    def scores(self, query: str) -> list[float]:
        np = self._np
        vector = np.asarray(self._embed([query]), dtype=float)[0]
        norm = np.linalg.norm(vector) or 1.0
        return (self._matrix @ (vector / norm)).tolist()


def sentence_transformer_embedder(model_name: str) -> Callable[[list[str]], object]:
    from sentence_transformers import SentenceTransformer  # extra opcional: pip install ".[dense]"

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, normalize_embeddings=True)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    coverage: float
    dense_score: float | None = None


def _rank(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], dense: DenseIndex | None = None, rrf_k: int = 60):
        self.chunks = chunks
        self._bm25 = BM25Index([c.text for c in chunks])
        self._dense = dense
        self._rrf_k = rrf_k

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self.chunks:
            return []
        lexical = self._bm25.scores(query)
        coverage = self._bm25.coverage(query)
        pool = min(len(self.chunks), max(k * 4, 20))
        lexical_rank = [i for i in _rank(lexical)[:pool] if lexical[i] > 0]

        dense_scores: list[float] | None = None
        if self._dense is None:
            fused = {i: lexical[i] for i in lexical_rank}
        else:
            dense_scores = self._dense.scores(query)
            fused = defaultdict(float)
            for position, i in enumerate(lexical_rank):
                fused[i] += 1 / (self._rrf_k + position + 1)
            for position, i in enumerate(_rank(dense_scores)[:pool]):
                fused[i] += 1 / (self._rrf_k + position + 1)

        order = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            Hit(
                chunk=self.chunks[i],
                score=fused[i],
                coverage=coverage[i],
                dense_score=dense_scores[i] if dense_scores else None,
            )
            for i in order
        ]
