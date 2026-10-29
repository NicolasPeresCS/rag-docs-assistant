"""Avaliação offline da recuperação (sem LLM).

Métricas:
  hit@k      - a fonte esperada aparece entre os k primeiros trechos
  MRR        - média do inverso da posição da fonte esperada
  recusa     - perguntas fora do escopo que o sistema corretamente se recusa a responder
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ragdocs.config import get_settings
from ragdocs.llm import ExtractiveLLM
from ragdocs.loaders import load_directory
from ragdocs.pipeline import RagPipeline

EVAL_FILE = Path(__file__).with_name("eval_set.json")


def evaluate(k: int = 3, verbose: bool = False) -> dict:
    settings = get_settings()
    pipeline = RagPipeline.from_documents(load_directory(settings.docs_dir), ExtractiveLLM(), settings)
    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    in_scope = [c for c in cases if c["expected_source"]]
    out_scope = [c for c in cases if not c["expected_source"]]

    hits, reciprocal = 0, 0.0
    for case in in_scope:
        ranked = [h.chunk.source for h in pipeline.retriever.search(case["question"], k)]
        if case["expected_source"] in ranked:
            hits += 1
            reciprocal += 1 / (ranked.index(case["expected_source"]) + 1)
        elif verbose:
            print(f"  ERRO: {case['question']!r} -> {ranked}")

    refused = sum(1 for c in out_scope if not pipeline.ask(c["question"]).answered)
    answered_in_scope = sum(1 for c in in_scope if pipeline.ask(c["question"]).answered)

    return {
        f"hit@{k}": hits / len(in_scope),
        "mrr": reciprocal / len(in_scope),
        "recusa_fora_do_escopo": refused / len(out_scope),
        "respondidas_no_escopo": answered_in_scope / len(in_scope),
        "n_dentro": len(in_scope),
        "n_fora": len(out_scope),
    }


if __name__ == "__main__":
    results = evaluate(verbose=True)
    for key, value in results.items():
        print(f"{key:24s} {value:.2f}" if isinstance(value, float) else f"{key:24s} {value}")
    sys.exit(0)
