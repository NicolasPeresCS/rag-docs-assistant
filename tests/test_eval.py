import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from run_eval import evaluate


def test_qualidade_minima_da_recuperacao():
    """Guarda de regressão: mudanças que piorem a busca quebram o CI."""
    results = evaluate(k=3)
    assert results["hit@3"] >= 0.9
    assert results["recusa_fora_do_escopo"] == 1.0
