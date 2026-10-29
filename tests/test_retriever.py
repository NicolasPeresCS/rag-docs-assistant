from ragdocs.chunking import Chunk
from ragdocs.retriever import BM25Index, DenseIndex, HybridRetriever, tokenize


def test_tokenize_remove_acentos_stopwords_e_plural():
    assert tokenize("A Política dos Dias de Férias") == ["politica", "dia", "feria"]
    assert "busines" not in tokenize("plano Business")


def test_bm25_prioriza_documento_relevante():
    docs = ["gatos dormem muito", "cachorros correm no parque", "o parque tem cachorros e gatos"]
    scores = BM25Index(docs).scores("cachorros parque")
    assert scores.index(max(scores)) in (1, 2)
    assert scores[0] == 0


def test_cobertura():
    cov = BM25Index(["alpha beta", "gamma"]).coverage("alpha beta delta")
    assert cov[0] > cov[1] == 0


def _chunks(*texts):
    return [Chunk(id=f"d#{i}", source="d", text=t) for i, t in enumerate(texts)]


def test_busca_sem_resultado_para_termo_desconhecido():
    retriever = HybridRetriever(_chunks("alpha beta", "gamma delta"))
    assert retriever.search("zzzz") == []


def test_busca_em_indice_vazio():
    assert HybridRetriever([]).search("qualquer coisa") == []


def test_fusao_rrf_com_embedder_falso():
    """O índice denso resgata um trecho que o BM25 não encontra (sinônimo)."""
    chunks = _chunks("o carro é rápido", "a banana é amarela", "o automóvel corre muito")
    vectors = {
        "o carro é rápido": [1.0, 0.0],
        "a banana é amarela": [0.0, 1.0],
        "o automóvel corre muito": [0.95, 0.05],
        "carro": [1.0, 0.0],
    }
    dense = DenseIndex([c.text for c in chunks], lambda texts: [vectors[t] for t in texts])
    hits = HybridRetriever(chunks, dense=dense).search("carro", k=3)
    ordered = [h.chunk.text for h in hits]
    assert ordered[0] == "o carro é rápido"
    assert "o automóvel corre muito" in ordered
    assert all(h.dense_score is not None for h in hits)
