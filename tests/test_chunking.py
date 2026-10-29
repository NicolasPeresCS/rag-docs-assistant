import pytest

from ragdocs.chunking import chunk_documents, chunk_text
from ragdocs.loaders import Document


def test_respeita_paragrafos_e_limite():
    text = "\n\n".join(f"Parágrafo {i} " + "palavra " * 20 for i in range(6))
    chunks = chunk_text(text, size=300, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 40 + 2 for c in chunks)


def test_paragrafo_gigante_e_dividido():
    chunks = chunk_text("palavra " * 500, size=200, overlap=0)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)


def test_overlap_repete_contexto():
    text = "Primeiro bloco com bastante texto aqui.\n\nSegundo bloco com bastante texto aqui."
    chunks = chunk_text(text, size=45, overlap=20)
    assert len(chunks) == 2
    assert chunks[1].startswith(chunks[0][-20:].split(" ", 1)[-1])


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        chunk_text("x", size=0)
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)


def test_ids_unicos_por_fonte():
    docs = [Document("a.md", "um\n\ndois\n\ntres"), Document("b.md", "quatro")]
    chunks = chunk_documents(docs, size=5, overlap=0)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {c.source for c in chunks} == {"a.md", "b.md"}


def test_secoes_markdown_nao_se_misturam():
    text = "# Titulo\n\n## A\n\ncorpo a\n\n## B\n\ncorpo b"
    chunks = chunk_text(text, size=700, overlap=0)
    assert len(chunks) == 2
    assert chunks[0].startswith("# Titulo") and "## A" in chunks[0] and "corpo b" not in chunks[0]
    assert chunks[1].startswith("## B")
