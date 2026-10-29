import pytest

from ragdocs.loaders import load_directory, load_uploaded


def test_texto_utf8_e_latin1():
    assert load_uploaded("a.txt", "olá".encode())[0].text == "olá"
    assert load_uploaded("a.txt", "olá".encode("latin-1"))[0].text == "olá"


def test_formato_invalido():
    with pytest.raises(ValueError):
        load_uploaded("planilha.xlsx", b"x")


def test_arquivo_vazio_nao_gera_documento():
    assert load_uploaded("vazio.md", b"   \n") == []


def test_pasta_inexistente():
    with pytest.raises(FileNotFoundError):
        load_directory("nao/existe")


def test_pdf_gera_documento_por_pagina():
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    import io

    buffer = io.BytesIO()
    c = reportlab.Canvas(buffer)
    c.drawString(72, 750, "Pagina um sobre reembolso")
    c.showPage()
    c.drawString(72, 750, "Pagina dois sobre suporte")
    c.save()
    docs = load_uploaded("manual.pdf", buffer.getvalue())
    assert [d.page for d in docs] == [1, 2]
    assert "reembolso" in docs[0].text
