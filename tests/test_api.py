from fastapi.testclient import TestClient

from ragdocs.api import create_app


def client(pipeline):
    return TestClient(create_app(pipeline))


def test_health(pipeline):
    body = client(pipeline).get("/health").json()
    assert body["status"] == "ok" and body["chunks"] > 0


def test_ask(pipeline):
    r = client(pipeline).post("/ask", json={"question": "Qual o horário do suporte por chat?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["answered"] and body["sources"][0]["source"] == "faq_produto.md"


def test_ask_valida_entrada(pipeline):
    c = client(pipeline)
    assert c.post("/ask", json={"question": "a"}).status_code == 422
    assert c.post("/ask", json={"question": "pergunta válida", "top_k": 99}).status_code == 422


def test_llm_fora_do_ar_vira_502(pipeline):
    from ragdocs.llm import LLMError

    class Broken:
        name = "broken"

        def complete(self, system, user):
            raise LLMError("indisponível")

    pipeline.llm = Broken()
    r = client(pipeline).post("/ask", json={"question": "Quanto custa o plano Business?"})
    assert r.status_code == 502
