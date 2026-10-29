from types import SimpleNamespace

import pytest

from ragdocs.config import get_settings
from ragdocs.llm import AnthropicLLM, ExtractiveLLM, LLMError, build_llm


def test_sem_chave_usa_modo_extrativo(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(build_llm(get_settings()), ExtractiveLLM)


def test_com_chave_usa_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-teste")
    assert isinstance(build_llm(get_settings()), AnthropicLLM)


def test_anthropic_extrai_texto_da_resposta():
    llm = AnthropicLLM("sk-teste", "modelo-x", 100)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=" Resposta [1] ")])

    llm._client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    assert llm.complete("sistema", "usuario") == "Resposta [1]"
    assert captured["model"] == "modelo-x" and captured["system"] == "sistema"


def test_anthropic_converte_erro_da_api():
    import anthropic

    llm = AnthropicLLM("sk-teste", "modelo-x", 100)

    def boom(**kwargs):
        raise anthropic.APIConnectionError(request=SimpleNamespace())

    llm._client = SimpleNamespace(messages=SimpleNamespace(create=boom))
    with pytest.raises(LLMError):
        llm.complete("s", "u")
