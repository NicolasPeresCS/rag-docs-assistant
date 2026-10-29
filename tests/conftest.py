import pytest

from ragdocs.config import get_settings
from ragdocs.loaders import load_directory
from ragdocs.pipeline import RagPipeline


class FakeLLM:
    """LLM falso: registra o prompt recebido e devolve uma resposta com citação."""

    name = "fake"

    def __init__(self, reply: str = "São 3 dias por semana [1]."):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def pipeline(fake_llm):
    settings = get_settings()
    return RagPipeline.from_documents(load_directory("data/sample_docs"), fake_llm, settings)
