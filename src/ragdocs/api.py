"""API REST com FastAPI."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .llm import LLMError
from .pipeline import RagPipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)


class SourceOut(BaseModel):
    id: int
    source: str
    page: int | None
    score: float
    cited: bool
    snippet: str


class AskResponse(BaseModel):
    answer: str
    answered: bool
    mode: str
    sources: list[SourceOut]


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(title="RAG Docs Assistant", version=__version__)
    app.state.pipeline = pipeline

    def get_pipeline(rebuild: bool = False) -> RagPipeline:
        if app.state.pipeline is None or rebuild:
            try:
                app.state.pipeline = RagPipeline.from_settings(get_settings(), rebuild=rebuild)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        return app.state.pipeline

    @app.get("/health")
    def health() -> dict:
        p = get_pipeline()
        return {"status": "ok", "chunks": len(p.chunks), "llm": p.llm.name, "version": __version__}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest) -> AskResponse:
        try:
            result = get_pipeline().ask(body.question, body.top_k)
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return AskResponse(
            answer=result.text,
            answered=result.answered,
            mode=result.mode,
            sources=[SourceOut(**s.__dict__) for s in result.sources],
        )

    @app.post("/reindex")
    def reindex() -> dict:
        p = get_pipeline(rebuild=True)
        return {"status": "reindexed", "chunks": len(p.chunks)}

    return app


app = create_app()
