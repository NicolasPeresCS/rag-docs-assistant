"""CLI: `python -m ragdocs.cli index` e `python -m ragdocs.cli ask "pergunta"`."""

from __future__ import annotations

import argparse

from .config import get_settings
from .pipeline import RagPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragdocs", description="Perguntas sobre seus documentos")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="(re)indexa a pasta de documentos")
    ask = sub.add_parser("ask", help="faz uma pergunta")
    ask.add_argument("question")
    ask.add_argument("-k", "--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.command == "index":
        pipeline = RagPipeline.from_settings(settings, rebuild=True)
        print(f"Indexados {len(pipeline.chunks)} trechos de {settings.docs_dir}")
        return 0

    pipeline = RagPipeline.from_settings(settings)
    result = pipeline.ask(args.question, args.top_k)
    print(result.text)
    for s in result.sources:
        mark = "*" if s.cited else " "
        page = f" p.{s.page}" if s.page else ""
        print(f" {mark} [{s.id}] {s.source}{page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
