FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data ./data
ENV DOCS_DIR=data/sample_docs INDEX_PATH=/tmp/index/chunks.json
EXPOSE 8000
CMD ["uvicorn", "ragdocs.api:app", "--host", "0.0.0.0", "--port", "8000"]
