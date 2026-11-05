# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers \
    EMBEDDING_ONNX_DIR=/cache/onnx \
    AUTO_BUILD_INDEX=false

# onnx: ONNX Runtime + tokenizer (khong torch). none: HashingEmbeddings, chi de smoke test.
ARG EMBEDDINGS=onnx

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gosu libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Phien ban goi nam trong uv.lock (sua pyproject.toml roi chay `uv lock` tren host); --frozen: lock lech thi loi.
# uv chi duoc mount luc build, khong nam trong image. Compile bytecode: moi tien trinh import nhanh hon ~3 s.
COPY pyproject.toml uv.lock /app/

RUN --mount=from=ghcr.io/astral-sh/uv:0.13.4,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    case "${EMBEDDINGS}" in \
        onnx) extras="--extra onnx" ;; \
        none) extras="" ;; \
        *) echo "EMBEDDINGS phai la onnx hoac none: ${EMBEDDINGS}" >&2; exit 1 ;; \
    esac \
    && UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never \
       uv sync --frozen --no-dev $extras \
    && uv pip check --python /opt/venv/bin/python

COPY rag /app/rag
COPY shared /app/shared
COPY services /app/services
COPY create_faiss.py check_device.py check_env.py README.md /app/
COPY prepare_wiki_sources.py /app/prepare_wiki_sources.py
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN useradd --create-home --shell /bin/sh appuser \
    && mkdir -p /app/faiss_index /app/hybrid_index /app/wiki_db /app/eval_data /cache \
    && chown -R appuser:appuser /app /cache \
    && chmod +x /app/docker-entrypoint.sh

# Commit cua code trong image, ghi vao metadata trace (make docker-build truyen vao); dat cuoi de khong lam mat cache.
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

EXPOSE 7860 8001 8002 8003 8004

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "services.api_gateway.app:app", "--host", "0.0.0.0", "--port", "7860"]
