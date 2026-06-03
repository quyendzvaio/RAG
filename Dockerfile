# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers \
    EMBEDDING_BACKEND=sentence-transformers \
    EMBEDDING_PROFILE=vietnamese \
    EMBEDDING_DEVICE=cpu \
    RERANKER_DEVICE=cpu \
    EMBEDDING_BATCH_SIZE=8 \
    RERANKER_BATCH_SIZE=4 \
    AUTO_BUILD_INDEX=false

ARG INSTALL_EMBEDDINGS=true
ARG TORCH_VERSION=2.3.1+cpu
ARG SENTENCE_TRANSFORMERS_VERSION=3.0.1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gosu libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt \
    && if [ "${INSTALL_EMBEDDINGS}" = "true" ]; then \
        python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
        && python -m pip install "sentence-transformers==${SENTENCE_TRANSFORMERS_VERSION}"; \
    fi \
    && python -m pip check

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

EXPOSE 7860 8001 8002 8003 8004

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "rag.app"]
