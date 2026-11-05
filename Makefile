COMPOSE ?= docker compose
EVAL_TAG ?= latest
# Commit cua code dua vao image (metadata trace LangSmith); -dirty khi con thay doi chua commit.
export GIT_COMMIT ?= $(shell git describe --always --dirty 2>/dev/null || echo unknown)
MODEL_CACHE ?= rag_model_cache
UV_BIN ?= $(shell command -v uv)
# Chi dung cho make export-onnx: cai tam trong container, khong vao image hay uv.lock.
EXPORT_DEPS ?= torch==2.3.1+cpu sentence-transformers==3.0.1 onnx==1.17.0

.PHONY: prepare-wiki-sources docker-build docker-up docker-down docker-logs docker-vector-logs docker-wiki-logs docker-build-vector-index build-wiki-layer docker-rebuild-all-data docker-clean-volumes eval-golden eval-retrieval eval-ragas-deps eval-ragas analyze-ingestion export-onnx

# Chay trong container (host khong can langchain); ghi ra ./wiki_data/raw_sources.
prepare-wiki-sources:
	mkdir -p wiki_data
	$(COMPOSE) run --rm --remove-orphans -v ./wiki_data:/app/wiki_data vector-rag-service python prepare_wiki_sources.py --clean

docker-build:
	$(COMPOSE) build

# Export model embedding sang ONNX vao volume model_cache (mot lan, hoac sau khi doi EMBEDDING_PROFILE/MODEL).
# Image khong co torch: lenh nay cai torch + sentence-transformers vao /tmp cua mot container tam (dung uv cua host,
# ghim cac goi da co trong image), export, roi container bi xoa. Service bao loi va nhac lenh nay khi thieu model.
export-onnx:
	$(COMPOSE) run --rm --no-deps -u root --entrypoint "" -v $(UV_BIN):/usr/local/bin/uv:ro vector-rag-service sh -c '\
		uv pip freeze --python /opt/venv/bin/python > /tmp/pins.txt \
		&& uv pip install --python /opt/venv/bin/python --target /tmp/export-deps -c /tmp/pins.txt \
			--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match $(EXPORT_DEPS) \
		&& PYTHONPATH=/tmp/export-deps:/app python -c "from rag.embeddings import get_embedding_model; print(type(get_embedding_model()).__name__)" \
		&& chown -R appuser:appuser /cache'

docker-up:
	$(COMPOSE) up --remove-orphans

docker-down:
	$(COMPOSE) down --remove-orphans

docker-logs:
	$(COMPOSE) logs -f

docker-vector-logs:
	$(COMPOSE) logs -f vector-rag-service

docker-wiki-logs:
	$(COMPOSE) logs -f wiki-rag-service

docker-build-vector-index:
	$(COMPOSE) run --rm --remove-orphans vector-rag-service python create_faiss.py

# Lop wiki do LLM viet cho cac de muc chon, vd: make build-wiki-layer SOURCES="aec013b7 059ac230".
# wiki-rag-service chi tra loi tu lop nay va tu nap lai khi index.md doi. Can llm-service dang chay.
build-wiki-layer:
	$(COMPOSE) run --rm --no-deps --remove-orphans -v ./wiki_data:/app/wiki_data vector-rag-service python build_wiki.py $(SOURCES)

docker-rebuild-all-data: prepare-wiki-sources docker-build-vector-index

docker-clean-volumes:
	$(COMPOSE) down -v --remove-orphans

# Sinh eval/golden.jsonl bang DeepSeek (can wiki_data/raw_sources va DEEPSEEK_API_KEY).
eval-golden:
	python3 scripts/make_golden_set.py

# Can stack dang chay (make docker-up). Vd: make eval-retrieval EVAL_TAG=refactor ARGS="--compare baseline"
eval-retrieval:
	python3 scripts/eval_retrieval.py --tag $(EVAL_TAG) $(ARGS)

# RAGAS: ragas cai vao volume rag_eval_deps (mot lan), khong vao image dich vu. Can stack dang chay.
# Image khong co uv: xuat danh sach goi bang uv tren host, roi mount chinh uv cua host vao container de cai.
eval-ragas-deps:
	mkdir -p .cache && uv export --frozen --only-group eval --no-hashes --no-emit-project -o .cache/eval-requirements.txt
	$(COMPOSE) run --rm --no-deps -u root --entrypoint "" -v rag_eval_deps:/deps -v $(UV_BIN):/usr/local/bin/uv:ro \
		-v ./.cache/eval-requirements.txt:/tmp/eval.txt:ro vector-rag-service \
		sh -c 'uv pip install --python /opt/venv/bin/python --target /deps -r /tmp/eval.txt && chown -R 1000:1000 /deps'

# vd: make eval-ragas EVAL_TAG=ragas-v1 ARGS="--limit 5"
eval-ragas:
	$(COMPOSE) run --rm --no-deps -e PYTHONPATH=/deps:/app -v rag_eval_deps:/deps:ro -v ./scripts:/app/scripts:ro \
		-v ./eval:/app/eval -v ./wiki_data:/app/wiki_data:ro vector-rag-service python scripts/eval_ragas.py --tag $(EVAL_TAG) $(ARGS)

# Bao cao chat luong parse/chunk/embedding input/wiki lookup (can wiki_data/raw_sources va model_cache).
analyze-ingestion:
	docker run --rm --entrypoint python \
		-v ./data_raw:/app/data_raw:ro -v ./wiki_data:/app/wiki_data:ro -v ./scripts:/app/scripts:ro -v $(MODEL_CACHE):/cache \
		rag-ttcsn:cpu scripts/analyze_ingestion.py
