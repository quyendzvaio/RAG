COMPOSE ?= docker compose
EVAL_TAG ?= latest

.PHONY: prepare-wiki-sources docker-build docker-up docker-down docker-logs docker-vector-logs docker-wiki-logs docker-build-vector-index docker-ingest-wiki docker-rebuild-all-data docker-clean-volumes eval-golden eval-retrieval analyze-ingestion

# Chay trong container (host khong can langchain); ghi ra ./wiki_data/raw_sources.
prepare-wiki-sources:
	mkdir -p wiki_data
	$(COMPOSE) run --rm --remove-orphans -v ./wiki_data:/app/wiki_data vector-rag-service python prepare_wiki_sources.py --clean

docker-build:
	$(COMPOSE) build

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

docker-ingest-wiki:
	$(COMPOSE) run --rm --remove-orphans wiki-rag-service python -m services.wiki_rag_service.ingest

docker-rebuild-all-data: prepare-wiki-sources docker-build-vector-index docker-ingest-wiki

docker-clean-volumes:
	$(COMPOSE) down -v --remove-orphans

# Sinh eval/golden.jsonl bang DeepSeek (can wiki_data/raw_sources va DEEPSEEK_API_KEY).
eval-golden:
	python3 scripts/make_golden_set.py

# Can stack dang chay (make docker-up). Vd: make eval-retrieval EVAL_TAG=refactor ARGS="--compare baseline"
eval-retrieval:
	python3 scripts/eval_retrieval.py --tag $(EVAL_TAG) $(ARGS)

# Bao cao chat luong parse/chunk/embedding input/wiki lookup (can wiki_data/raw_sources va model_cache).
analyze-ingestion:
	docker run --rm --entrypoint python -e HF_HOME=/cache/huggingface -e SENTENCE_TRANSFORMERS_HOME=/cache/sentence-transformers \
		-v ./data_raw:/app/data_raw:ro -v ./wiki_data:/app/wiki_data:ro -v ./scripts:/app/scripts:ro -v rag_model_cache:/cache \
		rag-ttcsn:cpu scripts/analyze_ingestion.py
