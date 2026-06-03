COMPOSE ?= docker compose

.PHONY: prepare-wiki-sources docker-build docker-up docker-down docker-logs docker-vector-logs docker-wiki-logs docker-build-vector-index docker-migrate-legacy-vector-index docker-ingest-wiki docker-rebuild-all-data docker-clean-volumes

prepare-wiki-sources:
	python3 prepare_wiki_sources.py --clean

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

docker-migrate-legacy-vector-index:
	docker run --rm \
		-v rag_ttcsn_rag_faiss_index:/old_faiss:ro \
		-v rag_ttcsn_rag_hybrid_index:/old_hybrid:ro \
		-v rag_ttcsn_vector_faiss_index:/new_faiss \
		-v rag_ttcsn_vector_hybrid_index:/new_hybrid \
		alpine sh -lc 'rm -rf /new_faiss/* /new_hybrid/* && cp -a /old_faiss/. /new_faiss/ && cp -a /old_hybrid/. /new_hybrid/ && chown -R 1000:1000 /new_faiss /new_hybrid'

docker-ingest-wiki:
	$(COMPOSE) run --rm --remove-orphans wiki-rag-service python -m services.wiki_rag_service.ingest

docker-rebuild-all-data: prepare-wiki-sources docker-build-vector-index docker-ingest-wiki

docker-clean-volumes:
	$(COMPOSE) down -v --remove-orphans
