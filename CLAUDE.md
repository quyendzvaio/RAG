# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Vietnamese legal-document RAG system that answers the same question two ways and compares them: **vector RAG** (parent-child: FAISS over clause-sized child chunks plus BM25 over whole articles, fused and reranked per article) and **wiki RAG** (BM25 over normalized Markdown raw sources stored in SQLite). The answer, the retrieved contexts and the per-run metrics are shown side by side in a Gradio UI at `http://localhost:7860/ui`.

There is no unit test suite and no lint config. To check a change, run the stack and call the API with `"use_llm": false`, which skips DeepSeek and returns an extractive answer. For retrieval changes, measure with the golden set (`eval/golden.jsonl`) and compare against a saved run: `make eval-retrieval EVAL_TAG=<new> ARGS="--compare baseline"` reports Recall@1/@5, MRR and latency for accented and unaccented questions, and how many top-k lists are identical (a pure refactor must keep them all identical).

## Commands

All commands run from the repo root. Index paths (`data_raw`, `faiss_index`, `hybrid_index`, `wiki_data/raw_sources`, `wiki_db`) are relative to the current working directory, not to the module.

```bash
cp .env.example .env && python3 check_env.py   # DEEPSEEK_API_KEY etc.
make docker-build
make docker-rebuild-all-data   # = prepare-wiki-sources + docker-build-vector-index + docker-ingest-wiki
make docker-up                 # gateway :7860, vector :8001, wiki :8002, llm :8003, evaluator :8004
make docker-vector-logs | docker-wiki-logs | docker-logs
```

Smoke test:

```bash
curl -s localhost:7860/ask/compare -H 'content-type: application/json' \
  -d '{"question":"doi tuong ap dung hien lay ghep mo","top_k":2,"candidate_k":20,"use_llm":false}'
```

Per-service introspection: `GET :8001/index/status`, `GET :8002/wiki/status`, `GET :8004/eval/runs`.

`make prepare-wiki-sources` runs `prepare_wiki_sources.py --clean` inside the image and writes `./wiki_data/raw_sources`, which the wiki service bind-mounts read-only. `scripts/*.py` (golden set, retrieval eval) use only the standard library and run on the host.

Running a service outside Docker means overriding the service URLs, which default to Docker hostnames (`VECTOR_SERVICE_URL`, `WIKI_SERVICE_URL`, `EVALUATOR_SERVICE_URL`, `LLM_SERVICE_URL`), e.g. `uvicorn services.vector_rag_service.app:app --port 8001`.

## Architecture

```
api-gateway (FastAPI + Gradio at /ui)
  /ask/compare ─┬─> vector-rag-service /answer ─┐
                └─> wiki-rag-service /wiki/answer ─┴─> llm-service /chat (DeepSeek)
                then ─> evaluator-service /eval/run (metrics → SQLite)
```

- `rag/`: the vector pipeline library.
  - `data_loader.py` parses the legal HTML by CSS class (`pChuong` chapter, `pDieu` article, `pGhiChu` legal note, `pNoiDung` content, `pChiDan` cross-reference). Each article becomes one parent `Document`, and each attached appendix (a line `PHỤ LỤC …`) becomes a separate parent with `part="appendix"` and the article's code. `page_content` is the article body only; heading, note, chapter and cross-references are metadata. Tables become `cell | cell` rows. `doc_type` comes from the code in `article_code` (`.TT.`, `.NĐ.`, …). Files with no article (stub "Đề mục" pages) are skipped.
  - `chunking.py` splits each parent into children of at most the embedding model's `max_seq_length` (256) tokens, measured with the model's own tokenizer. It cuts at khoản (`1.`), then điểm (`a)`), line, sentence and word, packs short pieces to about 200 tokens, and prefixes every child with the article heading.
  - `vector_store.py` builds and loads FAISS (children), `parents.jsonl` (parents), BM25 (over parents) and `index_manifest.json` (`version: 2`).
  - `query_engine.py` does the hybrid search.
- `services/<name>/app.py`: one FastAPI app per service. All services run from one image (`rag-ttcsn:cpu`) and differ only in the uvicorn command in `docker-compose.yml`.
- `shared/schemas.py`: the Pydantic contracts between services (`AskRequest`, `ServiceAnswer`, `ContextItem`, `EvaluationRecord`, …). Changing a field here changes every service.
- `shared/answering.py`: the answer flow both RAG services share (`answer_question`: retrieve → DeepSeek or extractive answer → `ServiceAnswer`). The two services differ only in their retrieve function and their `PromptStyle`.
- `shared/config.py`: every env var, read once by `get_settings()`.

### Retrieval scoring (vector service)

Retrieval works per parent (article). `candidate_hits` takes `candidate_k * CHILD_FANOUT` children from FAISS and groups them into parents, ranked by their best child; BM25 ranks parents directly. `hybrid_search` fuses the two parent rankings with weighted reciprocal rank (`weight / (rank + RRF_K)`, `RRF_K = 60`, weights 0.55 / 0.45), takes the top `candidate_k`, then reranks: final score = `0.65 * rerank + 0.35 * fused`, both min-max normalized over the candidates (raw RRF scores are about 0.01, so unnormalized they barely move the order). The constants are at the top of `rag/query_engine.py`. The reranker scores the same text that goes to the LLM (`context_text`): the whole parent when it fits in `PARENT_CONTEXT_CHARS` (6000), otherwise the matched children, their neighbours and the best keyword children, in article order. It is a neural CrossEncoder only when `RERANKER_MODEL` is set; otherwise it uses `_lexical_rerank_score` (token coverage, phrase matches and metadata boosts).

### Wiki service

`prepare_wiki_sources` reuses `parse_legal_html` to write `wiki_data/raw_sources/{sources/*.md, articles.jsonl, manifest.json, index.md, RAW_SOURCE_SCHEMA.md}`. Ingest loads `articles.jsonl` into SQLite. `articles.jsonl` carries each article's full `content`, and ingest stores it in the `content` column. At query time the service scores with in-memory BM25 over heading (repeated `HEADING_REPEAT` = 3 times) + full content + chapter + legal note + doc type, with `b = 0.9` so long articles don't win on length (the same `BM25Index` class as the vector service; `searchable_text` is built at ingest, so changing it means re-ingesting) and returns `content` from SQLite. It never re-reads the Markdown. A DB with the old schema (no `content` column) is dropped and recreated on startup, so it must be re-ingested. The raw-source layer is treated as immutable; LLM-generated wiki pages belong in a separate layer, `wiki_data/wiki/`.

`build_wiki.py` (`rag/wiki_builder.py`, `make build-wiki-layer SOURCES=...`) is an experimental Karpathy-style "LLM Wiki" layer for selected đề mục: through llm-service it plans topics, then writes `de-muc/`, `van-ban/` and `chu-de/` pages in which every point cites `[Điều <code>]`, and generates `index.md`, `log.md` and `SCHEMA.md`. Links, article lists and the index are generated by code; `lint_wiki` checks the citations and links the LLM wrote. A response that uses up `max_tokens` is treated as cut off and retried once at double budget. The wiki service does not read this layer yet.

## Gotchas

- **Index/embedding coupling.** `index_manifest.json` records the embedding backend, profile and model. On load, `_validate_manifest` raises if `EMBEDDING_BACKEND`, `EMBEDDING_PROFILE` or `EMBEDDING_MODEL` resolve differently (device and batch size don't matter). A silent fallback to `HashingEmbeddings` at load time also trips this check. `_validate_manifest` also rejects any manifest whose `version` isn't `INDEX_VERSION` (2 = parent-child). Changing the embedding profile, `chunking.py` or `data_loader` parsing means rebuilding with `make docker-build-vector-index`. Parsing also feeds the wiki, so re-run `make prepare-wiki-sources docker-ingest-wiki` as well.
- **BM25 pickle holds no documents.** `bm25_store.pkl` stores only term stats and keys. The parents are re-attached from `parents.jsonl` on load, and children are grouped from the FAISS docstore by `parent_id`, so all three files must come from the same build.
- **Shared tokenizer.** `rag.bm25_store.tokenize` (`[\wÀ-ỹ]+`, lowercased, no word segmentation) feeds vector BM25, wiki BM25 and the evaluator's overlap metrics. Changing it shifts all three and invalidates the BM25 pickle.
- **Failures degrade instead of erroring.** If sentence-transformers fails to import, embeddings fall back to `HashingEmbeddings` (smoke-test quality only; the manifest then reads `hashing`). Without an API key, or on any DeepSeek error, `llm-service` returns placeholder text with `fallback=True`. The answer endpoints return `ServiceAnswer(error=...)` with HTTP 200, and the gateway drops evaluator failures. When results look wrong, check service logs and the `error` / `fallback` fields before suspecting the retrieval logic.
- **GPU if usable, else CPU.** `EMBEDDING_DEVICE` and `RERANKER_DEVICE` default to `auto`: `rag/device.py` picks CUDA only when it is available with compute capability ≥ 5 and ≥ 3 GB of VRAM, otherwise CPU. The local machine (16 GB RAM, old GPU) and the Docker image (CPU torch wheel, `TORCH_VERSION=2.3.1+cpu`) therefore always run on CPU. Heavy jobs such as building the vector index run on a remote GPU host (Kaggle) and the index is copied back. Device doesn't enter the manifest check. Never `pip install sentence-transformers` into the image without the CPU index URL, or it pulls CUDA torch.
- **Build data before `docker-up`.** Containers never auto-build unless `AUTO_BUILD_INDEX=true`. The services still start without data, but vector `/health` reports `index_not_ready: ...` and wiki `/health` reports `empty_db`. Their healthchecks fail, and the gateway's `depends_on: service_healthy` keeps it from ever starting. The wiki DB can also be re-ingested live with `POST :8002/wiki/ingest`.
- **Loading the indexes runs pickle.** `FAISS.load_local(..., allow_dangerous_deserialization=True)` and the BM25 pickle execute code on load, so load only indexes built locally.
- **BM25 pickle layout.** `BM25Index`/`BM25Store` keep their old attribute names so existing `bm25_store.pkl` files still load. Renaming an attribute requires rebuilding the vector index.
- **Unpinned requirements.** `requirements.txt` uses lower bounds only. `gradio` is capped at `<6` because gradio 6 needs `huggingface-hub>=1.16`, which conflicts with the pinned `sentence-transformers==3.0.1`, and the build's `pip check` fails.
- **Bit-exact scoring.** Write the RRF term as `weight * (1.0 / (rank + RRF_K))`. `weight / (rank + RRF_K)` differs in the last bit for some ranks, which reorders ties and breaks snapshot comparisons.

## Conventions

- Log, error and print messages are in **unaccented** Vietnamese (`"Dang tai FAISS vector index..."`). LLM prompts and user-facing answer text are in accented Vietnamese.
- Config comes only from `shared/config.get_settings()` (frozen dataclasses, defaults matching `docker-compose.yml`, `.env` loaded if present). Don't add `os.getenv` elsewhere. Algorithm constants (fusion weights, chunk size) are named module constants next to their code, not env vars.
- Library code logs with `logging.getLogger(__name__)`. Services call `shared.log.setup_logging()`. `print` is only for CLI script output.
- Services cache their heavy store (and the reranker, `get_reranker()`) with `@lru_cache(maxsize=1)` and warm it in the FastAPI `lifespan` (`PRELOAD_VECTOR_INDEX`, `PRELOAD_WIKI_CACHE`).
