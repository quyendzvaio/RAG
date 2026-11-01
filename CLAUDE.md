# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Vietnamese legal-document RAG system that answers the same question two ways and compares them: **vector RAG** (parent-child: FAISS over clause-sized child chunks plus BM25 over whole articles, fused and reranked per article) and **wiki RAG** (answers only from an LLM-written wiki layer: the LLM reads the wiki's `index.md`, picks the topic/document pages for the question and answers from them; no BM25). The answer, the retrieved contexts and the per-run metrics are shown side by side in a Gradio UI at `http://localhost:7860/ui`.

There is no unit test suite and no lint config. To check a change, run the stack and call the API with `"use_llm": false`, which skips the DeepSeek answer and returns an extractive one (the wiki service still calls the LLM to pick pages). For retrieval changes, measure with the golden set (`eval/golden.jsonl`) and compare against a saved run: `make eval-retrieval EVAL_TAG=<new> ARGS="--compare baseline"` reports Recall@1/@5, MRR and latency for accented and unaccented questions, and how many top-k lists are identical (a pure refactor must keep them all identical). `ARGS="--services vector"` skips the wiki service (its page selection costs LLM tokens). `--equivalents eval/golden_equivalents.jsonl` also reports `+tuong-duong` metrics that count an article with the same content as the gold one as a hit; `scripts/golden_equivalents.py` builds that file with an LLM judge from saved runs, and a non-reasoning judge accepts wrong pairs (e.g. "hối phiếu nhận nợ" for "hối phiếu đòi nợ"), so review it before trusting those numbers.

## Commands

All commands run from the repo root. Index paths (`data_raw`, `faiss_index`, `hybrid_index`, `wiki_data/raw_sources`, `wiki_data/wiki`) are relative to the current working directory, not to the module.

```bash
cp .env.example .env && python3 check_env.py   # DEEPSEEK_API_KEY etc.
make docker-build
make docker-rebuild-all-data   # = prepare-wiki-sources + docker-build-vector-index
make build-wiki-layer SOURCES="aec013b7 059ac230"   # LLM writes wiki pages for those đề mục (costs tokens)
make docker-up                 # gateway :7860, vector :8001, wiki :8002, llm :8003, evaluator :8004
make docker-vector-logs | docker-wiki-logs | docker-logs
```

Smoke test:

```bash
curl -s localhost:7860/ask/compare -H 'content-type: application/json' \
  -d '{"question":"doi tuong ap dung hien lay ghep mo","top_k":2,"candidate_k":20,"use_llm":false}'
```

Per-service introspection: `GET :8001/index/status`, `GET :8002/wiki/status`, `GET :8004/eval/runs`.

Answer quality is measured with RAGAS (`scripts/eval_ragas.py`, faithfulness / answer relevancy / context precision / context recall): `make eval-ragas-deps` once (exports the `eval` dependency group from `uv.lock` and installs it into the `rag_eval_deps` volume, never into the service image), then `make eval-ragas EVAL_TAG=<tag> ARGS="--limit 5"` with the stack running. The judge is the project's own LLM through llm-service (so it grades its own answers), embeddings are the project's model. The judge's contexts are `shared/answering.format_context` (title, `Citation:` line, content), exactly what the answering LLM saw; scores saved before this change saw the content only (the judge then called correct document numbers and dates unfaithful), and cached scores are reused per tag and judge, so re-score under a new tag. References are written once by the LLM from the gold article (golden set) or the whole đề mục (`eval/synthesis.jsonl`) and cached in `eval/ragas/`, as are the Vietnamese-adapted ragas prompts; service answers are cached per tag, and scores are saved per tag and judge every `SCORE_BATCH` (12) samples in `<tag>-scores.jsonl`, so an interrupted run resumes. `--judge-model` sends the judge (and reference writer) to another model through the same llm-service, to keep the grader fixed while the answering model changes. A quota error (402, "insufficient balance") stops the run at once instead of retrying. The script stubs `ChatVertexAI`/`VertexAI`, which ragas imports only for `isinstance` checks and langchain-community 0.4 no longer has. A full run costs about 40k judge tokens per answered sample.

`make prepare-wiki-sources` runs `prepare_wiki_sources.py --clean` inside the image and writes `./wiki_data/raw_sources`, which `build_wiki.py` reads. The wiki service bind-mounts all of `./wiki_data` read-only. `scripts/*.py` (golden set, retrieval eval) use only the standard library and run on the host.

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

Before any of this, `services/vector_rag_service/query_rewrite.py` sends the question to the LLM (llm-service's default model) to restore missing diacritics, since FAISS and BM25 treat "thue" and "thuế" as different words (unaccented golden Recall@5 0.25 → 0.90 on 20 questions with qwen3-14b; accented unchanged). Only retrieval uses the rewrite; the answer prompt gets the original question. On an LLM error, or when more than `MAX_WORD_CHANGE` of the words change once diacritics are stripped (the model rephrased or answered), it keeps the original. This adds one LLM call (~1 s, ~2k prompt tokens through the current proxy) per new question, not counted in `token_usage`.

Retrieval works per parent (article). `candidate_hits` takes `candidate_k * CHILD_FANOUT` children from FAISS and groups them into parents, ranked by their best child; BM25 ranks parents directly. `hybrid_search` fuses the two parent rankings with weighted reciprocal rank (`weight / (rank + RRF_K)`, `RRF_K = 60`, weights 0.55 / 0.45), takes the top `candidate_k`, then reranks: final score = `0.65 * rerank + 0.35 * fused`, both min-max normalized over the candidates (raw RRF scores are about 0.01, so unnormalized they barely move the order). The constants are at the top of `rag/query_engine.py`. The reranker scores the same text that goes to the LLM (`context_text`): the whole parent when it fits in `PARENT_CONTEXT_CHARS` (6000), otherwise the matched children, their neighbours and the best keyword children, in article order. It is a neural CrossEncoder only when `RERANKER_MODEL` is set; otherwise it uses `_lexical_rerank_score` (token coverage, phrase matches and metadata boosts). Before BM25 and the rerank (not FAISS) the question gets `ABBREVIATIONS` expanded both ways (CMND ↔ chứng minh nhân dân). After the rerank, `boost_article_intent` adds `INTENT_BOOST` (0.35, on the normalized [0, 1] scale) to articles whose title matches the question type in `ARTICLE_INTENTS` ("những ai…" → "Đối tượng áp dụng", "phạm vi…" → "Phạm vi điều chỉnh", "trách nhiệm thi hành", "hiệu lực"): opening and closing articles have generic titles and bodies that never restate the topic, so they lost to specific articles. Adding the document title from `legal_note` to BM25 or the rerank text did not help (it is shared by every article of the document) and was dropped. `scripts/scoring_lab.py` mirrors all of this in its "current" variant.

### Wiki service

`prepare_wiki_sources` reuses `parse_legal_html` to write `wiki_data/raw_sources/{sources/*.md, articles.jsonl, manifest.json, index.md, RAW_SOURCE_SCHEMA.md}`, with each article's full `content` in `articles.jsonl`. This raw-source layer is immutable; it is the input of the wiki layer, not something the service searches.

`build_wiki.py` (`rag/wiki_builder.py`, `make build-wiki-layer SOURCES=<html file prefixes>`) is a Karpathy-style "LLM Wiki" for selected đề mục. Through llm-service it plans topics, then writes `wiki_data/wiki/{de-muc,van-ban,chu-de}/*.md`, in which every point cites `[Điều <code>]`, plus `index.md`, `log.md` and `SCHEMA.md`. Links, article lists and the index are generated by code; `lint_wiki` checks the citations and links the LLM wrote. Building is cumulative per run: a run writes the pages of the đề mục it is given and regenerates `index.md` over all pages on disk.

At query time (`services/wiki_rag_service/wiki_layer.py`) the service uses **only** these pages. One LLM call gets the list of đề mục that have a wiki plus `index.md`. It must first answer `ĐỀ MỤC: <code>` or `ĐỀ MỤC: KHÔNG`, then list up to `min(top_k, MAX_PAGES=3)` pages; only pages of the named đề mục are kept. (Asking only for pages let the LLM attach out-of-scope questions to pages that share a few words: 21/144 such golden questions got pages, versus 4/144 with the đề mục step.) The contexts are those pages without their generated footer lists, cut at `WIKI_CONTEXT_CHARS` (13000) instead of the vector service's 6000. With no page the answer is `no_context` and no answer LLM call is made. Page selection calls the LLM even with `use_llm: false`; that flag only switches the answer to extractive. The selection tokens are added to `token_usage`. `LatestWikiLayer` reloads the pages whenever `index.md`'s mtime changes, so a new build is served without a restart. Each page context carries `metadata.article_codes` (the cited articles, in raw `article_code` form) and `source_file`, which `scripts/eval_retrieval.py` uses to count a page as a hit for the golden article.

## Gotchas

- **Index/embedding coupling.** `index_manifest.json` records the embedding backend, profile and model. On load, `_validate_manifest` raises if `EMBEDDING_BACKEND`, `EMBEDDING_PROFILE` or `EMBEDDING_MODEL` resolve differently (device, batch size and `EMBEDDING_RUNTIME` don't matter). A silent fallback to `HashingEmbeddings` at load time also trips this check. `_validate_manifest` also rejects any manifest whose `version` isn't `INDEX_VERSION` (2 = parent-child). Changing the embedding profile, `chunking.py` or `data_loader` parsing means rebuilding with `make docker-build-vector-index`. Parsing also feeds the wiki, so re-run `make prepare-wiki-sources build-wiki-layer` as well.
- **BM25 pickle holds no documents.** `bm25_store.pkl` stores only term stats and keys. The parents are re-attached from `parents.jsonl` on load, and children are grouped from the FAISS docstore by `parent_id`, so all three files must come from the same build.
- **Shared tokenizer.** `rag.bm25_store.tokenize` (`[\wÀ-ỹ]+`, lowercased, no word segmentation) feeds vector BM25 and the evaluator's overlap metrics. Changing it shifts both and invalidates the BM25 pickle.
- **LLM truncation.** llm-service doesn't return `finish_reason`, so `shared/llm.complete` treats a reply that uses up `max_tokens` as cut off and retries once at double the budget. Answers (`ANSWER_MAX_TOKENS` = 6000) that are still cut get a visible note appended; the wiki builder raises instead of writing a half page. `DEEPSEEK_EXTRA_BODY` (JSON merged into every chat request) turns reasoning off where the provider allows it, e.g. `{"thinking":{"type":"disabled"}}`. Token budgets are caps, set high because reasoning models (glm through kiraai, whose thinking cannot be turned off) spend 300-1300+ completion tokens on reasoning first: a 200-token page-selection budget returned empty content.
- **Failures degrade instead of erroring.** If sentence-transformers fails to import, embeddings fall back to `HashingEmbeddings` (smoke-test quality only; the manifest then reads `hashing`). Without an API key, or on any DeepSeek error, `llm-service` returns placeholder text with `fallback=True`. The answer endpoints return `ServiceAnswer(error=...)` with HTTP 200, and the gateway drops evaluator failures. When results look wrong, check service logs and the `error` / `fallback` fields before suspecting the retrieval logic.
- **GPU if usable, else CPU.** `EMBEDDING_DEVICE` and `RERANKER_DEVICE` default to `auto`: `rag/device.py` picks CUDA only when it is available with compute capability ≥ 5 and ≥ 3 GB of VRAM, otherwise CPU. The local machine (16 GB RAM, old GPU) and the Docker image (CPU torch wheel `2.3.1+cpu`) therefore always run on CPU. Heavy jobs such as building the vector index run on a remote GPU host (Kaggle) and the index is copied back. Device doesn't enter the manifest check.
- **Embeddings run on ONNX Runtime on CPU.** With `EMBEDDING_RUNTIME=onnx` (the default) and a CPU device, `get_embedding_model` returns `rag/onnx_embeddings.OnnxEmbeddings` instead of `HuggingFaceEmbeddings`; on a GPU, or if onnxruntime fails, it uses torch. The first use exports the sentence-transformers model (Transformer + mean/cls Pooling only) with torch, fuses it with onnxruntime's transformer optimizer and caches model, tokenizer and `embedding.json` in `EMBEDDING_ONNX_DIR` (`/cache/onnx` in the image, the `model_cache` volume). It reproduces `HuggingFaceEmbeddings` exactly (newlines → spaces, strip, truncate at `max_seq_length`, L2-normalize): on the golden set the vectors differ by < 1e-6 and all 150 top-5 lists are identical, so the same index serves both runtimes; query embedding is about 2.3× faster (77 → 34 ms), batch document embedding barely changes (the CPU is compute-bound). Don't quantize to int8: dynamic quantization of this PhoBERT model on an AVX2-only CPU drops cosine to the fp32 vectors to 0.71-0.84. `token_counter` reads `tokenizer`/`max_seq_length` from `.client` (HuggingFaceEmbeddings) or from the model itself (OnnxEmbeddings).
- **Build data before `docker-up`.** Containers never auto-build unless `AUTO_BUILD_INDEX=true`. The services still start without data, but vector `/health` reports `index_not_ready: ...` and wiki `/health` reports `empty_wiki` (no `wiki_data/wiki/index.md`). Their healthchecks fail, and the gateway's `depends_on: service_healthy` keeps it from ever starting. The wiki service only covers the đề mục that have been built; anything else gets "no page found".
- **Loading the indexes runs pickle.** `FAISS.load_local(..., allow_dangerous_deserialization=True)` and the BM25 pickle execute code on load, so load only indexes built locally.
- **BM25 pickle layout.** `BM25Index`/`BM25Store` keep their old attribute names so existing `bm25_store.pkl` files still load. Renaming an attribute requires rebuilding the vector index.
- **Dependencies are managed with uv.** `pyproject.toml` declares them (lower bounds), `uv.lock` pins them; the Dockerfile runs `uv sync --frozen` into `/opt/venv` (on `PATH`; the image has no pip) and then `uv pip check`. Change a dependency by editing `pyproject.toml` and running `uv lock` on the host, then rebuild; a stale lock fails the build. The extra `embeddings` holds torch, sentence-transformers, onnxruntime and onnx (`INSTALL_EMBEDDINGS=false` leaves it out); `torch` comes only from the explicit `pytorch-cpu` index (`[tool.uv.sources]`), otherwise it pulls CUDA wheels. The `eval` group (ragas) is never synced into the image. `gradio` is capped at `<6` because gradio 6 needs `huggingface-hub>=1.16`, which conflicts with `sentence-transformers==3.0.1`. `langchain-community` has a floor of 0.4.2 because left alone uv resolved 0.3.x.
- **Bit-exact scoring.** Write the RRF term as `weight * (1.0 / (rank + RRF_K))`. `weight / (rank + RRF_K)` differs in the last bit for some ranks, which reorders ties and breaks snapshot comparisons.

## Conventions

- Log, error and print messages are in **unaccented** Vietnamese (`"Dang tai FAISS vector index..."`). LLM prompts and user-facing answer text are in accented Vietnamese.
- Config comes only from `shared/config.get_settings()` (frozen dataclasses, defaults matching `docker-compose.yml`, `.env` loaded if present). Don't add `os.getenv` elsewhere. Algorithm constants (fusion weights, chunk size) are named module constants next to their code, not env vars.
- Library code logs with `logging.getLogger(__name__)`. Services call `shared.log.setup_logging()`. `print` is only for CLI script output.
- Services cache their heavy store (and the reranker, `get_reranker()`) with `@lru_cache(maxsize=1)` and warm it in the FastAPI `lifespan` (`PRELOAD_VECTOR_INDEX`, `PRELOAD_WIKI_CACHE`).
