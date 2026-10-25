import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException

from rag.data_loader import article_heading
from rag.query_engine import RankedDocument, get_reranker, hybrid_search
from rag.vector_store import HybridStore, index_readiness, load_hybrid_vectorstore
from shared.answering import VECTOR_PROMPT, answer_question
from shared.config import get_settings
from shared.http import timer_ms
from shared.log import setup_logging
from shared.schemas import AskRequest, ContextItem, HealthResponse, SearchRequest, ServiceAnswer


SERVICE_NAME = "vector-rag-service"

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if get_settings().preload_vector_index:
        _preload()
    yield


app = FastAPI(title=SERVICE_NAME, version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready, reason = index_readiness()
    if not ready:
        return HealthResponse(service=SERVICE_NAME, status=f"index_not_ready: {reason}")
    try:
        _store()
    except Exception as exc:
        return HealthResponse(service=SERVICE_NAME, status=f"index_not_ready: {exc}")
    return HealthResponse(service=SERVICE_NAME)


@app.get("/index/status")
def index_status():
    ready, reason = index_readiness()
    if not ready:
        return {"status": "not_ready", "reason": reason}
    store = _store()
    return {
        "status": "ready",
        "manifest": store.manifest,
        "faiss_ntotal": store.vectorstore.index.ntotal,
        "bm25_docs": store.bm25_store.total_docs,
    }


@app.post("/search", response_model=List[ContextItem])
def search(request: SearchRequest) -> List[ContextItem]:
    try:
        return _retrieve(request.query, request.top_k, request.candidate_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/answer", response_model=ServiceAnswer)
def answer(request: AskRequest) -> ServiceAnswer:
    return answer_question(SERVICE_NAME, request, _retrieve, VECTOR_PROMPT)


@lru_cache(maxsize=1)
def _store() -> HybridStore:
    from rag.embeddings import get_embedding_model

    return load_hybrid_vectorstore(get_embedding_model())


def _preload() -> None:
    ready, reason = index_readiness()
    if not ready:
        logger.info("Bo qua preload vector index: %s", reason)
        return
    try:
        with timer_ms() as elapsed:
            _store()
            get_reranker()
        logger.info("Da preload vector index va reranker trong %.2f ms", elapsed())
    except Exception as exc:
        logger.warning("Khong preload duoc vector index: %s", exc)


def _retrieve(query: str, top_k: int, candidate_k: int) -> List[ContextItem]:
    ranked = hybrid_search(query, _store(), top_k=top_k, candidate_k=candidate_k)
    return [_to_context(item, index) for index, item in enumerate(ranked, 1)]


def _to_context(ranked_doc: RankedDocument, index: int) -> ContextItem:
    metadata = ranked_doc.document.metadata or {}
    return ContextItem(
        id=str(metadata.get("chunk_id", index)),
        source=metadata.get("source", ""),
        title=article_heading(metadata),
        content=ranked_doc.document.page_content,
        metadata=metadata,
        score=ranked_doc.score,
        citation=metadata.get("legal_note") or metadata.get("source") or f"vector-context-{index}",
    )
