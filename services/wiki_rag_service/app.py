import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException

from services.wiki_rag_service.store import WikiStore
from shared.answering import WIKI_PROMPT, answer_question
from shared.config import get_settings
from shared.http import timer_ms
from shared.log import setup_logging
from shared.schemas import AskRequest, ContextItem, HealthResponse, SearchRequest, ServiceAnswer


SERVICE_NAME = "wiki-rag-service"

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if get_settings().preload_wiki_cache:
        _preload()
    yield


app = FastAPI(title=SERVICE_NAME, version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = wiki_store().status()
    return HealthResponse(service=SERVICE_NAME, status="ok" if status["article_count"] else "empty_db")


@app.get("/wiki/status")
def status():
    return wiki_store().status()


@app.post("/wiki/ingest")
def ingest():
    try:
        return wiki_store().ingest()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/wiki/query", response_model=List[ContextItem])
def query(request: SearchRequest) -> List[ContextItem]:
    try:
        return _retrieve(request.query, request.top_k, request.candidate_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/wiki/answer", response_model=ServiceAnswer)
def answer(request: AskRequest) -> ServiceAnswer:
    return answer_question(SERVICE_NAME, request, _retrieve, WIKI_PROMPT)


@lru_cache(maxsize=1)
def wiki_store() -> WikiStore:
    paths = get_settings().paths
    return WikiStore(db_path=paths.wiki_db, raw_sources_dir=paths.raw_sources_dir)


def _preload() -> None:
    try:
        with timer_ms() as elapsed:
            result = wiki_store().warm_cache()
        logger.info("Da preload wiki cache %d articles trong %.2f ms", result["article_count"], elapsed())
    except Exception as exc:
        logger.warning("Khong preload duoc wiki cache: %s", exc)


def _retrieve(query: str, top_k: int, _candidate_k: int) -> List[ContextItem]:
    # Wiki chi xep hang BM25 mot lan, khong co buoc rerank nen bo qua candidate_k.
    return wiki_store().search(query, top_k=top_k)
