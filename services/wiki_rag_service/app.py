import logging
from collections import Counter
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException

from services.wiki_rag_service.wiki_layer import MAX_PAGES, LatestWikiLayer, WikiLayer
from shared.answering import WIKI_PROMPT, answer_question
from shared.config import get_settings
from shared.http import timer_ms
from shared.llm import add_usage
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
    return HealthResponse(service=SERVICE_NAME, status="ok" if wiki_layer().ready else "empty_wiki")


@app.get("/wiki/status")
def status():
    return wiki_layer().status()


@app.post("/wiki/query", response_model=List[ContextItem])
def query(request: SearchRequest) -> List[ContextItem]:
    try:
        contexts, _usage = _retrieve(request.query, request.top_k)
        return contexts
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/wiki/answer", response_model=ServiceAnswer)
def answer(request: AskRequest) -> ServiceAnswer:
    # Chon trang luon can LLM (ke ca use_llm=false: khi do chi bo buoc sinh cau tra loi).
    selection_usage: Counter = Counter()

    def retrieve(query: str, top_k: int, _candidate_k: int) -> List[ContextItem]:
        contexts, usage = _retrieve(query, top_k)
        selection_usage.update(usage)
        return contexts

    result = answer_question(SERVICE_NAME, request, retrieve, WIKI_PROMPT)
    result.token_usage = add_usage(result.token_usage, selection_usage)
    return result


@lru_cache(maxsize=1)
def _latest_layer() -> LatestWikiLayer:
    return LatestWikiLayer(get_settings().paths.wiki_dir)


def wiki_layer() -> WikiLayer:
    return _latest_layer().get()


def _preload() -> None:
    try:
        with timer_ms() as elapsed:
            pages = wiki_layer().status()["page_count"]
        logger.info("Da preload %d trang wiki trong %.2f ms", pages, elapsed())
    except Exception as exc:
        logger.warning("Khong preload duoc lop wiki: %s", exc)


def _retrieve(query: str, top_k: int) -> tuple:
    """(cac trang wiki LLM chon, token dung de chon). Khong co BM25 hay Dieu goc."""
    pages, usage = wiki_layer().select(query, max_pages=min(top_k, MAX_PAGES))
    logger.info("Trang wiki cho '%s': %s", query[:60], [page.path for page in pages])
    return [page.context() for page in pages], usage
