import os
from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException

from services.wiki_rag_service.store import WikiStore
from shared.http import service_client, timer_ms
from shared.schemas import AskRequest, ChatMessage, ChatRequest, ContextItem, HealthResponse, SearchRequest, ServiceAnswer


app = FastAPI(title="wiki-rag-service", version="0.1.0")

WIKI_DB_PATH = os.getenv("WIKI_DB_PATH", "wiki_db/wiki.db")
RAW_SOURCES_DIR = os.getenv("RAW_SOURCES_DIR", "wiki_data/raw_sources")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8003")
PRELOAD_WIKI_CACHE = os.getenv("PRELOAD_WIKI_CACHE", "true").lower() in {"1", "true", "yes", "on"}


@app.on_event("startup")
def preload_wiki_cache() -> None:
    if not PRELOAD_WIKI_CACHE:
        return
    try:
        with timer_ms() as elapsed:
            result = _wiki_store().warm_cache()
        print(f"Da preload wiki cache {result['article_count']} articles trong {elapsed():.2f} ms")
    except Exception as exc:
        print(f"Khong preload duoc wiki cache: {exc}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = _wiki_store().status()
    return HealthResponse(
        service="wiki-rag-service",
        status="ok" if status["article_count"] else "empty_db",
    )


@app.get("/wiki/status")
def status():
    return _wiki_store().status()


@app.post("/wiki/ingest")
def ingest():
    try:
        return _wiki_store().ingest()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/wiki/query", response_model=List[ContextItem])
def query(request: SearchRequest) -> List[ContextItem]:
    try:
        return _wiki_store().search(request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/wiki/answer", response_model=ServiceAnswer)
def answer(request: AskRequest) -> ServiceAnswer:
    with timer_ms() as elapsed:
        try:
            contexts = query(SearchRequest(query=request.question, top_k=request.top_k, candidate_k=request.candidate_k))
            answer_text, token_usage = _answer_with_llm(request.question, contexts) if request.use_llm else (_extractive_answer(contexts), {})
            return ServiceAnswer(
                service="wiki-rag-service",
                question=request.question,
                answer=answer_text,
                contexts=contexts,
                citations=[context.citation for context in contexts if context.citation],
                latency_ms=elapsed(),
                token_usage=token_usage,
            )
        except Exception as exc:
            return ServiceAnswer(
                service="wiki-rag-service",
                question=request.question,
                answer="",
                latency_ms=elapsed(),
                error=str(exc),
            )


@lru_cache(maxsize=1)
def _wiki_store() -> WikiStore:
    return WikiStore(db_path=WIKI_DB_PATH, raw_sources_dir=RAW_SOURCES_DIR)


def _answer_with_llm(question: str, contexts: List[ContextItem]):
    context_text = "\n\n".join(
        f"[{idx}] {context.title}\nCitation: {context.citation}\n{context.content[:1800]}"
        for idx, context in enumerate(contexts, 1)
    )
    prompt = (
        "Bạn là trợ lý pháp luật Việt Nam dùng wiki Markdown đã chuẩn hóa. "
        "Trả lời chỉ dựa trên context wiki/raw-source được cung cấp. "
        "Luôn trích citation theo số context. Nếu thiếu căn cứ, nói rõ.\n\n"
        f"Câu hỏi: {question}\n\nWiki context:\n{context_text}"
    )
    chat_request = ChatRequest(
        messages=[
            ChatMessage(role="system", content="Bạn trả lời chính xác, có căn cứ, không bịa."),
            ChatMessage(role="user", content=prompt),
        ],
        temperature=0.1,
        max_tokens=1000,
    )
    with service_client() as client:
        response = client.post(f"{LLM_SERVICE_URL}/chat", json=chat_request.model_dump())
        response.raise_for_status()
        data = response.json()
    return data.get("content", ""), data.get("token_usage", {})


def _extractive_answer(contexts: List[ContextItem]) -> str:
    if not contexts:
        return "Không tìm thấy wiki context phù hợp."
    lines = ["Các wiki/raw-source context được truy xuất:"]
    for index, context in enumerate(contexts, 1):
        lines.append(f"[{index}] {context.title} - {context.citation}\n{context.content[:500]}")
    return "\n\n".join(lines)
