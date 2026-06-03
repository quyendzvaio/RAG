import os
from functools import lru_cache
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException

from rag.query_engine import hybrid_search
from rag.vector_store import BM25_PATH, FAISS_DIR, HybridStore, MANIFEST_PATH, load_hybrid_vectorstore
from shared.http import service_client, timer_ms
from shared.schemas import AskRequest, ChatMessage, ChatRequest, ContextItem, HealthResponse, SearchRequest, ServiceAnswer


app = FastAPI(title="vector-rag-service", version="0.1.0")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8003")
PRELOAD_VECTOR_INDEX = os.getenv("PRELOAD_VECTOR_INDEX", "true").lower() in {"1", "true", "yes", "on"}


@app.on_event("startup")
def preload_index() -> None:
    if not PRELOAD_VECTOR_INDEX:
        return
    ready, reason = _index_readiness()
    if not ready:
        print(f"Bo qua preload vector index: {reason}")
        return
    try:
        print("Dang preload vector index de request dau tien khong bi tre...")
        with timer_ms() as elapsed:
            _store()
        print(f"Da preload vector index trong {elapsed():.2f} ms")
    except Exception as exc:
        print(f"Khong preload duoc vector index: {exc}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready, reason = _index_readiness()
    if not ready:
        return HealthResponse(service="vector-rag-service", status=f"index_not_ready: {reason}")
    try:
        _store()
    except Exception as exc:
        return HealthResponse(service="vector-rag-service", status=f"index_not_ready: {exc}")
    return HealthResponse(service="vector-rag-service")


@app.get("/index/status")
def index_status():
    ready, reason = _index_readiness()
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
        ranked = hybrid_search(request.query, _store(), top_k=request.top_k, candidate_k=request.candidate_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [_context_from_ranked(item, index) for index, item in enumerate(ranked, 1)]


@app.post("/answer", response_model=ServiceAnswer)
def answer(request: AskRequest) -> ServiceAnswer:
    with timer_ms() as elapsed:
        try:
            contexts = search(SearchRequest(query=request.question, top_k=request.top_k, candidate_k=request.candidate_k))
            answer_text, token_usage = _answer_with_llm(request.question, contexts) if request.use_llm else (_extractive_answer(contexts), {})
            return ServiceAnswer(
                service="vector-rag-service",
                question=request.question,
                answer=answer_text,
                contexts=contexts,
                citations=[context.citation for context in contexts if context.citation],
                latency_ms=elapsed(),
                token_usage=token_usage,
            )
        except Exception as exc:
            return ServiceAnswer(
                service="vector-rag-service",
                question=request.question,
                answer="",
                latency_ms=elapsed(),
                error=str(exc),
            )


@lru_cache(maxsize=1)
def _store() -> HybridStore:
    ready, reason = _index_readiness()
    if not ready:
        raise RuntimeError(f"Hybrid index chua san sang: {reason}. Hay chay: make docker-build-vector-index")
    from rag.embeddings import get_embedding_model

    embedding_model = get_embedding_model()
    store = load_hybrid_vectorstore(embedding_model)
    if store is None:
        raise RuntimeError("Hybrid index chua san sang. Hay build index truoc.")
    return store


def _index_readiness():
    checks = [
        (Path(FAISS_DIR), "thieu thu muc faiss_index"),
        (Path(FAISS_DIR) / "index.faiss", "thieu faiss_index/index.faiss"),
        (Path(FAISS_DIR) / "index.pkl", "thieu faiss_index/index.pkl"),
        (Path(BM25_PATH), "thieu hybrid_index/bm25_store.pkl"),
        (Path(MANIFEST_PATH), "thieu hybrid_index/index_manifest.json"),
    ]
    for path, reason in checks:
        if not path.exists():
            return False, reason
    return True, "ready"


def _context_from_ranked(ranked_doc, index: int) -> ContextItem:
    doc = ranked_doc.document
    metadata = doc.metadata or {}
    title = " ".join(
        value
        for value in (
            metadata.get("article_code", ""),
            metadata.get("article_title", ""),
        )
        if value
    ).strip()
    citation = metadata.get("legal_note") or metadata.get("source") or f"vector-context-{index}"
    return ContextItem(
        id=str(metadata.get("chunk_id", index)),
        source=metadata.get("source", ""),
        title=title,
        content=doc.page_content,
        metadata=metadata,
        score=ranked_doc.score,
        citation=citation,
    )


def _answer_with_llm(question: str, contexts: List[ContextItem]):
    context_text = "\n\n".join(
        f"[{idx}] {context.title}\nCitation: {context.citation}\n{context.content[:1800]}"
        for idx, context in enumerate(contexts, 1)
    )
    prompt = (
        "Bạn là trợ lý pháp luật Việt Nam. Trả lời chỉ dựa trên context được cung cấp. "
        "Luôn nêu căn cứ/citation theo số thứ tự context. Nếu không đủ căn cứ, nói rõ.\n\n"
        f"Câu hỏi: {question}\n\nContext:\n{context_text}"
    )
    chat_request = ChatRequest(
        messages=[
            ChatMessage(role="system", content="Bạn trả lời ngắn gọn, đúng căn cứ, không bịa thêm."),
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
        return "Không tìm thấy context phù hợp."
    lines = ["Các căn cứ được truy xuất:"]
    for index, context in enumerate(contexts, 1):
        lines.append(f"[{index}] {context.title} - {context.citation}\n{context.content[:500]}")
    return "\n\n".join(lines)
