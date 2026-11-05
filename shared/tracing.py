"""Tracing LangSmith cho luong hoi dap; tat mac dinh, bat bang LANGSMITH_TRACING=true va LANGSMITH_API_KEY.

SDK `langsmith` tu doc cac bien LANGSMITH_* (get_settings() da nap .env). Moi ham gan `@traceable` thanh mot run;
khi tracing tat, decorator chi goi thang ham. Mot cau hoi di qua nhieu service van la mot trace: client gan header
`langsmith-trace`/`baggage` cua run hien tai (`trace_headers`, qua `shared.http.service_client`), service nhan
khoi phuc run cha bang `TracingMiddleware` (`add_tracing`).
"""
import hashlib
from typing import Any, Dict

from langsmith import traceable  # noqa: F401  (dung lai qua module nay)
from langsmith.middleware import TracingMiddleware
from langsmith.run_helpers import get_current_run_tree, tracing_context  # noqa: F401
from langsmith.utils import ContextThreadPoolExecutor  # noqa: F401  (luong con giu run cha)
from langsmith.uuid import uuid7  # noqa: F401  (id trace LangSmith dung UUID v7)


def add_tracing(app) -> None:
    """Noi tiep trace cua service goi den (header `langsmith-trace`); khong co header thi khong lam gi."""
    app.add_middleware(TracingMiddleware)


def trace_headers() -> Dict[str, str]:
    run = get_current_run_tree()
    return run.to_headers() if run is not None else {}


def annotate(**metadata: Any) -> None:
    """Them metadata vao run hien tai (vd. phien ban prompt, index) de loc trace tren LangSmith."""
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata({key: value for key, value in metadata.items() if value is not None})


def text_version(*texts: str) -> str:
    """Ma ngan cua noi dung prompt: doi chu nao trong prompt thi ma doi, khong can tu tang so phien ban."""
    return hashlib.sha1("\x1f".join(texts).encode("utf-8")).hexdigest()[:8]


def as_documents(contexts) -> Dict[str, list]:
    """Output cua run `retriever` theo dang LangSmith hien thi (page_content + metadata)."""
    return {
        "documents": [
            {
                "type": "Document",
                "page_content": context.content[:2000],
                "metadata": {"title": context.title, "citation": context.citation, "score": context.score,
                             "id": context.id},
            }
            for context in contexts
        ]
    }
