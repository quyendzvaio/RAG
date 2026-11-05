import json
import logging
import time
from typing import Dict

import httpx
from fastapi import FastAPI

from shared.config import get_settings
from shared.http import DEFAULT_TIMEOUT, timer_ms
from shared.log import setup_logging
from shared.schemas import ChatRequest, ChatResponse, HealthResponse
from shared.tracing import add_tracing, get_current_run_tree, traceable, tracing_context


setup_logging()
logger = logging.getLogger(__name__)
app = FastAPI(title="llm-service", version="0.1.0")
add_tracing(app)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings().deepseek
    return HealthResponse(service="llm-service", status="ok" if settings.configured else "missing_api_key")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings().deepseek
    model = request.model or settings.model
    with timer_ms() as elapsed:
        if not settings.configured:
            return ChatResponse(
                model=model,
                content=_fallback_response(request),
                latency_ms=elapsed(),
                fallback=True,
                error="DEEPSEEK_API_KEY is not configured.",
            )

        payload = {
            "stream": False,  # mot so provider qua 9router (vd. mmg/) mac dinh tra SSE khi thieu truong nay
            **settings.extra_body,
            "model": model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        # Chi trace khi request thuoc mot trace (header tu service goi den): loi goi le nhu wiki builder hay giam
        # khao RAGAS khong tao trace rieng cho tung lan goi.
        traced = get_current_run_tree() is not None
        try:
            with tracing_context(enabled=None if traced else False):
                data = _chat_with_retry(payload, model)
        except Exception as exc:
            return ChatResponse(
                model=model,
                content=_fallback_response(request),
                latency_ms=elapsed(),
                fallback=True,
                error=_error_text(exc),
            )

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage: Dict = data.get("usage", {})
    return ChatResponse(
        model=data.get("model", model),
        content=message.get("content") or "",
        token_usage=usage,
        latency_ms=elapsed(),
    )


# Loi tam thoi cua gateway phia provider (vd. kiraai cat request o giay thu 60 bang 504): goi lai mot lan.
# 4xx (402 het tien, 429 gioi han) khong goi lai: lap lai cung khong qua duoc.
RETRY_STATUSES = {502, 503, 504}
RETRY_DELAY_SECONDS = 2.0


def _chat_with_retry(payload: Dict, model: str) -> Dict:
    try:
        return _chat_completion(payload, langsmith_extra={"metadata": {"ls_model_name": model}})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in RETRY_STATUSES:
            raise
        logger.warning("Provider tra %d, goi lai sau %.0f s", exc.response.status_code, RETRY_DELAY_SECONDS)
    time.sleep(RETRY_DELAY_SECONDS)
    return _chat_completion(payload, langsmith_extra={"metadata": {"ls_model_name": model, "retry": 1}})


@traceable(
    run_type="llm",
    name="chat_completions",
    process_inputs=lambda inputs: inputs["payload"],
    process_outputs=lambda data: {**data, "usage_metadata": _usage_metadata(data.get("usage") or {})},
)
def _chat_completion(payload: Dict) -> Dict:
    """Mot lan goi provider; API key chi nam trong header nen khong vao input cua trace."""
    settings = get_settings().deepseek
    headers = {"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        response = client.post(settings.chat_completions_url, json=payload, headers=headers)
        response.raise_for_status()
        return _parse_completion(response.text)


def _usage_metadata(usage: Dict) -> Dict[str, int]:
    """Dang token LangSmith doc de cong token/chi phi cua trace."""
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _parse_completion(text: str) -> Dict:
    """JSON OpenAI; mot so gateway (vd. cl/ qua 9router) boc trong {"data": {...}} va them dong "data: [DONE]"."""
    data, _end = json.JSONDecoder().raw_decode(text.lstrip())
    return data["data"] if "choices" not in data and isinstance(data.get("data"), dict) else data


def _error_text(exc: Exception) -> str:
    """Kem body loi cua provider (vd. 429 FreeUsageLimitError, 402 het so du) de client phan biet loi tam thoi."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{exc}\n{exc.response.text[:500]}"
    return str(exc)


def _fallback_response(request: ChatRequest) -> str:
    user_messages = [message.content for message in request.messages if message.role == "user"]
    last = user_messages[-1] if user_messages else ""
    return (
        "LLM fallback: DeepSeek API key chua duoc cau hinh hoac request bi loi. "
        "Day la phan tra loi tam de kiem tra pipeline.\n\n"
        f"Prompt cuoi:\n{last[:1200]}"
    )
