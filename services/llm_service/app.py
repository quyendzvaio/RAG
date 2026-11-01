import json
from typing import Dict

import httpx
from fastapi import FastAPI

from shared.config import get_settings
from shared.http import DEFAULT_TIMEOUT, timer_ms
from shared.log import setup_logging
from shared.schemas import ChatRequest, ChatResponse, HealthResponse


setup_logging()
app = FastAPI(title="llm-service", version="0.1.0")


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
        headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
                response = client.post(settings.chat_completions_url, json=payload, headers=headers)
                response.raise_for_status()
                data = _parse_completion(response.text)
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
