import os
from typing import Dict

import httpx
from fastapi import FastAPI

from rag.settings import deepseek_settings
from shared.http import DEFAULT_TIMEOUT, timer_ms
from shared.schemas import ChatRequest, ChatResponse, HealthResponse


app = FastAPI(title="llm-service", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = deepseek_settings()
    return HealthResponse(service="llm-service", status="ok" if settings.configured else "missing_api_key")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    settings = deepseek_settings()
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
                response = client.post(f"{settings.base_url.rstrip('/')}/v1/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            return ChatResponse(
                model=model,
                content=_fallback_response(request),
                latency_ms=elapsed(),
                fallback=True,
                error=str(exc),
            )

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage: Dict = data.get("usage", {})
    return ChatResponse(
        model=data.get("model", model),
        content=message.get("content", ""),
        token_usage=usage,
        latency_ms=elapsed(),
    )


def _fallback_response(request: ChatRequest) -> str:
    user_messages = [message.content for message in request.messages if message.role == "user"]
    last = user_messages[-1] if user_messages else ""
    return (
        "LLM fallback: DeepSeek API key chua duoc cau hinh hoac request bi loi. "
        "Day la phan tra loi tam de kiem tra pipeline.\n\n"
        f"Prompt cuoi:\n{last[:1200]}"
    )
