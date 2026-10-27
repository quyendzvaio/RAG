"""Goi llm-service /chat, dung chung cho luong tra loi va wiki builder."""
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

from shared.config import get_settings
from shared.http import DEFAULT_TIMEOUT
from shared.schemas import ChatMessage, ChatRequest, ChatResponse


logger = logging.getLogger(__name__)

TRUNCATED_RETRY_FACTOR = 2  # bi cat thi goi lai mot lan voi max_tokens gap chung nay


@dataclass
class Completion:
    content: str
    model: str
    token_usage: Dict[str, int] = field(default_factory=dict)  # cong ca lan goi lai
    fallback: bool = False
    error: Optional[str] = None
    truncated: bool = False  # van dung het max_tokens sau khi goi lai


def complete(
    messages: List[ChatMessage],
    max_tokens: int,
    temperature: float = 0.1,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    retry_truncated: bool = True,
    model: Optional[str] = None,
) -> Completion:
    """llm-service khong tra finish_reason: dung het max_tokens nghia la cau tra loi bi cat giua chung.

    `model` rong = model llm-service cau hinh (DEEPSEEK_MODEL); dat de goi model khac qua cung provider.
    """
    request = ChatRequest(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{get_settings().services.llm}/chat", json=request.model_dump())
        response.raise_for_status()
    chat = ChatResponse(**response.json())
    usage = {key: value for key, value in chat.token_usage.items() if isinstance(value, int)}
    truncated = not chat.fallback and usage.get("completion_tokens", 0) >= max_tokens
    if truncated and retry_truncated:
        logger.warning("LLM bi cat o %d token, goi lai voi %d", max_tokens, max_tokens * TRUNCATED_RETRY_FACTOR)
        retry = complete(messages, max_tokens * TRUNCATED_RETRY_FACTOR, temperature, timeout, False, model)
        retry.token_usage = add_usage(usage, retry.token_usage)
        return retry
    return Completion(chat.content, chat.model, usage, chat.fallback, chat.error, truncated)


def add_usage(*usages: Dict) -> Dict[str, int]:
    total: Counter = Counter()
    for usage in usages:
        total.update({key: value for key, value in usage.items() if isinstance(value, int)})
    return dict(total)
