"""Luong tra loi dung chung cho vector-rag-service va wiki-rag-service.

Hai service chi khac nhau o ham retrieve va cau chu cua prompt (`PromptStyle`).
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from shared.config import get_settings
from shared.http import service_client, timer_ms
from shared.schemas import AskRequest, ChatMessage, ChatRequest, ContextItem, ServiceAnswer


# (query, top_k, candidate_k) -> contexts
Retriever = Callable[[str, int, int], List[ContextItem]]

LLM_CONTEXT_CHARS = 1800
EXTRACTIVE_CONTEXT_CHARS = 500


@dataclass(frozen=True)
class PromptStyle:
    system: str
    instructions: str
    context_label: str
    no_context: str
    extractive_header: str


VECTOR_PROMPT = PromptStyle(
    system="Bạn trả lời ngắn gọn, đúng căn cứ, không bịa thêm.",
    instructions=(
        "Bạn là trợ lý pháp luật Việt Nam. Trả lời chỉ dựa trên context được cung cấp. "
        "Luôn nêu căn cứ/citation theo số thứ tự context. Nếu không đủ căn cứ, nói rõ."
    ),
    context_label="Context",
    no_context="Không tìm thấy context phù hợp.",
    extractive_header="Các căn cứ được truy xuất:",
)

WIKI_PROMPT = PromptStyle(
    system="Bạn trả lời chính xác, có căn cứ, không bịa.",
    instructions=(
        "Bạn là trợ lý pháp luật Việt Nam dùng wiki Markdown đã chuẩn hóa. "
        "Trả lời chỉ dựa trên context wiki/raw-source được cung cấp. "
        "Luôn trích citation theo số context. Nếu thiếu căn cứ, nói rõ."
    ),
    context_label="Wiki context",
    no_context="Không tìm thấy wiki context phù hợp.",
    extractive_header="Các wiki/raw-source context được truy xuất:",
)


def answer_question(service: str, request: AskRequest, retrieve: Retriever, style: PromptStyle) -> ServiceAnswer:
    """Retrieve roi tra loi; moi loi deu tra ve ServiceAnswer(error=...) thay vi raise."""
    with timer_ms() as elapsed:
        try:
            contexts = retrieve(request.question, request.top_k, request.candidate_k)
            if request.use_llm:
                answer_text, token_usage = _answer_with_llm(request.question, contexts, style)
            else:
                answer_text, token_usage = extractive_answer(contexts, style), {}
        except Exception as exc:
            return ServiceAnswer(
                service=service,
                question=request.question,
                answer="",
                latency_ms=elapsed(),
                error=str(exc),
            )
        return ServiceAnswer(
            service=service,
            question=request.question,
            answer=answer_text,
            contexts=contexts,
            citations=[context.citation for context in contexts if context.citation],
            latency_ms=elapsed(),
            token_usage=token_usage,
        )


def extractive_answer(contexts: List[ContextItem], style: PromptStyle) -> str:
    if not contexts:
        return style.no_context
    lines = [style.extractive_header]
    for index, context in enumerate(contexts, 1):
        lines.append(f"[{index}] {context.title} - {context.citation}\n{context.content[:EXTRACTIVE_CONTEXT_CHARS]}")
    return "\n\n".join(lines)


def build_prompt(question: str, contexts: List[ContextItem], style: PromptStyle) -> List[ChatMessage]:
    context_text = "\n\n".join(
        f"[{index}] {context.title}\nCitation: {context.citation}\n{context.content[:LLM_CONTEXT_CHARS]}"
        for index, context in enumerate(contexts, 1)
    )
    prompt = f"{style.instructions}\n\nCâu hỏi: {question}\n\n{style.context_label}:\n{context_text}"
    return [
        ChatMessage(role="system", content=style.system),
        ChatMessage(role="user", content=prompt),
    ]


def _answer_with_llm(question: str, contexts: List[ContextItem], style: PromptStyle) -> Tuple[str, Dict]:
    chat_request = ChatRequest(messages=build_prompt(question, contexts, style), temperature=0.1, max_tokens=1000)
    with service_client() as client:
        response = client.post(f"{get_settings().services.llm}/chat", json=chat_request.model_dump())
        response.raise_for_status()
        data = response.json()
    return data.get("content", ""), data.get("token_usage", {})
