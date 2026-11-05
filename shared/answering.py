"""Luong tra loi dung chung cho vector-rag-service va wiki-rag-service.

Hai service chi khac nhau o ham retrieve va cau chu cua prompt (`PromptStyle`).
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from shared.config import get_settings
from shared.http import timer_ms
from shared.llm import complete
from shared.schemas import AskRequest, ChatMessage, ContextItem, ServiceAnswer
from shared.tracing import annotate, text_version, traceable


# (query, top_k, candidate_k) -> contexts
Retriever = Callable[[str, int, int], List[ContextItem]]

LLM_CONTEXT_CHARS = 6000  # = PARENT_CONTEXT_CHARS cua vector service: gui nguyen Dieu cho LLM
WIKI_CONTEXT_CHARS = 13000  # trang wiki dai nhat ~12300 ky tu (khong tinh danh sach Dieu cuoi trang)
EXTRACTIVE_CONTEXT_CHARS = 500
ANSWER_MAX_TOKENS = 6000  # tran: 1000 cat ngang cau liet ke ho so/dieu kien, model co suy luan (glm) tieu them vai nghin; bi cat thi shared.llm goi lai gap doi
TRUNCATED_NOTE = "\n\n_(Câu trả lời bị cắt do vượt giới hạn độ dài.)_"


@dataclass(frozen=True)
class PromptStyle:
    system: str
    instructions: str
    context_label: str
    no_context: str
    extractive_header: str
    context_chars: int = LLM_CONTEXT_CHARS  # moi context cat toi day truoc khi gui LLM


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
        "Bạn là trợ lý pháp luật Việt Nam. Trả lời chỉ dựa trên các trang wiki pháp luật được cung cấp "
        "(trang tổng hợp nhiều Điều, mỗi ý có mã [Điều ...]). Nêu căn cứ theo số trang và giữ nguyên mã "
        "[Điều ...] mà trang trích. Nếu các trang không đủ căn cứ, nói rõ."
    ),
    context_label="Trang wiki",
    no_context="Không tìm thấy trang wiki phù hợp với câu hỏi.",
    extractive_header="Các trang wiki được chọn:",
    context_chars=WIKI_CONTEXT_CHARS,
)


def _trace_inputs(inputs: dict) -> dict:
    return {"service": inputs["service"], **inputs["request"].model_dump()}


def _trace_outputs(answer: ServiceAnswer) -> dict:
    # Context day du da nam o run retriever ben duoi; o day chi giu phan de doc nhanh.
    return {
        "answer": answer.answer,
        "citations": answer.citations,
        "context_count": len(answer.contexts),
        "token_usage": answer.token_usage,
        "latency_ms": answer.latency_ms,
        "error": answer.error,
    }


@traceable(name="answer_question", process_inputs=_trace_inputs, process_outputs=_trace_outputs)
def answer_question(service: str, request: AskRequest, retrieve: Retriever, style: PromptStyle) -> ServiceAnswer:
    """Retrieve roi tra loi; moi loi deu tra ve ServiceAnswer(error=...) thay vi raise."""
    annotate(
        service=service,
        prompt_version=text_version(style.system, style.instructions, style.context_label),
        llm_model=get_settings().deepseek.model,
        git_commit=get_settings().tracing.git_commit,
    )
    with timer_ms() as elapsed:
        try:
            contexts = retrieve(request.question, request.top_k, request.candidate_k)
            if not contexts:
                answer_text, token_usage = style.no_context, {}
            elif request.use_llm:
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


def format_context(context: ContextItem, chars: int) -> str:
    """Mot context dung nhu LLM tra loi thay: tieu de, Citation (so hieu, ngay), noi dung cat o `chars`."""
    return f"{context.title}\nCitation: {context.citation}\n{context.content[:chars]}"


def build_prompt(question: str, contexts: List[ContextItem], style: PromptStyle) -> List[ChatMessage]:
    context_text = "\n\n".join(
        f"[{index}] {format_context(context, style.context_chars)}" for index, context in enumerate(contexts, 1)
    )
    prompt = f"{style.instructions}\n\nCâu hỏi: {question}\n\n{style.context_label}:\n{context_text}"
    return [
        ChatMessage(role="system", content=style.system),
        ChatMessage(role="user", content=prompt),
    ]


def _answer_with_llm(question: str, contexts: List[ContextItem], style: PromptStyle) -> Tuple[str, Dict]:
    completion = complete(build_prompt(question, contexts, style), ANSWER_MAX_TOKENS)
    answer = completion.content + (TRUNCATED_NOTE if completion.truncated else "")
    return answer, completion.token_usage
