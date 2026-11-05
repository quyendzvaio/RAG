"""Viet lai cau hoi truoc khi truy xuat: nguoi dung hay go thieu dau, ma FAISS va BM25 coi "thue" va "thuế" la hai tu
khac nhau (golden khong dau: Recall@5 0.147; qwen3-14b them dau roi truy xuat: 0.900, cau co dau goc: 0.953).

Chi dung cho truy xuat; cau tra loi van dua cau hoi goc cho LLM. LLM loi hoac tra ve cau khac han thi dung cau goc.
"""
import logging
import re
import unicodedata
from functools import lru_cache
from typing import Dict, Tuple

from shared.llm import complete
from shared.schemas import ChatMessage
from shared.tracing import traceable


logger = logging.getLogger(__name__)

# Tran, khong phai muc dung: model co suy luan khong tat duoc (qwen3.5-flash-sale qua kiraai) nghi ~700-1000 token
# truoc khi in cau; 200 token tra ve noi dung rong va bo qua buoc them dau.
REWRITE_MAX_TOKENS = 1500
MAX_WORD_CHANGE = 0.3  # so tu lech qua ty le nay (LLM tra loi hoac dien dat lai) thi bo ban viet lai
REWRITE_SYSTEM = (
    "Bạn chuẩn hóa câu hỏi pháp luật tiếng Việt trước khi tra cứu văn bản.\n"
    "Viết lại cần chú ý một số từ gửi vào có thể bị thiếu dấu tiếng Việt: thêm đúng dấu cho các từ đó.\n"
    "Giữ nguyên từng từ và thứ tự, không thêm, bớt hay diễn đạt lại, không trả lời câu hỏi. "
    "Câu đã đủ dấu thì chép lại nguyên văn. Chỉ in ra câu đã viết lại."
)
WORD_RE = re.compile(r"[\wÀ-ỹ]+")


@traceable(name="rewrite_query", process_outputs=lambda result: {"query": result[0], "token_usage": result[1]})
def rewrite_query(question: str) -> Tuple[str, Dict[str, int]]:
    """(cau dung de truy xuat, token LLM da dung cho lan nay). Cau da gap thi lay tu cache, 0 token."""
    misses = _rewrite.cache_info().misses
    rewritten, usage = _rewrite(question)
    return rewritten, usage if _rewrite.cache_info().misses > misses else {}


@lru_cache(maxsize=1024)
def _rewrite(question: str) -> Tuple[str, Dict[str, int]]:
    try:
        completion = complete(
            [ChatMessage(role="system", content=REWRITE_SYSTEM), ChatMessage(role="user", content=question)],
            REWRITE_MAX_TOKENS,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("Khong viet lai duoc cau hoi, dung cau goc: %s", exc)
        return question, {}
    if completion.fallback or not completion.content.strip():
        logger.warning("Khong viet lai duoc cau hoi, dung cau goc: %s", completion.error)
        return question, completion.token_usage
    rewritten = completion.content.strip().splitlines()[0].strip().strip('"')
    if _word_change(question, rewritten) > MAX_WORD_CHANGE:
        logger.warning("Bo cau viet lai lech qua nhieu: %r -> %r", question, rewritten)
        return question, completion.token_usage
    if rewritten != question:
        logger.info("Viet lai cau hoi: %r -> %r", question, rewritten)
    return rewritten, completion.token_usage


def _word_change(original: str, rewritten: str) -> float:
    """Ty le tu khac nhau sau khi bo dau (them dau dung thi bang 0)."""
    before, after = WORD_RE.findall(_strip_marks(original)), WORD_RE.findall(_strip_marks(rewritten))
    same = sum(1 for a, b in zip(before, after) if a == b)
    return 1 - same / max(len(before), len(after), 1)


def _strip_marks(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(char for char in text if unicodedata.category(char) != "Mn")
