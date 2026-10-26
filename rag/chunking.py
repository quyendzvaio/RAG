"""Chunking parent-child: parent = mot Dieu (hoac Phu luc), child = cum khoan vua cua so embedding.

Child duoc dung de embed va tim kiem; parent la thu tra ve cho LLM. Moi child bat dau bang tieu de Dieu
de embedding biet doan van thuoc Dieu nao. Ranh gioi cat theo cau truc phap ly: khoan ("1."),
roi diem ("a)"), roi dong, roi cau, cuoi cung moi cat theo tu.
"""
import re
from typing import Callable, List

from langchain_core.documents import Document

from rag.data_loader import article_heading


TokenCounter = Callable[[str], int]

CHILD_TARGET_TOKENS = 200  # gom cac khoan ngan lien nhau toi muc nay (tinh ca tieu de)
MIN_BODY_TOKENS = 64  # tieu de dai van chua lai it nhat chung nay token cho noi dung
LEAD_IN_MAX_TOKENS = 60  # cau dan cua khoan ("2. Hồ sơ gồm:") duoc lap lai o dau moi diem neu ngan
SPECIAL_TOKENS = 2  # <s> va </s> cua tokenizer PhoBERT
KHOAN_RE = re.compile(r"^\d+[a-z]?\.\s")
DIEM_RE = re.compile(r"^[a-zđ]\)\s")
SENTENCE_END_RE = re.compile(r"(?<=[.;:!?])\s+")


def split_parent_child(parents: List[Document], count_tokens: TokenCounter, max_tokens: int) -> List[Document]:
    """Tao child cho moi parent. `max_tokens` la max_seq_length cua embedding model."""
    children: List[Document] = []
    for parent in parents:
        header = " ".join(article_heading(parent.metadata).split())
        header_tokens = count_tokens(header + "\n")
        budget = max(max_tokens - SPECIAL_TOKENS - header_tokens, MIN_BODY_TOKENS)
        target = max(min(CHILD_TARGET_TOKENS - header_tokens, budget), MIN_BODY_TOKENS)
        bodies = _pack(_units(parent.page_content, budget, count_tokens), target, budget, count_tokens)
        for child_index, body in enumerate(bodies):
            children.append(
                Document(
                    page_content=f"{header}\n{body}" if header else body,
                    metadata={
                        "chunk_id": len(children),
                        "parent_id": parent.metadata["parent_id"],
                        "child_index": child_index,
                        "source_file": parent.metadata.get("source_file", ""),
                        "article_code": parent.metadata.get("article_code", ""),
                    },
                )
            )
    return children


def child_body(child: Document) -> str:
    """Noi dung child khong co dong tieu de."""
    return child.page_content.split("\n", 1)[1] if "\n" in child.page_content else child.page_content


def _units(text: str, budget: int, count_tokens: TokenCounter) -> List[str]:
    """Chia noi dung parent thanh cac don vi <= budget token, uu tien ranh gioi khoan."""
    units: List[str] = []
    for block in _blocks(text.split("\n"), KHOAN_RE):
        if count_tokens(block) <= budget:
            units.append(block)
            continue
        pieces = _blocks(block.split("\n"), DIEM_RE)
        lead = pieces[0] if len(pieces) > 1 and count_tokens(pieces[0]) <= LEAD_IN_MAX_TOKENS else ""
        for position, piece in enumerate(pieces):
            if lead and position > 0:
                piece = f"{lead}\n{piece}"
            elif lead:
                continue  # cau dan da duoc lap lai o dau moi diem
            units.extend(_fit(piece, budget, count_tokens))
    return units


def _blocks(lines: List[str], start_re: re.Pattern) -> List[str]:
    """Gom dong thanh khoi; mot khoi moi bat dau o dong khop `start_re`."""
    blocks: List[List[str]] = []
    for line in lines:
        if not line:
            continue
        if not blocks or start_re.match(line):
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return ["\n".join(block) for block in blocks]


def _fit(text: str, budget: int, count_tokens: TokenCounter) -> List[str]:
    """Cat `text` thanh cac doan <= budget: theo dong, roi theo cau, roi theo tu."""
    if count_tokens(text) <= budget:
        return [text]
    for pattern, joiner in (("\n", "\n"), (SENTENCE_END_RE, " ")):
        parts = text.split(pattern) if isinstance(pattern, str) else pattern.split(text)
        if len(parts) > 1:
            pieces = [piece for part in parts for piece in _fit(part, budget, count_tokens)]
            return _pack(pieces, budget, budget, count_tokens, joiner)
    words = text.split()
    if len(words) <= 1:
        return [text]  # mot "tu" qua dai: de tokenizer tu cat
    middle = len(words) // 2
    return _fit(" ".join(words[:middle]), budget, count_tokens) + _fit(" ".join(words[middle:]), budget, count_tokens)


def _pack(units: List[str], target: int, budget: int, count_tokens: TokenCounter, joiner: str = "\n") -> List[str]:
    """Gom cac don vi lien nhau thanh child toi `target` token, khong bao gio vuot `budget`."""
    packed: List[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue
        candidate = f"{current}{joiner}{unit}"
        if count_tokens(candidate) <= target:
            current = candidate
        else:
            packed.append(current)
            current = unit
    if current:
        packed.append(current)
    return packed
