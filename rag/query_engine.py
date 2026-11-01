"""Hybrid search cho vector-rag-service, o cap parent (Dieu/Phu luc).

FAISS tim child -> gom ve parent (diem cua child tot nhat); BM25 cham truc tiep parent;
reciprocal rank fusion -> rerank tren doan context se gui cho LLM.
"""
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import chain
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from langchain_core.documents import Document

from rag.bm25_store import document_key, tokenize
from rag.chunking import child_body
from rag.device import choose_torch_device
from rag.vector_store import HybridStore
from shared.config import RerankerSettings, get_settings


logger = logging.getLogger(__name__)

VECTOR_WEIGHT = 0.55
BM25_WEIGHT = 0.45
RRF_K = 60  # hang so cua reciprocal rank fusion: weight / (rank + RRF_K)
RERANK_WEIGHT = 0.65  # diem cuoi = RERANK_WEIGHT * rerank chuan hoa + phan con lai * fusion chuan hoa
CHILD_FANOUT = 4  # FAISS tra ve candidate_k * CHILD_FANOUT child truoc khi gom theo parent
PARENT_CONTEXT_CHARS = 6000  # parent dai hon thi context chi gom cac child lien quan
RERANK_METADATA_KEYS = (
    "chapter",
    "chapter_title",
    "article_code",
    "article_title",
    "legal_note",
    "doc_type",
    "effective_date",
)
# Viet tat thuong gap trong van ban: cau hoi viet tat hay viet day du deu them dang con lai (cho BM25 va rerank,
# khong cho FAISS). Vd. Dieu "Số và thời hạn sử dụng của CMND" voi cau hoi "chứng minh nhân dân ...".
ABBREVIATIONS = {
    "CMND": "chứng minh nhân dân",
    "CCCD": "căn cước công dân",
    "UBND": "ủy ban nhân dân",
    "HĐND": "hội đồng nhân dân",
    "BHXH": "bảo hiểm xã hội",
    "BHYT": "bảo hiểm y tế",
    "BHTN": "bảo hiểm thất nghiệp",
    "TNDN": "thu nhập doanh nghiệp",
    "TNCN": "thu nhập cá nhân",
    "GTGT": "giá trị gia tăng",
    "QNCN": "quân nhân chuyên nghiệp",
    "HKDD": "hàng không dân dụng",
    "NNT": "người nộp thuế",
    "NHNN": "ngân hàng nhà nước",
    "NSNN": "ngân sách nhà nước",
    "CNTT": "công nghệ thông tin",
    "PCCC": "phòng cháy chữa cháy",
    "TTHC": "thủ tục hành chính",
    "GDNN": "giáo dục nghề nghiệp",
    "ĐVSNCL": "đơn vị sự nghiệp công lập",
}
# Cau hoi hoi ve "ai phai tuan theo", "pham vi", "trach nhiem thi hanh", "hieu luc" -> Dieu mo dau/cuoi van ban
# (tieu de chung chung, noi dung khong nhac lai chu de nen thua cac Dieu cu the). Cong INTENT_BOOST vao diem cuoi
# (thang [0, 1]) cho Dieu co tieu de dung loai. Golden: Recall@5 0.913 -> 0.953 (cung voi ABBREVIATIONS), khong cau nao mat.
INTENT_BOOST = 0.35
ARTICLE_INTENTS = (
    (r"\b(những ai|ai phải|đối tượng nào|áp dụng (đối với|cho) (ai|những ai|đối tượng nào))", r"đối tượng áp dụng"),
    (
        r"(phạm vi điều chỉnh|những nội dung nào được (hướng dẫn|quy định)|bao gồm những nội dung"
        r"|quy định (về )?những (nội dung|vấn đề) (gì|nào))",
        r"phạm vi (điều chỉnh|áp dụng)",
    ),
    (r"(trách nhiệm thi hành|chịu trách nhiệm (tổ chức )?thi hành)", r"(trách nhiệm thi hành|tổ chức thực hiện)"),
    (r"có hiệu lực (thi hành )?(từ|kể từ)? ?(ngày nào|khi nào)", r"hiệu lực"),
)
_INTENT_PATTERNS = [(re.compile(question, re.I), re.compile(title, re.I)) for question, title in ARTICLE_INTENTS]


@dataclass
class RankedDocument:
    document: Document  # parent
    score: float
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    matched_children: List[int] = field(default_factory=list)  # child_index trung vector, tot nhat truoc
    context: str = ""  # doan noi dung parent tra ve cho LLM


class CrossEncoderReranker:
    """CrossEncoder neu cau hinh RERANKER_MODEL va tai duoc; neu khong `score` tra ve None."""

    def __init__(self, config: RerankerSettings):
        self.model_name = config.model
        self.batch_size = config.batch_size
        self.model = None
        if not self.model_name:
            return
        try:
            from sentence_transformers import CrossEncoder

            choice = choose_torch_device(config.device, "RERANKER_DEVICE", get_settings().force_cuda)
            logger.info("Reranker device: %s (%s)", choice.device, choice.reason)
            self.model = CrossEncoder(self.model_name, device=choice.device)
            logger.info("Da tai neural reranker: %s", self.model_name)
        except Exception as exc:
            logger.warning("Khong tai duoc neural reranker '%s', dung lexical rerank. Loi: %s", self.model_name, exc)
            self.model = None

    def score(self, query: str, texts: List[str]) -> Optional[List[float]]:
        if self.model is None or not texts:
            return None
        pairs = [(query, text) for text in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        return [float(score) for score in scores]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(get_settings().reranker)


def hybrid_search(question: str, store: HybridStore, top_k: int = 5, candidate_k: int = 40) -> List[RankedDocument]:
    vector_hits, bm25_hits, matched = candidate_hits(question, store, candidate_k)

    candidates: Dict[str, RankedDocument] = {}
    _merge_ranked_hits(candidates, vector_hits, "vector", VECTOR_WEIGHT)
    _merge_ranked_hits(candidates, bm25_hits, "bm25", BM25_WEIGHT)
    for key, candidate in candidates.items():
        candidate.matched_children = matched.get(key, [])

    fused = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
    return boost_article_intent(question, rerank(question, store, fused[:candidate_k]))[:top_k]


def expand_abbreviations(question: str) -> str:
    """Them dang day du cua viet tat co trong cau hoi va viet tat cua cum day du co trong cau hoi."""
    lowered = question.lower().replace("uỷ", "ủy")
    extra = [full for short, full in ABBREVIATIONS.items() if re.search(rf"\b{short}\b", question)]
    extra += [short for short, full in ABBREVIATIONS.items() if full in lowered]
    return f"{question} {' '.join(extra)}" if extra else question


def boost_article_intent(question: str, ranked: List[RankedDocument]) -> List[RankedDocument]:
    titles = [title for pattern, title in _INTENT_PATTERNS if pattern.search(question)]
    if not titles:
        return ranked
    for candidate in ranked:
        if any(title.search(candidate.document.metadata.get("article_title", "")) for title in titles):
            candidate.score += INTENT_BOOST
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def candidate_hits(
    question: str, store: HybridStore, candidate_k: int
) -> Tuple[List[Tuple[Document, float]], List[Tuple[Document, float]], Dict[str, List[int]]]:
    """(parent theo vector, parent theo BM25, parent_id -> child_index trung vector)."""
    child_hits = _vector_search(store, question, candidate_k * CHILD_FANOUT)
    vector_hits: List[Tuple[Document, float]] = []
    matched: Dict[str, List[int]] = {}
    for child, distance in child_hits:
        parent_id = child.metadata["parent_id"]
        if parent_id not in matched:
            matched[parent_id] = []
            vector_hits.append((store.parents[parent_id], distance))
        matched[parent_id].append(child.metadata["child_index"])
    bm25_hits = store.bm25_store.search_with_scores(expand_abbreviations(question), top_k=candidate_k)
    return vector_hits[:candidate_k], bm25_hits, matched


def rerank(question: str, store: HybridStore, candidates: List[RankedDocument]) -> List[RankedDocument]:
    if not candidates:
        return []

    question = expand_abbreviations(question)
    for candidate in candidates:
        candidate.context = context_text(question, store, candidate)
    texts = [_rerank_text(candidate.document.metadata, candidate.context) for candidate in candidates]
    scores = get_reranker().score(question, texts)
    if scores is None:
        scores = [
            _lexical_rerank_score(question, text, candidate.document.metadata)
            for text, candidate in zip(texts, candidates)
        ]

    # Ca hai diem deu min-max ve [0, 1] tren tap ung vien: diem RRF tho chi ~0.01, khong chuan hoa
    # thi phan fusion gan nhu khong anh huong thu tu.
    fused = _min_max([candidate.score for candidate in candidates])
    for candidate, score, normalized, fused_score in zip(candidates, scores, _min_max(scores), fused):
        candidate.rerank_score = score
        candidate.score = RERANK_WEIGHT * normalized + (1 - RERANK_WEIGHT) * fused_score

    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _min_max(values: List[float]) -> List[float]:
    low = min(values)
    spread = max(max(values) - low, 1e-9)
    return [(value - low) / spread for value in values]


def _vector_search(store: HybridStore, question: str, candidate_k: int) -> List[Tuple[Document, float]]:
    try:
        return store.vectorstore.similarity_search_with_score(question, k=candidate_k)
    except Exception as exc:
        raise RuntimeError(
            "Vector search bi loi. Hybrid RAG khong the tiep tuc vi ket qua se sai/lech. "
            f"Loi: {exc}. Hay kiem tra embedding model va rebuild lai index."
        ) from exc


def _merge_ranked_hits(
    candidates: Dict[str, RankedDocument],
    hits: Iterable[Tuple[Document, float]],
    source: str,
    weight: float,
) -> None:
    """Cong diem reciprocal rank cua mot nguon (vector/bm25) vao candidates."""
    for rank, (document, _raw_score) in enumerate(hits, start=1):
        key = document_key(document)
        candidate = candidates.setdefault(key, RankedDocument(document=document, score=0.0))
        candidate.score += weight * (1.0 / (rank + RRF_K))
        setattr(candidate, f"{source}_rank", rank)


def context_text(question: str, store: HybridStore, ranked: RankedDocument) -> str:
    """Ca parent neu du ngan; neu khong thi cac child lien quan nhat, giu thu tu trong Dieu.

    Uu tien: child trung vector, roi child ke ben chung, roi child khop tu khoa nhat.
    """
    parent = ranked.document
    if len(parent.page_content) <= PARENT_CONTEXT_CHARS:
        return parent.page_content
    children = store.children.get(parent.metadata["parent_id"], [])
    if not children:
        return parent.page_content[:PARENT_CONTEXT_CHARS]

    def by_keyword() -> Iterator[int]:
        # Chi tinh khi child trung vector va child ke ben chua lap day context.
        yield from sorted(
            range(len(children)),
            key=lambda index: -_lexical_rerank_score(question, children[index].page_content, {}),
        )

    neighbors = [index + step for index in ranked.matched_children for step in (-1, 1)]
    selected: List[int] = []
    used = 0
    for index in chain(ranked.matched_children, neighbors, by_keyword()):
        if index in selected or not 0 <= index < len(children):
            continue
        size = len(child_body(children[index]))
        if selected and used + size > PARENT_CONTEXT_CHARS:
            if used >= PARENT_CONTEXT_CHARS * 0.9:
                break
            continue
        selected.append(index)
        used += size

    parts: List[str] = []
    previous = None
    for index in sorted(selected):
        if previous is not None and index != previous + 1:
            parts.append("…")
        parts.append(child_body(children[index]))
        previous = index
    return "\n".join(parts)


def _rerank_text(metadata: dict, content: str) -> str:
    prefix = " ".join(str(metadata[key]) for key in RERANK_METADATA_KEYS if metadata.get(key))
    return f"{prefix}\n{content}"


def _lexical_rerank_score(question: str, text: str, metadata: dict) -> float:
    """Diem thay CrossEncoder: do phu token, khop cum 2-4 token va boost metadata."""
    query_tokens = tokenize(question)
    text_tokens = tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0

    text_set = set(text_tokens)
    query_set = set(query_tokens)
    coverage = sum(1 for token in query_tokens if token in text_set) / len(query_tokens)
    unique_coverage = len(query_set & text_set) / len(query_set)

    lowered = text.lower()
    phrase_boost = 0.0
    for size in (4, 3, 2):
        for index in range(0, max(len(query_tokens) - size + 1, 0)):
            if " ".join(query_tokens[index : index + size]) in lowered:
                phrase_boost += 0.08 * size

    metadata_boost = 0.0
    for key in ("article_code", "legal_note", "doc_type", "effective_date"):
        value = str(metadata.get(key, "")).lower()
        if value and any(token in value for token in query_set):
            metadata_boost += 0.08

    return coverage + 0.5 * unique_coverage + phrase_boost + metadata_boost
