"""Hybrid search cho vector-rag-service: FAISS + BM25 -> reciprocal rank fusion -> rerank."""
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document

from rag.bm25_store import document_key, tokenize
from rag.device import choose_torch_device
from rag.vector_store import HybridStore
from shared.config import RerankerSettings, get_settings


logger = logging.getLogger(__name__)

VECTOR_WEIGHT = 0.55
BM25_WEIGHT = 0.45
RRF_K = 60  # hang so cua reciprocal rank fusion: weight / (rank + RRF_K)
RERANK_WEIGHT = 0.65  # diem cuoi = RERANK_WEIGHT * rerank chuan hoa + phan con lai * diem fusion
RERANK_METADATA_KEYS = (
    "chapter",
    "chapter_title",
    "article_code",
    "article_title",
    "legal_note",
    "doc_type",
    "effective_date",
)


@dataclass
class RankedDocument:
    document: Document
    score: float
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rerank_score: Optional[float] = None


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

    def score(self, query: str, documents: List[Document]) -> Optional[List[float]]:
        if self.model is None or not documents:
            return None
        pairs = [(query, _rerank_text(document)) for document in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        return [float(score) for score in scores]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(get_settings().reranker)


def hybrid_search(question: str, store: HybridStore, top_k: int = 5, candidate_k: int = 40) -> List[RankedDocument]:
    vector_hits = _vector_search(store, question, candidate_k)
    bm25_hits = store.bm25_store.search_with_scores(question, top_k=candidate_k)

    candidates: Dict[str, RankedDocument] = {}
    _merge_ranked_hits(candidates, vector_hits, "vector", VECTOR_WEIGHT)
    _merge_ranked_hits(candidates, bm25_hits, "bm25", BM25_WEIGHT)

    fused = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
    return rerank(question, fused[:candidate_k])[:top_k]


def rerank(question: str, candidates: List[RankedDocument]) -> List[RankedDocument]:
    if not candidates:
        return []

    documents = [candidate.document for candidate in candidates]
    scores = get_reranker().score(question, documents)
    if scores is None:
        scores = [_lexical_rerank_score(question, document) for document in documents]

    min_score = min(scores)
    score_range = max(max(scores) - min_score, 1e-9)
    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = score
        normalized = (score - min_score) / score_range
        candidate.score = RERANK_WEIGHT * normalized + (1 - RERANK_WEIGHT) * candidate.score

    return sorted(candidates, key=lambda item: item.score, reverse=True)


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


def _rerank_text(document: Document) -> str:
    metadata = document.metadata
    prefix = " ".join(str(metadata[key]) for key in RERANK_METADATA_KEYS if metadata.get(key))
    return f"{prefix}\n{document.page_content}"


def _lexical_rerank_score(question: str, document: Document) -> float:
    """Diem thay CrossEncoder: do phu token, khop cum 2-4 token va boost metadata."""
    query_tokens = tokenize(question)
    text = _rerank_text(document)
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
        value = str(document.metadata.get(key, "")).lower()
        if value and any(token in value for token in query_set):
            metadata_boost += 0.08

    return coverage + 0.5 * unique_coverage + phrase_boost + metadata_boost
