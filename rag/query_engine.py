import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document

from rag.bm25_store import tokenize
from rag.device import choose_torch_device
from rag.vector_store import HybridStore


@dataclass
class RankedDocument:
    document: Document
    score: float
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class CrossEncoderReranker:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "")
        self.model = None
        if not self.model_name:
            return
        try:
            from sentence_transformers import CrossEncoder

            choice = choose_torch_device("RERANKER_DEVICE", default="cpu")
            print(f"Reranker device: {choice.device} ({choice.reason})")
            self.model = CrossEncoder(self.model_name, device=choice.device)
            print(f"Da tai neural reranker: {self.model_name}")
        except Exception as exc:
            print(f"Khong tai duoc neural reranker '{self.model_name}', dung lexical rerank. Loi: {exc}")
            self.model = None

    def score(self, query: str, documents: List[Document]) -> Optional[List[float]]:
        if self.model is None or not documents:
            return None
        pairs = [(query, _rerank_text(document)) for document in documents]
        scores = self.model.predict(pairs, batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "8")))
        return [float(score) for score in scores]


_RERANKER: Optional[CrossEncoderReranker] = None


def search_query(
    question: str,
    store,
    top_k: int = 5,
    candidate_k: int = 40,
    vector_weight: float = 0.55,
    bm25_weight: float = 0.45,
) -> List[Document]:
    ranked = hybrid_search(
        question=question,
        store=store,
        top_k=top_k,
        candidate_k=candidate_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )
    return [item.document for item in ranked]


def hybrid_search(
    question: str,
    store,
    top_k: int = 5,
    candidate_k: int = 40,
    vector_weight: float = 0.55,
    bm25_weight: float = 0.45,
) -> List[RankedDocument]:
    print(f"Cau hoi: {question}")
    if isinstance(store, HybridStore):
        hybrid_store = store
    else:
        return _vector_only_search(question, store, top_k)

    vector_hits = _vector_search(hybrid_store, question, candidate_k)
    bm25_hits = hybrid_store.bm25_store.search_with_scores(question, top_k=candidate_k)

    candidates: Dict[str, RankedDocument] = {}
    _merge_ranked_hits(candidates, vector_hits, "vector", vector_weight)
    _merge_ranked_hits(candidates, bm25_hits, "bm25", bm25_weight)

    ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
    reranked = rerank(question, ranked[:candidate_k])
    return reranked[:top_k]


def rerank(question: str, candidates: List[RankedDocument]) -> List[RankedDocument]:
    global _RERANKER
    if not candidates:
        return []
    if _RERANKER is None:
        _RERANKER = CrossEncoderReranker()

    documents = [candidate.document for candidate in candidates]
    neural_scores = _RERANKER.score(question, documents)
    if neural_scores is None:
        neural_scores = [_lexical_rerank_score(question, document) for document in documents]

    min_score = min(neural_scores)
    max_score = max(neural_scores)
    score_range = max(max_score - min_score, 1e-9)
    for candidate, score in zip(candidates, neural_scores):
        candidate.rerank_score = score
        normalized = (score - min_score) / score_range
        candidate.score = 0.65 * normalized + 0.35 * candidate.score

    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _vector_search(hybrid_store: HybridStore, question: str, candidate_k: int) -> List[Tuple[Document, float]]:
    try:
        return hybrid_store.vectorstore.similarity_search_with_score(question, k=candidate_k)
    except Exception as exc:
        raise RuntimeError(
            "Vector search bi loi. Hybrid RAG khong the tiep tuc vi ket qua se sai/lech. "
            f"Loi: {exc}. Hay kiem tra embedding model va rebuild lai index."
        ) from exc


def _vector_only_search(question: str, vectorstore, top_k: int) -> List[RankedDocument]:
    try:
        docs = vectorstore.similarity_search(question, k=top_k)
    except Exception as exc:
        print("Loi khi tim kiem tuong dong:", str(exc))
        return []
    return [RankedDocument(document=doc, score=1.0 / (index + 1), vector_rank=index + 1) for index, doc in enumerate(docs)]


def _merge_ranked_hits(
    candidates: Dict[str, RankedDocument],
    hits: Iterable[Tuple[Document, float]],
    source: str,
    weight: float,
) -> None:
    for rank, (document, _raw_score) in enumerate(hits, start=1):
        key = _document_key(document)
        contribution = weight * (1.0 / (rank + 60))
        if key not in candidates:
            candidates[key] = RankedDocument(document=document, score=0.0)
        candidates[key].score += contribution
        if source == "vector":
            candidates[key].vector_rank = rank
        elif source == "bm25":
            candidates[key].bm25_rank = rank


def _document_key(document: Document) -> str:
    metadata = document.metadata
    if "chunk_id" in metadata:
        return str(metadata["chunk_id"])
    return "|".join(
        str(metadata.get(key, ""))
        for key in ("source", "article_code", "article_index")
    ) or document.page_content[:80]


def _rerank_text(document: Document) -> str:
    metadata = document.metadata
    prefix = " ".join(
        str(metadata.get(key, ""))
        for key in (
            "chapter",
            "chapter_title",
            "article_code",
            "article_title",
            "legal_note",
            "doc_type",
            "effective_date",
        )
        if metadata.get(key)
    )
    return f"{prefix}\n{document.page_content}"


def _lexical_rerank_score(question: str, document: Document) -> float:
    query_tokens = tokenize(question)
    text_tokens = tokenize(_rerank_text(document))
    if not query_tokens or not text_tokens:
        return 0.0

    text_set = set(text_tokens)
    query_set = set(query_tokens)
    overlap = sum(1 for token in query_tokens if token in text_set)
    unique_overlap = len(query_set & text_set)
    coverage = overlap / len(query_tokens)
    unique_coverage = unique_overlap / len(query_set)

    text = _rerank_text(document).lower()
    phrase_boost = 0.0
    for size in (4, 3, 2):
        for index in range(0, max(len(query_tokens) - size + 1, 0)):
            phrase = " ".join(query_tokens[index : index + size])
            if phrase in text:
                phrase_boost += 0.08 * size

    metadata = document.metadata
    metadata_boost = 0.0
    for key in ("article_code", "legal_note", "doc_type", "effective_date"):
        value = str(metadata.get(key, "")).lower()
        if value and any(token in value for token in query_set):
            metadata_boost += 0.08

    return coverage + 0.5 * unique_coverage + phrase_boost + metadata_boost
