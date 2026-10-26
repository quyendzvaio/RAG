import math
import os
import pickle
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document


TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
# page_content cua parent chi la noi dung Dieu, nen moi truong duoi day chi duoc dem mot lan.
SEARCHABLE_METADATA_KEYS = (
    "article_code",
    "article_title",
    "chapter",
    "chapter_title",
    "legal_note",
    "doc_type",
)


def tokenize(text: str) -> List[str]:
    """Tokenizer dung chung cho vector BM25, wiki BM25 va metric cua evaluator."""
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25 Okapi tren danh sach token, khong phu thuoc kieu document.

    Ten thuoc tinh duoc giu nguyen de pickle `bm25_store.pkl` cu van load duoc.
    """

    def __init__(self, token_lists: Sequence[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_freqs: List[Dict[str, int]] = []
        self.doc_lengths: List[int] = []
        doc_freqs: Dict[str, int] = defaultdict(int)
        inverted_index: Dict[str, List[int]] = defaultdict(list)
        for index, tokens in enumerate(token_lists):
            counts = Counter(tokens)
            self.term_freqs.append(dict(counts))
            self.doc_lengths.append(len(tokens))
            for token in counts:
                doc_freqs[token] += 1
                inverted_index[token].append(index)
        self.doc_freqs = dict(doc_freqs)
        self.inverted_index = dict(inverted_index)
        self.total_docs = len(self.term_freqs)
        self.avg_doc_length = sum(self.doc_lengths) / max(self.total_docs, 1)

    def top_k(self, query_terms: List[str], k: int) -> List[Tuple[int, float]]:
        """Tra ve (vi tri document, diem) giam dan; hoa diem thi vi tri nho dung truoc."""
        candidates = set()
        for term in query_terms:
            candidates.update(self.inverted_index.get(term, ()))

        scores = []
        for index in candidates:
            score = self._score(query_terms, index)
            if score > 0:
                scores.append((index, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:k]

    def _score(self, query_terms: List[str], index: int) -> float:
        term_freq = self.term_freqs[index]
        doc_length = self.doc_lengths[index] or 1
        norm = self.k1 * (1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1))
        score = 0.0
        for term in query_terms:
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue
            df = self.doc_freqs.get(term, 0)
            idf = math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))
            score += idf * (tf * (self.k1 + 1)) / (tf + norm)
        return score


class BM25Store(BM25Index):
    """BM25 tren cac parent (Dieu/Phu luc) cua hybrid index.

    Pickle chi luu term stats va `document_keys`; parent duoc gan lai tu `parents.jsonl`
    khi load (`attach_documents`), nen hai file phai cung mot lan build.
    """

    version = 2

    def __init__(self, documents: Sequence[Document]):
        super().__init__([tokenize(_searchable_text(document)) for document in documents])
        self.document_keys = [document_key(document, index) for index, document in enumerate(documents)]
        self._documents: List[Document] = []
        self.attach_documents(documents)

    def attach_documents(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)
        if len(self._documents) != self.total_docs:
            raise ValueError(
                f"BM25/document store mismatch: BM25 co {self.total_docs} docs, "
                f"docstore co {len(self._documents)} docs."
            )

        for index, document in enumerate(self._documents[:20]):
            if document_key(document, index) != self.document_keys[index]:
                raise ValueError(
                    "BM25/document store mismatch: thu tu hoac khoa document khong khop. "
                    "Hay rebuild lai hybrid index."
                )

    def search_with_scores(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        if not self._documents:
            raise RuntimeError("BM25Store chua duoc attach document store.")
        return [(self._documents[index], score) for index, score in self.top_k(tokenize(query), top_k)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_documents"] = []
        return state


def document_key(document: Document, fallback_index: Optional[int] = None) -> str:
    """Khoa dinh danh: chunk_id cua child, parent_id cua parent, neu khong thi source|article."""
    metadata = document.metadata or {}
    if "chunk_id" in metadata:
        return str(metadata["chunk_id"])
    if "parent_id" in metadata:
        return str(metadata["parent_id"])
    key = "|".join(
        str(metadata.get(name, "")) for name in ("source", "article_code", "article_index")
    ).strip("|")
    if key:
        return key
    return str(fallback_index) if fallback_index is not None else document.page_content[:80]


def _searchable_text(document: Document) -> str:
    metadata = document.metadata
    metadata_text = " ".join(str(metadata.get(key, "")) for key in SEARCHABLE_METADATA_KEYS)
    return f"{metadata_text}\n{document.page_content}"


def save_bm25_store(store: BM25Store, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(store, file)


def load_bm25_store(path: str) -> BM25Store:
    with open(path, "rb") as file:
        return pickle.load(file)
