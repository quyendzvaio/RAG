import math
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document


TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class BM25Store:
    term_freqs: List[Dict[str, int]]
    doc_freqs: Dict[str, int]
    doc_lengths: List[int]
    avg_doc_length: float
    document_keys: List[str]
    inverted_index: Dict[str, List[int]]
    total_docs: int
    k1: float = 1.5
    b: float = 0.75
    version: int = 2
    _documents: List[Document] = field(default_factory=list, repr=False, compare=False)

    @classmethod
    def from_documents(cls, documents: Sequence[Document]) -> "BM25Store":
        term_freqs: List[Dict[str, int]] = []
        doc_freqs: Dict[str, int] = defaultdict(int)
        doc_lengths: List[int] = []
        inverted_index: Dict[str, List[int]] = defaultdict(list)
        document_keys: List[str] = []

        for index, document in enumerate(documents):
            document_keys.append(document_key(document, index))
            tokens = tokenize(_searchable_text(document))
            counts = Counter(tokens)
            term_freqs.append(dict(counts))
            doc_lengths.append(len(tokens))
            for token in counts:
                doc_freqs[token] += 1
                inverted_index[token].append(index)

        avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
        store = cls(
            term_freqs=term_freqs,
            doc_freqs=dict(doc_freqs),
            doc_lengths=doc_lengths,
            avg_doc_length=avg_doc_length,
            document_keys=document_keys,
            inverted_index={term: indices for term, indices in inverted_index.items()},
            total_docs=len(documents),
        )
        store.attach_documents(list(documents))
        return store

    def attach_documents(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)
        if len(self._documents) != self.total_docs:
            raise ValueError(
                f"BM25/document store mismatch: BM25 co {self.total_docs} docs, "
                f"docstore co {len(self._documents)} docs."
            )

        mismatches = []
        for index, document in enumerate(self._documents[: min(len(self._documents), 20)]):
            if document_key(document, index) != self.document_keys[index]:
                mismatches.append(index)
        if mismatches:
            raise ValueError(
                "BM25/document store mismatch: thu tu hoac chunk_id khong khop. "
                "Hay rebuild lai hybrid index."
            )

    def search_with_scores(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        if not self._documents:
            raise RuntimeError("BM25Store chua duoc attach document store.")

        query_terms = tokenize(query)
        if not query_terms:
            return []

        candidate_indices = set()
        for term in query_terms:
            candidate_indices.update(self.inverted_index.get(term, []))

        scores: List[Tuple[int, float]] = []
        for index in candidate_indices:
            score = self._score_document(query_terms, index, self.term_freqs[index])
            if score > 0:
                scores.append((index, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [(self._documents[index], score) for index, score in scores[:top_k]]

    def _score_document(self, query_terms: Iterable[str], index: int, term_freq: Dict[str, int]) -> float:
        score = 0.0
        doc_length = self.doc_lengths[index] or 1
        for term in query_terms:
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue
            df = self.doc_freqs.get(term, 0)
            idf = math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))
            norm = self.k1 * (1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1))
            score += idf * (tf * (self.k1 + 1)) / (tf + norm)
        return score

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_documents"] = []
        return state

    def __setstate__(self, state):
        legacy_documents = state.pop("documents", None)
        self.__dict__.update(state)
        self._documents = []
        if not hasattr(self, "version"):
            self.version = 1
        if not hasattr(self, "total_docs"):
            self.total_docs = len(getattr(self, "term_freqs", []))
        if not hasattr(self, "document_keys"):
            self.document_keys = [
                document_key(document, index)
                for index, document in enumerate(legacy_documents or [])
            ]
        if not hasattr(self, "inverted_index"):
            inverted_index: Dict[str, List[int]] = defaultdict(list)
            for index, term_freq in enumerate(self.term_freqs):
                for token in term_freq:
                    inverted_index[token].append(index)
            self.inverted_index = dict(inverted_index)


def document_key(document: Document, fallback_index: Optional[int] = None) -> str:
    metadata = document.metadata or {}
    if "chunk_id" in metadata:
        return str(metadata["chunk_id"])
    parts = [
        str(metadata.get("source", "")),
        str(metadata.get("article_code", "")),
        str(metadata.get("article_index", "")),
    ]
    key = "|".join(parts).strip("|")
    if key:
        return key
    return str(fallback_index) if fallback_index is not None else document.page_content[:80]


def _searchable_text(document: Document) -> str:
    metadata = document.metadata
    metadata_text = " ".join(
        str(metadata.get(key, ""))
        for key in (
            "chapter",
            "chapter_title",
            "article_code",
            "article_title",
            "legal_note",
            "doc_type",
            "effective_date",
            "source",
            "related_text",
        )
    )
    return f"{metadata_text}\n{document.page_content}"


def save_bm25_store(store: BM25Store, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(store, file)


def load_bm25_store(path: str) -> BM25Store:
    with open(path, "rb") as file:
        return pickle.load(file)
