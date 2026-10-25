import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.bm25_store import BM25Store, load_bm25_store, save_bm25_store
from rag.data_loader import load_html_documents, split_documents
from rag.embeddings import embedding_config
from shared.config import get_settings


logger = logging.getLogger(__name__)


@dataclass
class HybridStore:
    vectorstore: FAISS
    bm25_store: BM25Store
    manifest: dict


def create_hybrid_vectorstore(embedding_model) -> HybridStore:
    paths = get_settings().paths
    logger.info("Dang doc cac file HTML tu thu muc: %s", paths.data_dir)
    documents = load_html_documents(paths.data_dir)
    if not documents:
        raise RuntimeError(f"Khong tim thay file HTML hoac tat ca file rong trong {paths.data_dir}.")

    logger.info("Da parse %d don vi phap ly. Dang chia chunk co cau truc...", len(documents))
    chunks = split_documents(documents)
    logger.info("Da tao %d chunk de index.", len(chunks))

    logger.info("Dang tao FAISS vector index...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(paths.faiss_dir, exist_ok=True)
    vectorstore.save_local(paths.faiss_dir)

    logger.info("Dang tao BM25 lexical index...")
    bm25_store = BM25Store(chunks)
    save_bm25_store(bm25_store, paths.bm25_store)

    manifest = _build_manifest(embedding_model, vectorstore, chunks, paths.data_dir)
    _save_manifest(manifest)

    logger.info("Da luu FAISS tai %s, BM25 tai %s, manifest tai %s", paths.faiss_dir, paths.bm25_store, paths.index_manifest)
    return HybridStore(vectorstore=vectorstore, bm25_store=bm25_store, manifest=manifest)


def index_readiness() -> Tuple[bool, str]:
    """Kiem tra du file index truoc khi load (khong load model)."""
    paths = get_settings().paths
    checks = [
        (paths.faiss_dir, "thieu thu muc faiss_index"),
        (os.path.join(paths.faiss_dir, "index.faiss"), "thieu faiss_index/index.faiss"),
        (os.path.join(paths.faiss_dir, "index.pkl"), "thieu faiss_index/index.pkl"),
        (paths.bm25_store, "thieu hybrid_index/bm25_store.pkl"),
        (paths.index_manifest, "thieu hybrid_index/index_manifest.json"),
    ]
    for path, reason in checks:
        if not os.path.exists(path):
            return False, reason
    return True, "ready"


def load_hybrid_vectorstore(embedding_model) -> HybridStore:
    ready, reason = index_readiness()
    if not ready:
        raise RuntimeError(f"Hybrid index chua san sang: {reason}. Hay chay: make docker-build-vector-index")

    paths = get_settings().paths
    manifest = _load_manifest()
    _validate_manifest(manifest, embedding_model)

    # Ca hai buoc load deu chay pickle: chi load index do chinh ung dung nay tao ra.
    logger.info("Dang tai FAISS vector index...")
    vectorstore = FAISS.load_local(paths.faiss_dir, embedding_model, allow_dangerous_deserialization=True)
    logger.info("Dang tai BM25 lexical index...")
    bm25_store = load_bm25_store(paths.bm25_store)
    bm25_store.attach_documents(_documents_from_vectorstore(vectorstore))
    return HybridStore(vectorstore=vectorstore, bm25_store=bm25_store, manifest=manifest)


def _build_manifest(embedding_model, vectorstore: FAISS, chunks: List[Document], data_dir: str) -> dict:
    config = embedding_config(embedding_model)
    config["dimensions"] = str(vectorstore.index.d)
    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": data_dir,
        "chunk_count": len(chunks),
        "faiss_ntotal": vectorstore.index.ntotal,
        "embedding": config,
        "warning": "Only load this index if it was generated locally from trusted data.",
    }


def _save_manifest(manifest: dict) -> None:
    paths = get_settings().paths
    os.makedirs(paths.hybrid_dir, exist_ok=True)
    with open(paths.index_manifest, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def _load_manifest() -> dict:
    with open(get_settings().paths.index_manifest, "r", encoding="utf-8") as file:
        return json.load(file)


def _validate_manifest(manifest: dict, embedding_model) -> None:
    expected = embedding_config(embedding_model)
    actual = manifest.get("embedding", {})
    mismatches = [
        (key, actual.get(key, ""), expected.get(key, ""))
        for key in ("backend", "profile", "model")
        if str(actual.get(key, "")) != str(expected.get(key, ""))
    ]
    if mismatches:
        details = "; ".join(f"{key}: index={old}, current={new}" for key, old, new in mismatches)
        raise ValueError(
            "Embedding config khong khop voi index hien tai. "
            f"{details}. Hay chay lai: make docker-build-vector-index"
        )


def _documents_from_vectorstore(vectorstore: FAISS) -> List[Document]:
    """Document theo dung thu tu hang FAISS, de gan lai cho BM25."""
    documents: List[Document] = []
    for index in range(vectorstore.index.ntotal):
        document = vectorstore.docstore.search(vectorstore.index_to_docstore_id[index])
        if not isinstance(document, Document):
            raise ValueError(f"Khong tim thay document cho FAISS row {index}. Hay rebuild lai index.")
        documents.append(document)
    return documents
