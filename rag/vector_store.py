import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.bm25_store import BM25Store, load_bm25_store, save_bm25_store
from rag.data_loader import load_html_documents, split_documents
from rag.embeddings import embedding_config


FAISS_DIR = "faiss_index"
HYBRID_DIR = "hybrid_index"
BM25_PATH = os.path.join(HYBRID_DIR, "bm25_store.pkl")
MANIFEST_PATH = os.path.join(HYBRID_DIR, "index_manifest.json")
DATA_DIR = "data_raw"


@dataclass
class HybridStore:
    vectorstore: FAISS
    bm25_store: BM25Store
    manifest: dict


def create_hybrid_vectorstore(embedding_model, data_dir: str = DATA_DIR) -> Optional[HybridStore]:
    print("Dang doc cac file HTML tu thu muc:", data_dir)
    documents = load_html_documents(data_dir)

    if not documents:
        print("Khong tim thay file HTML hoac tat ca file rong.")
        return None

    print(f"Da parse {len(documents)} don vi phap ly. Dang chia chunk co cau truc...")
    chunks = split_documents(documents)
    print(f"Da tao {len(chunks)} chunk de index.")

    print("Dang tao FAISS vector index...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)

    print("Dang tao BM25 lexical index...")
    bm25_store = BM25Store.from_documents(chunks)
    save_bm25_store(bm25_store, BM25_PATH)

    manifest = _build_manifest(embedding_model, vectorstore, chunks, data_dir)
    _save_manifest(manifest)

    print(f"Da luu FAISS tai: {FAISS_DIR}")
    print(f"Da luu BM25 tai: {BM25_PATH}")
    print(f"Da luu manifest tai: {MANIFEST_PATH}")
    return HybridStore(vectorstore=vectorstore, bm25_store=bm25_store, manifest=manifest)


def create_faiss_vectorstore(embedding_model):
    return create_hybrid_vectorstore(embedding_model)


def load_hybrid_vectorstore(embedding_model) -> Optional[HybridStore]:
    if not os.path.exists(FAISS_DIR):
        print("Vectorstore chua ton tai. Hay chay tao index truoc.")
        return None
    if not os.path.exists(BM25_PATH):
        print("BM25 index chua ton tai. Hay chay tao index lai.")
        return None
    if not os.path.exists(MANIFEST_PATH):
        print("Index manifest chua ton tai. Hay rebuild lai hybrid index de tranh load sai embedding model.")
        return None

    manifest = _load_manifest()
    _validate_manifest(manifest, embedding_model)

    print("Dang tai FAISS vector index...")
    print("Canh bao: chi load FAISS index do chinh ung dung nay tao ra.")
    vectorstore = FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)

    documents = _documents_from_vectorstore(vectorstore)
    print("Dang tai BM25 lexical index...")
    print("Canh bao: chi load BM25 pickle do chinh ung dung nay tao ra.")
    bm25_store = load_bm25_store(BM25_PATH)
    bm25_store.attach_documents(documents)
    return HybridStore(vectorstore=vectorstore, bm25_store=bm25_store, manifest=manifest)


def load_faiss_vectorstore(embedding_model):
    hybrid_store = load_hybrid_vectorstore(embedding_model)
    return hybrid_store.vectorstore if hybrid_store else None


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
    os.makedirs(HYBRID_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def _load_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _validate_manifest(manifest: dict, embedding_model) -> None:
    expected = embedding_config(embedding_model)
    actual = manifest.get("embedding", {})
    mismatches = []
    for key in ("backend", "profile", "model"):
        if str(actual.get(key, "")) != str(expected.get(key, "")):
            mismatches.append((key, actual.get(key, ""), expected.get(key, "")))

    if mismatches:
        details = "; ".join(f"{key}: index={old}, current={new}" for key, old, new in mismatches)
        raise ValueError(
            "Embedding config khong khop voi index hien tai. "
            f"{details}. Hay chay lai: python3 create_faiss.py"
        )


def _documents_from_vectorstore(vectorstore: FAISS) -> List[Document]:
    documents: List[Document] = []
    for index in range(vectorstore.index.ntotal):
        docstore_id = vectorstore.index_to_docstore_id[index]
        document = vectorstore.docstore.search(docstore_id)
        if not isinstance(document, Document):
            raise ValueError(f"Khong tim thay document cho FAISS row {index}. Hay rebuild lai index.")
        documents.append(document)
    return documents
