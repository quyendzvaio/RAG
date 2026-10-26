"""Build va load hybrid index parent-child.

- FAISS chua cac child (cum khoan, co tieu de Dieu) de tim kiem ngu nghia.
- `parents.jsonl` chua parent (Dieu/Phu luc day du + metadata), la thu tra ve cho LLM.
- BM25 cham diem truc tiep tren parent.
"""
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.bm25_store import BM25Store, load_bm25_store, save_bm25_store
from rag.chunking import CHILD_TARGET_TOKENS, split_parent_child
from rag.data_loader import load_html_documents
from rag.embeddings import embedding_config, token_counter
from shared.config import get_settings


logger = logging.getLogger(__name__)

INDEX_VERSION = 2  # 1 = chunk 1200 ky tu khong co parent; 2 = parent-child
EMBED_LOG_BATCH = 2048  # so child moi lan embed + ghi log tien do


@dataclass
class HybridStore:
    vectorstore: FAISS
    bm25_store: BM25Store
    parents: Dict[str, Document]
    children: Dict[str, List[Document]]  # parent_id -> child theo thu tu child_index
    manifest: dict


def create_hybrid_vectorstore(embedding_model) -> HybridStore:
    paths = get_settings().paths
    logger.info("Dang doc cac file HTML tu thu muc: %s", paths.data_dir)
    parents = load_html_documents(paths.data_dir)
    if not parents:
        raise RuntimeError(f"Khong tim thay file HTML hoac tat ca file rong trong {paths.data_dir}.")

    count_tokens, max_tokens = token_counter(embedding_model)
    logger.info("Da parse %d parent (Dieu/Phu luc). Dang tach child <= %d token...", len(parents), max_tokens)
    children = split_parent_child(parents, count_tokens, max_tokens)
    logger.info("Da tao %d child de embed.", len(children))

    vectorstore = _embed_children(children, embedding_model)
    os.makedirs(paths.faiss_dir, exist_ok=True)
    vectorstore.save_local(paths.faiss_dir)

    logger.info("Dang tao BM25 lexical index tren parent...")
    bm25_store = BM25Store(parents)
    save_bm25_store(bm25_store, paths.bm25_store)
    _save_parents(parents)

    manifest = _build_manifest(embedding_model, vectorstore, parents, children, max_tokens, paths.data_dir)
    _save_manifest(manifest)

    logger.info("Da luu FAISS tai %s, BM25 + parents + manifest tai %s", paths.faiss_dir, paths.hybrid_dir)
    return HybridStore(
        vectorstore=vectorstore,
        bm25_store=bm25_store,
        parents={parent.metadata["parent_id"]: parent for parent in parents},
        children=_group_children(children),
        manifest=manifest,
    )


def index_readiness() -> Tuple[bool, str]:
    """Kiem tra du file index truoc khi load (khong load model)."""
    paths = get_settings().paths
    checks = [
        (paths.faiss_dir, "thieu thu muc faiss_index"),
        (os.path.join(paths.faiss_dir, "index.faiss"), "thieu faiss_index/index.faiss"),
        (os.path.join(paths.faiss_dir, "index.pkl"), "thieu faiss_index/index.pkl"),
        (paths.bm25_store, "thieu hybrid_index/bm25_store.pkl"),
        (paths.parents, "thieu hybrid_index/parents.jsonl"),
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
    logger.info("Dang tai parents va BM25 lexical index...")
    parents = _load_parents()
    bm25_store = load_bm25_store(paths.bm25_store)
    bm25_store.attach_documents(parents)
    return HybridStore(
        vectorstore=vectorstore,
        bm25_store=bm25_store,
        parents={parent.metadata["parent_id"]: parent for parent in parents},
        children=_group_children(_documents_from_vectorstore(vectorstore)),
        manifest=manifest,
    )


def _embed_children(children: List[Document], embedding_model) -> FAISS:
    texts = [child.page_content for child in children]
    vectors: List[List[float]] = []
    started = time.perf_counter()
    for start in range(0, len(texts), EMBED_LOG_BATCH):
        vectors.extend(embedding_model.embed_documents(texts[start : start + EMBED_LOG_BATCH]))
        elapsed = time.perf_counter() - started
        logger.info("Da embed %d/%d child (%.0f giay)", len(vectors), len(texts), elapsed)
    return FAISS.from_embeddings(
        list(zip(texts, vectors)), embedding_model, metadatas=[child.metadata for child in children]
    )


def _group_children(children: List[Document]) -> Dict[str, List[Document]]:
    grouped: Dict[str, List[Document]] = defaultdict(list)
    for child in children:
        grouped[child.metadata["parent_id"]].append(child)
    for siblings in grouped.values():
        siblings.sort(key=lambda child: child.metadata["child_index"])
    return dict(grouped)


def _save_parents(parents: List[Document]) -> None:
    paths = get_settings().paths
    os.makedirs(paths.hybrid_dir, exist_ok=True)
    with open(paths.parents, "w", encoding="utf-8") as file:
        for parent in parents:
            record = {"page_content": parent.page_content, "metadata": parent.metadata}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_parents() -> List[Document]:
    with open(get_settings().paths.parents, "r", encoding="utf-8") as file:
        return [Document(**json.loads(line)) for line in file if line.strip()]


def _build_manifest(
    embedding_model, vectorstore: FAISS, parents: List[Document], children: List[Document], max_tokens: int, data_dir: str
) -> dict:
    config = embedding_config(embedding_model)
    config["dimensions"] = str(vectorstore.index.d)
    return {
        "version": INDEX_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": data_dir,
        "parent_count": len(parents),
        "child_count": len(children),
        "faiss_ntotal": vectorstore.index.ntotal,
        "chunking": {
            "strategy": "parent-child",
            "child_max_tokens": max_tokens,
            "child_target_tokens": CHILD_TARGET_TOKENS,
        },
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
    if manifest.get("version") != INDEX_VERSION:
        raise ValueError(
            f"Index version {manifest.get('version')} khong con duoc ho tro (can {INDEX_VERSION}, parent-child). "
            "Hay chay lai: make docker-build-vector-index"
        )
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
    """Child theo dung thu tu hang FAISS."""
    documents: List[Document] = []
    for index in range(vectorstore.index.ntotal):
        document = vectorstore.docstore.search(vectorstore.index_to_docstore_id[index])
        if not isinstance(document, Document):
            raise ValueError(f"Khong tim thay document cho FAISS row {index}. Hay rebuild lai index.")
        documents.append(document)
    return documents
