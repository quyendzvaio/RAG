import hashlib
import os
from typing import Dict, List

from langchain_core.embeddings import Embeddings

from rag.device import choose_torch_device


EMBEDDING_PROFILES = {
    "vietnamese": "bkai-foundation-models/vietnamese-bi-encoder",
    "bge-m3": "BAAI/bge-m3",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "multilingual-minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


class HashingEmbeddings(Embeddings):
    """Fallback de index/test khi chua cai embedding model semantic."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


def get_embedding_model():
    backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().lower()
    profile = os.getenv("EMBEDDING_PROFILE", "vietnamese").strip().lower()
    default_model = EMBEDDING_PROFILES.get(profile, EMBEDDING_PROFILES["vietnamese"])
    model_name = os.getenv("EMBEDDING_MODEL", default_model)

    if backend in {"sentence-transformers", "huggingface", "hf"}:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            choice = choose_torch_device("EMBEDDING_DEVICE", default="cpu")
            print(f"Embedding device: {choice.device} ({choice.reason})")
            if choice.gpu_name:
                print(
                    "GPU:",
                    choice.gpu_name,
                    "compute_capability:",
                    choice.compute_capability,
                    "memory_mb:",
                    choice.total_memory_mb,
                )
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": choice.device},
                encode_kwargs={
                    "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
                    "normalize_embeddings": True,
                },
            )
        except Exception as exc:
            print("Khong tai duoc HuggingFaceEmbeddings, dung fallback hashing.")
            print("Loi:", exc)
            print("Luu y: hashing chi phu hop smoke test, khong phu hop chat luong production.")
            return HashingEmbeddings()

    if backend == "hashing":
        return HashingEmbeddings(dimensions=int(os.getenv("HASHING_EMBEDDING_DIM", "384")))

    raise ValueError(f"Embedding backend khong ho tro: {backend}")


def embedding_config(embedding_model) -> Dict[str, str]:
    if isinstance(embedding_model, HashingEmbeddings):
        return {
            "backend": "hashing",
            "profile": "hashing",
            "model": "hashing",
            "dimensions": str(embedding_model.dimensions),
        }

    profile = os.getenv("EMBEDDING_PROFILE", "vietnamese").strip().lower()
    default_model = EMBEDDING_PROFILES.get(profile, EMBEDDING_PROFILES["vietnamese"])
    model_name = os.getenv("EMBEDDING_MODEL", getattr(embedding_model, "model_name", default_model))
    backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().lower()
    return {
        "backend": backend,
        "profile": profile,
        "model": str(model_name),
        "dimensions": "",
    }
