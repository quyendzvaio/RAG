import hashlib
import logging
from typing import Dict, List, Optional

from langchain_core.embeddings import Embeddings

from rag.device import choose_torch_device
from shared.config import EmbeddingSettings, get_settings


logger = logging.getLogger(__name__)


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


def get_embedding_model(config: Optional[EmbeddingSettings] = None):
    config = config or get_settings().embedding

    if config.backend in {"sentence-transformers", "huggingface", "hf"}:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            choice = choose_torch_device(config.device, "EMBEDDING_DEVICE", get_settings().force_cuda)
            logger.info("Embedding device: %s (%s)", choice.device, choice.reason)
            if choice.gpu_name:
                logger.info(
                    "GPU: %s compute_capability: %s memory_mb: %s",
                    choice.gpu_name,
                    choice.compute_capability,
                    choice.total_memory_mb,
                )
            return HuggingFaceEmbeddings(
                model_name=config.model,
                model_kwargs={"device": choice.device},
                encode_kwargs={"batch_size": config.batch_size, "normalize_embeddings": True},
            )
        except Exception as exc:
            logger.warning(
                "Khong tai duoc HuggingFaceEmbeddings, dung fallback hashing "
                "(chi phu hop smoke test, khong phu hop production). Loi: %s",
                exc,
            )
            return HashingEmbeddings()

    if config.backend == "hashing":
        return HashingEmbeddings(dimensions=config.hashing_dim)

    raise ValueError(f"Embedding backend khong ho tro: {config.backend}")


def embedding_config(embedding_model) -> Dict[str, str]:
    """Cau hinh embedding ghi vao index manifest va dung de kiem tra khi load."""
    if isinstance(embedding_model, HashingEmbeddings):
        return {
            "backend": "hashing",
            "profile": "hashing",
            "model": "hashing",
            "dimensions": str(embedding_model.dimensions),
        }

    config = get_settings().embedding
    return {
        "backend": config.backend,
        "profile": config.profile,
        "model": config.model,
        "dimensions": "",
    }
