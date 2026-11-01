"""Cau hinh tap trung cho moi service va script.

Doc bien moi truong (va .env neu co) mot lan qua `get_settings()`. Default khop
docker-compose.yml. Duong dan tuong doi tinh tu thu muc dang chay: repo root khi chay
tren host, /app trong container.
"""
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict

from dotenv import load_dotenv


EMBEDDING_PROFILES = {
    "vietnamese": "bkai-foundation-models/vietnamese-bi-encoder",
    "bge-m3": "BAAI/bge-m3",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "multilingual-minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
DEEPSEEK_PLACEHOLDER_KEY = "sk-your-deepseek-api-key"


@dataclass(frozen=True)
class EmbeddingSettings:
    backend: str
    profile: str
    model: str
    device: str
    batch_size: int
    hashing_dim: int
    runtime: str  # "onnx" (ONNX Runtime tren CPU) hoac "torch"; khong vao manifest vi vector nhu nhau
    onnx_dir: str


@dataclass(frozen=True)
class RerankerSettings:
    model: str  # rong = dung lexical rerank thay cho CrossEncoder
    device: str
    batch_size: int


@dataclass(frozen=True)
class DeepSeekSettings:
    api_key: str
    base_url: str
    model: str
    # Truong JSON them vao moi request chat (DEEPSEEK_EXTRA_BODY), vd. {"thinking": {"type": "disabled"}}
    # de tat suy luan; moi provider nhan tham so khac nhau.
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_key != DEEPSEEK_PLACEHOLDER_KEY)

    @property
    def chat_completions_url(self) -> str:
        """Chap nhan ca base URL co san /v1 (proxy OpenAI-compatible) lan khong co."""
        return chat_completions_url(self.base_url)


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"


@dataclass(frozen=True)
class ServiceUrls:
    vector: str
    wiki: str
    evaluator: str
    llm: str


@dataclass(frozen=True)
class Paths:
    data_dir: str
    faiss_dir: str
    hybrid_dir: str
    raw_sources_dir: str
    wiki_dir: str
    eval_db: str

    @property
    def bm25_store(self) -> str:
        return os.path.join(self.hybrid_dir, "bm25_store.pkl")

    @property
    def index_manifest(self) -> str:
        return os.path.join(self.hybrid_dir, "index_manifest.json")

    @property
    def parents(self) -> str:
        return os.path.join(self.hybrid_dir, "parents.jsonl")


@dataclass(frozen=True)
class Settings:
    embedding: EmbeddingSettings
    reranker: RerankerSettings
    deepseek: DeepSeekSettings
    services: ServiceUrls
    paths: Paths
    preload_vector_index: bool
    preload_wiki_cache: bool
    force_cuda: bool


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    profile = _env("EMBEDDING_PROFILE", "vietnamese").lower()
    return Settings(
        embedding=EmbeddingSettings(
            backend=_env("EMBEDDING_BACKEND", "sentence-transformers").lower(),
            profile=profile,
            model=_env("EMBEDDING_MODEL", EMBEDDING_PROFILES.get(profile, EMBEDDING_PROFILES["vietnamese"])),
            device=_env("EMBEDDING_DEVICE", "auto").lower(),
            batch_size=int(_env("EMBEDDING_BATCH_SIZE", "8")),
            hashing_dim=int(_env("HASHING_EMBEDDING_DIM", "384")),
            runtime=_env("EMBEDDING_RUNTIME", "onnx").lower(),
            onnx_dir=_env("EMBEDDING_ONNX_DIR", ".cache/onnx"),
        ),
        reranker=RerankerSettings(
            model=_env("RERANKER_MODEL", ""),
            device=_env("RERANKER_DEVICE", "auto").lower(),
            batch_size=int(_env("RERANKER_BATCH_SIZE", "4")),
        ),
        deepseek=DeepSeekSettings(
            api_key=_env("DEEPSEEK_API_KEY", ""),
            base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=_env("DEEPSEEK_MODEL", "deepseek-chat"),
            extra_body=json.loads(_env("DEEPSEEK_EXTRA_BODY", "") or "{}"),
        ),
        services=ServiceUrls(
            vector=_env("VECTOR_SERVICE_URL", "http://vector-rag-service:8001"),
            wiki=_env("WIKI_SERVICE_URL", "http://wiki-rag-service:8002"),
            evaluator=_env("EVALUATOR_SERVICE_URL", "http://evaluator-service:8004"),
            llm=_env("LLM_SERVICE_URL", "http://llm-service:8003"),
        ),
        paths=Paths(
            data_dir="data_raw",
            faiss_dir="faiss_index",
            hybrid_dir="hybrid_index",
            raw_sources_dir=_env("RAW_SOURCES_DIR", "wiki_data/raw_sources"),
            wiki_dir=_env("WIKI_DIR", "wiki_data/wiki"),
            eval_db=_env("EVAL_DB_PATH", "eval_data/eval.db"),
        ),
        preload_vector_index=_env_bool("PRELOAD_VECTOR_INDEX", True),
        preload_wiki_cache=_env_bool("PRELOAD_WIKI_CACHE", True),
        force_cuda=_env_bool("FORCE_CUDA", False),
    )


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
