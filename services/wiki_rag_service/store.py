import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from rag.bm25_store import BM25Index, tokenize
from shared.schemas import ContextItem


logger = logging.getLogger(__name__)

# Toan van Dieu dai hon nhieu so voi tieu de: lap tieu de (BM25F don gian) va chuan hoa do dai manh hon
# de Dieu dai khong thang chi vi chua nhieu tu. Do tren golden set: R@1 0.547 -> 0.640, MRR 0.680 -> 0.750.
HEADING_REPEAT = 3
BM25_B = 0.9
COLUMNS = ("id", "source_markdown", "article_heading", "content_preview", "content", "metadata_json", "searchable_text")

@dataclass
class _SearchCache:
    rows: List[Dict]
    index: BM25Index


class WikiStore:
    def __init__(self, db_path: str, raw_sources_dir: str):
        self.db_path = db_path
        self.raw_sources_dir = Path(raw_sources_dir)
        self._search_cache: Optional[_SearchCache] = None
        self._cache_lock = Lock()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def ingest(self) -> Dict[str, int]:
        articles_path = self.raw_sources_dir / "articles.jsonl"
        if not articles_path.exists():
            raise FileNotFoundError(f"Khong thay {articles_path}. Hay chay prepare_wiki_sources truoc.")

        with self._connect() as conn:
            conn.execute("DELETE FROM articles")
            count = 0
            with articles_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    metadata = record.get("metadata", {})
                    searchable = _searchable_record_text(record)
                    conn.execute(
                        f"INSERT INTO articles ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                        (
                            record.get("id", ""),
                            record.get("source_markdown", ""),
                            record.get("article_heading", ""),
                            record.get("content_preview", ""),
                            record.get("content", ""),
                            json.dumps(metadata, ensure_ascii=False),
                            searchable,
                        ),
                    )
                    count += 1
            conn.commit()
        self._search_cache = None
        return {"article_count": count}

    def status(self) -> Dict:
        with self._connect() as conn:
            article_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        return {
            "db_path": self.db_path,
            "raw_sources_dir": str(self.raw_sources_dir),
            "article_count": article_count,
            "cache_ready": self._search_cache is not None,
        }

    def warm_cache(self) -> Dict[str, int]:
        return {"article_count": len(self._load_search_cache().rows)}

    def search(self, query: str, top_k: int = 5) -> List[ContextItem]:
        cache = self._load_search_cache()
        contexts = []
        for index, score in cache.index.top_k(tokenize(query), top_k):
            row = cache.rows[index]
            metadata = json.loads(row["metadata_json"])
            citation = metadata.get("legal_note") or metadata.get("source_file") or row["source_markdown"]
            contexts.append(
                ContextItem(
                    id=row["id"],
                    source=row["source_markdown"],
                    title=row["article_heading"],
                    content=row["content"] or row["content_preview"],
                    metadata=metadata,
                    score=score,
                    citation=citation,
                )
            )
        return contexts

    def _init_db(self) -> None:
        with self._connect() as conn:
            existing = [row["name"] for row in conn.execute("PRAGMA table_info(articles)")]
            if existing and set(existing) != set(COLUMNS):
                # Schema cu (chua co cot content): du lieu se duoc ingest lai tu articles.jsonl.
                logger.warning("Bang articles co schema cu %s, tao lai bang. Hay chay ingest.", existing)
                conn.execute("DROP TABLE articles")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    source_markdown TEXT NOT NULL,
                    article_heading TEXT,
                    content_preview TEXT,
                    content TEXT,
                    metadata_json TEXT NOT NULL,
                    searchable_text TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_article_rows(self):
        with self._connect() as conn:
            rows = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM articles").fetchall()
        return [dict(row) for row in rows]

    def _load_search_cache(self) -> _SearchCache:
        if self._search_cache is not None:
            return self._search_cache

        with self._cache_lock:
            if self._search_cache is None:
                rows = self._load_article_rows()
                index = BM25Index([tokenize(row["searchable_text"]) for row in rows], b=BM25_B)
                self._search_cache = _SearchCache(rows=rows, index=index)
            return self._search_cache


def _searchable_record_text(record: Dict) -> str:
    metadata = record.get("metadata", {})
    return "\n".join(
        str(value)
        for value in (
            " ".join([record.get("article_heading", "")] * HEADING_REPEAT),
            record.get("content") or record.get("content_preview", ""),
            metadata.get("chapter", ""),
            metadata.get("chapter_title", ""),
            metadata.get("legal_note", ""),
            metadata.get("doc_type", ""),
        )
        if value
    )

