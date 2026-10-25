import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from rag.bm25_store import BM25Index, tokenize
from shared.schemas import ContextItem


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
                        """
                        INSERT INTO articles (
                            id, source_markdown, article_heading, content_preview,
                            metadata_json, searchable_text
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.get("id", ""),
                            record.get("source_markdown", ""),
                            record.get("article_heading", ""),
                            record.get("content_preview", ""),
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
            content = _read_source_section(self.raw_sources_dir / row["source_markdown"], row["article_heading"])
            citation = metadata.get("legal_note") or metadata.get("source_file") or row["source_markdown"]
            contexts.append(
                ContextItem(
                    id=row["id"],
                    source=row["source_markdown"],
                    title=row["article_heading"],
                    content=content or row["content_preview"],
                    metadata=metadata,
                    score=score,
                    citation=citation,
                )
            )
        return contexts

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    source_markdown TEXT NOT NULL,
                    article_heading TEXT,
                    content_preview TEXT,
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
            rows = conn.execute(
                "SELECT id, source_markdown, article_heading, content_preview, metadata_json, searchable_text FROM articles"
            ).fetchall()
        return [dict(row) for row in rows]

    def _load_search_cache(self) -> _SearchCache:
        if self._search_cache is not None:
            return self._search_cache

        with self._cache_lock:
            if self._search_cache is None:
                rows = self._load_article_rows()
                index = BM25Index([tokenize(row["searchable_text"]) for row in rows])
                self._search_cache = _SearchCache(rows=rows, index=index)
            return self._search_cache


def _searchable_record_text(record: Dict) -> str:
    metadata = record.get("metadata", {})
    return "\n".join(
        str(value)
        for value in (
            record.get("article_heading", ""),
            record.get("content_preview", ""),
            metadata.get("chapter", ""),
            metadata.get("chapter_title", ""),
            metadata.get("legal_note", ""),
            metadata.get("doc_type", ""),
            metadata.get("effective_date", ""),
            metadata.get("related_text", ""),
        )
        if value
    )


def _read_source_section(path: Path, heading: str) -> str:
    if not path.exists() or not heading:
        return ""
    text = path.read_text(encoding="utf-8")
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return ""
    next_start = text.find("\n## ", start + len(marker))
    section = text[start : next_start if next_start >= 0 else len(text)]
    return section[:3000]
