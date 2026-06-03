import json
import math
import os
import sqlite3
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from rag.bm25_store import tokenize
from shared.schemas import ContextItem


class WikiStore:
    def __init__(self, db_path: str, raw_sources_dir: str):
        self.db_path = db_path
        self.raw_sources_dir = Path(raw_sources_dir)
        self._search_cache: Optional[Dict] = None
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
        cache = self._load_search_cache()
        return {"article_count": len(cache["rows"])}

    def search(self, query: str, top_k: int = 5) -> List[ContextItem]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        cache = self._load_search_cache()
        rows = cache["rows"]
        total_docs = cache["total_docs"]
        df = cache["df"]
        tokenized_rows = cache["tokenized_rows"]
        avg_len = cache["avg_len"]
        scored: List[Tuple[int, float]] = []
        for index, counts in enumerate(tokenized_rows):
            score = _bm25_score(query_terms, counts, df, total_docs, avg_len)
            if score > 0:
                scored.append((index, score))
        scored.sort(key=lambda item: item[1], reverse=True)

        contexts = []
        for index, score in scored[:top_k]:
            row = rows[index]
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

    def _load_search_cache(self) -> Dict:
        if self._search_cache is not None:
            return self._search_cache

        with self._cache_lock:
            if self._search_cache is not None:
                return self._search_cache

            rows = self._load_article_rows()
            total_docs = max(len(rows), 1)
            df = Counter()
            tokenized_rows = []
            for row in rows:
                counts = Counter(tokenize(row["searchable_text"]))
                tokenized_rows.append(counts)
                for token in counts:
                    df[token] += 1

            avg_len = sum(sum(counts.values()) for counts in tokenized_rows) / total_docs
            self._search_cache = {
                "rows": rows,
                "total_docs": total_docs,
                "df": df,
                "tokenized_rows": tokenized_rows,
                "avg_len": avg_len,
            }
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


def _bm25_score(query_terms, counts, df, total_docs, avg_len, k1=1.5, b=0.75) -> float:
    score = 0.0
    doc_len = sum(counts.values()) or 1
    for term in query_terms:
        tf = counts.get(term, 0)
        if tf == 0:
            continue
        idf = math.log(1 + (total_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
        norm = k1 * (1 - b + b * doc_len / max(avg_len, 1))
        score += idf * (tf * (k1 + 1)) / (tf + norm)
    return score


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
