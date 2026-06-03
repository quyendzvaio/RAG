import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List

from rag.bm25_store import tokenize
from shared.schemas import EvaluationRecord, MetricScore, ServiceAnswer


class EvalStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def save(self, record: EvaluationRecord) -> int:
        payload = record.model_dump()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO eval_runs (created_at, question, payload_json) VALUES (?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), record.question, json.dumps(payload, ensure_ascii=False)),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def recent(self, limit: int = 20) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, question, payload_json FROM eval_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "question": row["question"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    question TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def compute_metrics(question: str, vector: ServiceAnswer, wiki: ServiceAnswer) -> List[MetricScore]:
    metrics = []
    metrics.extend(_answer_metrics("vector", question, vector))
    metrics.extend(_answer_metrics("wiki", question, wiki))
    metrics.append(
        MetricScore(
            name="latency_delta_ms_vector_minus_wiki",
            value=round(vector.latency_ms - wiki.latency_ms, 3),
            explanation="Duong neu vector cham hon wiki; am neu vector nhanh hon wiki.",
        )
    )
    return metrics


def _answer_metrics(prefix: str, question: str, answer: ServiceAnswer) -> List[MetricScore]:
    question_terms = set(tokenize(question))
    answer_terms = set(tokenize(answer.answer))
    context_text = " ".join(context.content for context in answer.contexts)
    context_terms = set(tokenize(context_text))
    overlap = len(question_terms & context_terms) / max(len(question_terms), 1)
    answer_overlap = len(question_terms & answer_terms) / max(len(question_terms), 1)
    return [
        MetricScore(
            name=f"{prefix}_latency_ms",
            value=round(answer.latency_ms, 3),
            explanation="Request latency reported by service.",
        ),
        MetricScore(
            name=f"{prefix}_context_count",
            value=float(len(answer.contexts)),
            explanation="Number of retrieved contexts.",
        ),
        MetricScore(
            name=f"{prefix}_citation_count",
            value=float(len(answer.citations)),
            explanation="Number of citations surfaced by service.",
        ),
        MetricScore(
            name=f"{prefix}_question_context_overlap",
            value=round(overlap, 4),
            explanation="Lexical proxy for retrieval relevance.",
        ),
        MetricScore(
            name=f"{prefix}_question_answer_overlap",
            value=round(answer_overlap, 4),
            explanation="Lexical proxy for answer relevance.",
        ),
        MetricScore(
            name=f"{prefix}_error",
            value=1.0 if answer.error else 0.0,
            explanation=answer.error or "No service error.",
        ),
    ]
