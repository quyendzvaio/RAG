from functools import lru_cache

from fastapi import FastAPI

from services.evaluator_service.store import EvalStore, compute_metrics
from shared.config import get_settings
from shared.log import setup_logging
from shared.schemas import EvaluationRecord, HealthResponse


setup_logging()
app = FastAPI(title="evaluator-service", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(service="evaluator-service")


@app.post("/eval/run")
def eval_run(record: EvaluationRecord):
    metrics = compute_metrics(record.question, record.vector_answer, record.wiki_answer)
    record.metrics = metrics
    run_id = _store().save(record)
    return {"run_id": run_id, "metrics": [metric.model_dump() for metric in metrics]}


@app.get("/eval/runs")
def eval_runs(limit: int = 20):
    return _store().recent(limit=limit)


@lru_cache(maxsize=1)
def _store() -> EvalStore:
    return EvalStore(get_settings().paths.eval_db)
