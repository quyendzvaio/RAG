import logging

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from gradio.routes import mount_gradio_app

from shared.config import get_settings
from shared.http import ANSWER_TIMEOUT, service_client
from shared.log import setup_logging
from shared.schemas import AskRequest, CompareResponse, EvaluationRecord, HealthResponse, ServiceAnswer
from shared.tracing import ContextThreadPoolExecutor, traceable, uuid7


setup_logging()
logger = logging.getLogger(__name__)
SERVICES = get_settings().services

app = FastAPI(title="api-gateway", version="0.1.0")


@app.get("/")
def root():
    return RedirectResponse(url="/ui")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(service="api-gateway")


@app.post("/ask/compare", response_model=CompareResponse)
def ask_compare(request: AskRequest) -> CompareResponse:
    # request_id cung la id trace goc tren LangSmith, va duoc luu vao eval_runs: tu mot dong SQLite tim duoc trace.
    request_id = str(uuid7())
    settings = get_settings()
    metadata = {"request_id": request_id, "git_commit": settings.tracing.git_commit,
                "llm_model": settings.deepseek.model, **request.model_dump(exclude={"question"})}
    return _compare(request, request_id, langsmith_extra={"run_id": request_id, "metadata": metadata})


def _compare_outputs(response: CompareResponse) -> dict:
    return {
        "vector_answer": response.vector.answer,
        "wiki_answer": response.wiki.answer,
        "errors": {"vector": response.vector.error, "wiki": response.wiki.error},
        "eval_run_id": response.run_id,
        "metrics": {metric.name: metric.value for metric in response.metrics},
    }


@traceable(name="ask_compare", process_inputs=lambda inputs: inputs["request"].model_dump(),
           process_outputs=_compare_outputs)
def _compare(request: AskRequest, request_id: str) -> CompareResponse:
    # ContextThreadPoolExecutor: hai luong con giu run hien tai, nen hai service noi vao cung mot trace.
    with ContextThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(_ask_vector_service, request)
        wiki_future = executor.submit(_ask_wiki_service, request)
        vector = vector_future.result()
        wiki = wiki_future.result()

    run_id, metrics = _evaluate(request.question, vector, wiki, request_id)
    return CompareResponse(question=request.question, vector=vector, wiki=wiki, metrics=metrics, run_id=run_id,
                           request_id=request_id)


@app.post("/ask/vector", response_model=ServiceAnswer)
def ask_vector(request: AskRequest) -> ServiceAnswer:
    return _ask_vector_service(request)


@app.post("/ask/wiki", response_model=ServiceAnswer)
def ask_wiki(request: AskRequest) -> ServiceAnswer:
    return _ask_wiki_service(request)


def _ask_vector_service(request: AskRequest) -> ServiceAnswer:
    return _call_answer("vector-rag-service", f"{SERVICES.vector}/answer", request)


def _ask_wiki_service(request: AskRequest) -> ServiceAnswer:
    return _call_answer("wiki-rag-service", f"{SERVICES.wiki}/wiki/answer", request)


def _call_answer(service: str, url: str, request: AskRequest) -> ServiceAnswer:
    try:
        with service_client(ANSWER_TIMEOUT) as client:
            response = client.post(url, json=request.model_dump())
            response.raise_for_status()
            return ServiceAnswer.model_validate(response.json())
    except Exception as exc:
        return ServiceAnswer(service=service, question=request.question, answer="", error=str(exc))


def _evaluate(question: str, vector: ServiceAnswer, wiki: ServiceAnswer, request_id: str):
    """Ghi lan chay vao evaluator; loi evaluator khong lam hong cau tra loi."""
    try:
        record = EvaluationRecord(question=question, vector_answer=vector, wiki_answer=wiki, request_id=request_id)
        with service_client() as client:
            response = client.post(f"{SERVICES.evaluator}/eval/run", json=record.model_dump())
            response.raise_for_status()
            data = response.json()
        return data.get("run_id"), data.get("metrics", [])
    except Exception as exc:
        logger.warning("Evaluator loi, bo qua metrics: %s", exc)
        return None, []


def _build_ui():
    def compare(question, top_k, candidate_k, use_llm):
        response = ask_compare(
            AskRequest(
                question=question,
                top_k=int(top_k),
                candidate_k=int(candidate_k),
                use_llm=bool(use_llm),
            )
        )
        return (
            _format_answer(response.vector),
            _format_answer(response.wiki),
            _format_metrics(response),
        )

    with gr.Blocks(title="RAG Compare Mode") as demo:
        gr.Markdown("# RAG Compare Mode\nSo sanh vector-rag-service va wiki-rag-service tren cung mot cau hoi.")
        question = gr.Textbox(label="Cau hoi", lines=3)
        with gr.Row():
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")
            candidate_k = gr.Slider(10, 100, value=40, step=5, label="Candidate K")
            use_llm = gr.Checkbox(value=True, label="Dung LLM")
        submit = gr.Button("Compare")
        with gr.Row():
            vector_output = gr.Textbox(label="Vector RAG", lines=24)
            wiki_output = gr.Textbox(label="Wiki RAG", lines=24)
        metrics_output = gr.Textbox(label="Metrics", lines=16)
        submit.click(compare, inputs=[question, top_k, candidate_k, use_llm], outputs=[vector_output, wiki_output, metrics_output])
    return demo


def _format_answer(answer: ServiceAnswer) -> str:
    lines = [
        f"Service: {answer.service}",
        f"Latency: {answer.latency_ms:.2f} ms",
    ]
    if answer.error:
        lines.append(f"ERROR: {answer.error}")
    lines.extend(["", "Answer:", answer.answer or "(empty)", "", "Contexts:"])
    for idx, context in enumerate(answer.contexts, 1):
        lines.append(f"[{idx}] {context.title}")
        lines.append(f"Citation: {context.citation}")
        lines.append(context.content[:900])
        lines.append("---")
    return "\n".join(lines)


def _format_metrics(response: CompareResponse) -> str:
    lines = [f"Run ID: {response.run_id}", f"Request ID (trace LangSmith): {response.request_id}", ""]
    for metric in response.metrics:
        lines.append(f"{metric.name}: {metric.value} - {metric.explanation}")
    return "\n".join(lines)


app = mount_gradio_app(app, _build_ui(), path="/ui")
