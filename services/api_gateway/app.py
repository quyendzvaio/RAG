import logging
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from gradio.routes import mount_gradio_app

from shared.config import get_settings
from shared.http import service_client
from shared.log import setup_logging
from shared.schemas import AskRequest, CompareResponse, EvaluationRecord, HealthResponse, ServiceAnswer


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
    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(_ask_vector_service, request)
        wiki_future = executor.submit(_ask_wiki_service, request)
        vector = vector_future.result()
        wiki = wiki_future.result()

    run_id, metrics = _evaluate(request.question, vector, wiki)
    return CompareResponse(question=request.question, vector=vector, wiki=wiki, metrics=metrics, run_id=run_id)


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
        with service_client() as client:
            response = client.post(url, json=request.model_dump())
            response.raise_for_status()
            return ServiceAnswer.model_validate(response.json())
    except Exception as exc:
        return ServiceAnswer(service=service, question=request.question, answer="", error=str(exc))


def _evaluate(question: str, vector: ServiceAnswer, wiki: ServiceAnswer):
    """Ghi lan chay vao evaluator; loi evaluator khong lam hong cau tra loi."""
    try:
        record = EvaluationRecord(question=question, vector_answer=vector, wiki_answer=wiki)
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
    lines = [f"Run ID: {response.run_id}", ""]
    for metric in response.metrics:
        lines.append(f"{metric.name}: {metric.value} - {metric.explanation}")
    return "\n".join(lines)


app = mount_gradio_app(app, _build_ui(), path="/ui")
