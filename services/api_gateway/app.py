import os
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from gradio.routes import mount_gradio_app

from shared.http import service_client
from shared.schemas import AskRequest, CompareRequest, CompareResponse, EvaluationRecord, HealthResponse, ServiceAnswer


VECTOR_SERVICE_URL = os.getenv("VECTOR_SERVICE_URL", "http://vector-rag-service:8001")
WIKI_SERVICE_URL = os.getenv("WIKI_SERVICE_URL", "http://wiki-rag-service:8002")
EVALUATOR_SERVICE_URL = os.getenv("EVALUATOR_SERVICE_URL", "http://evaluator-service:8004")

app = FastAPI(title="api-gateway", version="0.1.0")


@app.get("/")
def root():
    return RedirectResponse(url="/ui")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(service="api-gateway")


@app.post("/ask/compare", response_model=CompareResponse)
def ask_compare(request: CompareRequest) -> CompareResponse:
    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(_call_answer, f"{VECTOR_SERVICE_URL}/answer", request)
        wiki_future = executor.submit(_call_answer, f"{WIKI_SERVICE_URL}/wiki/answer", request)
        vector = vector_future.result()
        wiki = wiki_future.result()

    metrics = []
    run_id = None
    try:
        eval_record = EvaluationRecord(question=request.question, vector_answer=vector, wiki_answer=wiki)
        with service_client() as client:
            response = client.post(f"{EVALUATOR_SERVICE_URL}/eval/run", json=eval_record.model_dump())
            response.raise_for_status()
            data = response.json()
        run_id = data.get("run_id")
        metrics = data.get("metrics", [])
    except Exception:
        metrics = []

    return CompareResponse(
        question=request.question,
        vector=vector,
        wiki=wiki,
        metrics=metrics,
        run_id=run_id,
    )


@app.post("/ask/vector", response_model=ServiceAnswer)
def ask_vector(request: AskRequest) -> ServiceAnswer:
    return _call_answer(f"{VECTOR_SERVICE_URL}/answer", request)


@app.post("/ask/wiki", response_model=ServiceAnswer)
def ask_wiki(request: AskRequest) -> ServiceAnswer:
    return _call_answer(f"{WIKI_SERVICE_URL}/wiki/answer", request)


def _call_answer(url: str, request) -> ServiceAnswer:
    try:
        payload = {
            "question": request.question,
            "top_k": request.top_k,
            "candidate_k": request.candidate_k,
            "use_llm": request.use_llm,
        }
        with service_client() as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return ServiceAnswer.model_validate(response.json())
    except Exception as exc:
        service = "vector-rag-service" if "vector" in url else "wiki-rag-service"
        return ServiceAnswer(service=service, question=request.question, answer="", error=str(exc))


def _build_ui():
    def compare(question, top_k, candidate_k, use_llm):
        response = ask_compare(
            CompareRequest(
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
        if hasattr(metric, "model_dump"):
            data = metric.model_dump()
        else:
            data = metric
        lines.append(f"{data.get('name')}: {data.get('value')} - {data.get('explanation', '')}")
    return "\n".join(lines)


app = mount_gradio_app(app, _build_ui(), path="/ui")
