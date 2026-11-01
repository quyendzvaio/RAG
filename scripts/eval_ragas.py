"""Danh gia vector va wiki service bang RAGAS: faithfulness, answer relevancy, context precision, context recall.

Chay trong image, ragas nam o volume rieng (make eval-ragas-deps, make eval-ragas). Giam khao la LLM cua project
(qua llm-service), embedding la model embedding cua project. Bo cau hoi:
- golden: eval/golden.jsonl (cau co dau), reference = cau tra loi mau LLM viet chi tu Dieu dap an.
- synthesis: eval/synthesis.jsonl (cau can nhieu Dieu), reference = cau tra loi mau tu toan bo Dieu cua de muc.
Reference, cau tra loi cua service va diem (luu sau moi lo SCORE_BATCH mau, theo giam khao) duoc cache trong eval/,
nen chay lai sau khi bi ngat chi lam phan con thieu.

    python scripts/eval_ragas.py --tag ragas-v1 --limit 5
"""
import argparse
import asyncio
import json
import math
import sys
import time
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

import httpx

# ragas import ChatVertexAI/VertexAI chi de kiem tra isinstance; langchain-community 0.4 da bo ca hai.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ImportError:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[_stub.__name__] = _stub
import langchain_community.llms as _community_llms

try:
    _community_llms.VertexAI
except Exception:
    _community_llms.VertexAI = type("VertexAI", (), {})

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from rag.embeddings import get_embedding_model
from rag.wiki_builder import read_pages
from shared.answering import LLM_CONTEXT_CHARS, WIKI_CONTEXT_CHARS, format_context
from shared.llm import complete
from shared.schemas import ChatMessage, ContextItem

EVAL_DIR = Path("eval")
RESULTS_DIR = EVAL_DIR / "results"
REFERENCES = EVAL_DIR / "ragas" / "references.jsonl"
PROMPTS_DIR = EVAL_DIR / "ragas" / "prompts"
CONTEXT_CHARS = {"vector": LLM_CONTEXT_CHARS, "wiki": WIKI_CONTEXT_CHARS}  # dung bang phan LLM tra loi da thay
METRIC_NAMES = ("faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall")
JUDGE_MAX_TOKENS = 4000
# Khung prompt cua ragas bang tieng Anh: model nho (qwen3-14b) viet cau hoi/menh de bang tieng Anh du cau lenh
# da dan tieng Viet, va answer relevancy so cau hoi do voi cau hoi goc bang embedding chi hieu tieng Viet
# (cosine ~0.2 thay vi ~0.6). System message nay ep moi noi dung sinh ra la tieng Viet.
JUDGE_SYSTEM = ("Mọi nội dung bạn viết (câu hỏi, mệnh đề, lý do, giải thích) phải bằng tiếng Việt, cùng ngôn ngữ với "
                "dữ liệu đầu vào, kể cả khi hướng dẫn viết bằng tiếng Anh. Giữ nguyên tên khóa JSON theo schema.")
REFERENCE_MAX_TOKENS = 2500
ROLES = {"human": "user", "ai": "assistant", "system": "system"}
RETRY_DELAYS = (5, 15, 30, 60, 120, 240)  # giay; proxy LLM tra 429 khi qua nhieu request
# Loi het tien/het quota (kiraai: 402 "Insufficient VND wallet balance"; opencode free: 429 "Free usage limit
# reached", retry-after ~14 h): thu lai vo ich, dung ca lan do.
QUOTA_ERROR_MARKERS = ("402 payment required", "insufficient", "quota", "balance", "usage limit")
MAX_FAILED_ANSWERS = 5  # so cau tra loi loi lien tiep (da het luot thu) thi dung: provider hong, khong phai cau kho
SCORE_BATCH = 12  # so mau moi lan evaluate; diem luu sau moi lo
LLM_FALLBACK_PREFIX = "LLM fallback:"  # llm-service tra text nay (fallback=True) khi DeepSeek loi
# Model giam khao (va viet reference), --judge-model; None = model llm-service dang dung de tra loi.
# Giu co dinh khi so sanh model tra loi, de diem doi la do cau tra loi chu khong do nguoi cham.
judge_model: Optional[str] = None


class Retryable(Exception):
    pass


class Fatal(Exception):
    """Loi khong thu lai duoc (het tien/quota): dung ca lan do, giu phan da cache."""


_fatal: list = []  # loi Fatal dau tien; ragas nuot exception trong job nen sau moi lo phai kiem tra


def check_fatal() -> None:
    if _fatal:
        raise Fatal(_fatal[0])


def with_retry(call, what: str):
    for delay in (0,) + RETRY_DELAYS:
        check_fatal()
        if delay:
            time.sleep(delay)
        try:
            return call()
        except (Retryable, httpx.HTTPError) as exc:
            last = exc
            if any(marker in str(exc).lower() for marker in QUOTA_ERROR_MARKERS):
                _fatal.append(f"{what}: {str(exc)[:300]}")
                raise Fatal(_fatal[0]) from exc
            print(f"Thu lai sau loi ({what}): {str(exc)[:120]}", file=sys.stderr)
    raise RuntimeError(f"Het so lan thu ({what}): {last}")


def complete_retry(messages, max_tokens: int, **kwargs):
    def call():
        completion = complete(messages, max_tokens, model=judge_model, **kwargs)
        if completion.fallback:
            raise Retryable(completion.error)
        return completion

    return with_retry(call, "llm-service")


class LLMServiceChat(BaseChatModel):
    """Chat model LangChain goi llm-service, de ragas dung LLM cua project lam giam khao."""

    temperature: float = 0.0
    max_tokens: int = JUDGE_MAX_TOKENS

    @property
    def _llm_type(self) -> str:
        return "llm-service"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        chat = [ChatMessage(role="system", content=JUDGE_SYSTEM)]
        chat += [ChatMessage(role=ROLES.get(message.type, "user"), content=str(message.content)) for message in messages]
        completion = complete_retry(chat, self.max_tokens, temperature=self.temperature)
        record_usage(completion.token_usage)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=completion.content))])


_usage: Counter = Counter()
_usage_lock = Lock()


def record_usage(usage: dict) -> None:
    with _usage_lock:
        _usage.update(usage)
        _usage["calls"] += 1


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def wiki_source_files() -> set:
    """File HTML cua cac de muc da co trang wiki: cau golden thuoc cac file nay la cau wiki tra loi duoc."""
    wiki_dir = Path("wiki_data/wiki")
    if not (wiki_dir / "index.md").exists():
        return set()
    return {meta.get("source_file") for _, meta, _ in read_pages(wiki_dir) if meta.get("type") == "de_muc"}


def load_questions(which: str, limit: int) -> list:
    rows = []
    if which in ("golden", "all"):
        golden = read_jsonl(EVAL_DIR / "golden.jsonl")[: limit or None]
        rows += [{"set": "golden", **row} for row in golden]
    if which in ("synthesis", "all"):
        synthesis = read_jsonl(EVAL_DIR / "synthesis.jsonl")[: limit or None]
        rows += [{"set": "synthesis", **row} for row in synthesis]
    return rows


def ensure_references(questions: list, workers: int) -> dict:
    """question -> reference. Thieu thi LLM viet tu Dieu goc trong raw sources (khong qua service nao)."""
    cached = {row["question"]: row["reference"] for row in read_jsonl(REFERENCES)}
    missing = [row for row in questions if row["question"] not in cached]
    if missing:
        articles = read_jsonl(Path("wiki_data/raw_sources/articles.jsonl"))
        with ThreadPoolExecutor(workers) as pool:
            jobs = [pool.submit(_write_reference, row, articles) for row in missing]
            for done, job in enumerate(as_completed(jobs), 1):
                row = job.result()
                append_jsonl(REFERENCES, [row])  # ghi ngay: loi giua chung khong mat phan da viet
                cached[row["question"]] = row["reference"]
                if done % 20 == 0 or done == len(jobs):
                    print(f"Da viet {done}/{len(jobs)} reference", file=sys.stderr)
    return cached


def _write_reference(row: dict, articles: list) -> dict:
    if row["set"] == "golden":
        sources = [a for a in articles if a["metadata"]["source_file"] == row["source_file"]
                   and a["metadata"]["article_code"] == row["article_code"]]
    else:
        sources = [a for a in articles if a["metadata"]["source_file"] == row["source_file"]]
    text = "\n\n".join(f"{a['article_heading']}\n{a['content']}" for a in sources)
    prompt = (
        "Dưới đây là các Điều luật. Trả lời câu hỏi chỉ dựa trên các Điều này: đầy đủ các ý câu hỏi cần, chính xác "
        "số liệu, ngắn gọn, nêu mã Điều làm căn cứ. Không thêm kiến thức ngoài.\n\n"
        f"Câu hỏi: {row['question']}\n\nCác Điều:\n\n{text}"
    )
    completion = complete_retry([ChatMessage(role="user", content=prompt)], REFERENCE_MAX_TOKENS, temperature=0.0,
                                timeout=httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0))
    if completion.truncated:
        raise RuntimeError(f"Reference bi cat: {row['question']}")
    record_usage(completion.token_usage)
    return {"question": row["question"], "set": row["set"], "reference": completion.content}


def ensure_answers(tag: str, questions: list, services: list, gateway: str, workers: int) -> dict:
    """(service, question) -> ServiceAnswer (dict); cache o eval/results/<tag>-answers.jsonl."""
    path = RESULTS_DIR / f"{tag}-answers.jsonl"
    cached = {(row["service"], row["question"]): row for row in read_jsonl(path)}
    jobs = [(service, row["question"]) for service in services for row in questions if (service, row["question"]) not in cached]

    def ask(job):
        service, question = job
        payload = {"question": question, "top_k": 5, "candidate_k": 40, "use_llm": True}

        def call():
            response = httpx.post(f"{gateway}/ask/{service}", json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            # Service tra loi loi bang HTTP 200 (error=..., hoac text fallback cua llm-service): khong cache.
            if data.get("error") or data.get("answer", "").startswith(LLM_FALLBACK_PREFIX):
                raise Retryable(data.get("error") or data["answer"][:120])
            return {**data, "service": service, "question": question}  # ServiceAnswer cung co hai khoa nay

        try:
            return with_retry(call, f"{service}: {question[:40]}")
        except RuntimeError as exc:
            # Loi lap lai (vd. model suy luan lau hon timeout 60 s cua proxy): ghi la khong tra loi duoc
            # (answered=False khi cham) thay vi dung ca lan do. Xoa dong nay trong cache de thu lai.
            return {"service": service, "question": question, "answer": "", "contexts": [], "error": str(exc)[:500]}

    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(jobs), workers * 4):
            batch = list(pool.map(ask, jobs[start : start + workers * 4]))
            good = [row for row in batch if not row.get("error")]
            failed = 0 if len(good) == len(batch) else failed + len(batch) - len(good)
            if failed >= MAX_FAILED_ANSWERS and not good:
                append_jsonl(path, good)
                raise Fatal(f"{failed} cau tra loi loi lien tiep, dung: {batch[-1]['error'][:300]}")
            append_jsonl(path, batch)
            cached.update({(row["service"], row["question"]): row for row in batch})
            print(f"Da lay {min(start + workers * 4, len(jobs))}/{len(jobs)} cau tra loi", file=sys.stderr)
    return cached


def build_metrics(llm, embeddings) -> list:
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    for metric in metrics:
        # Prompt few-shot cua ragas bang tieng Anh: dich sang tieng Viet mot lan roi cache, de vi du cung ngon ngu
        # voi du lieu (quan trong voi answer relevancy: cau hoi sinh lai phai la tieng Viet moi so duoc embedding).
        directory = PROMPTS_DIR / metric.name
        try:
            if directory.exists():
                metric.load_prompts(str(directory), "vietnamese")
            else:
                metric.set_prompts(**asyncio.run(metric.adapt_prompts("vietnamese", llm)))
                directory.mkdir(parents=True, exist_ok=True)
                metric.save_prompts(str(directory))
        except Exception as exc:
            print(f"Khong dich duoc prompt cua {metric.name}, dung tieng Anh: {exc}", file=sys.stderr)
    return metrics


def score(tag: str, service: str, questions: list, references: dict, answers: dict, metrics: list, llm, embeddings,
          workers: int) -> list:
    """Diem tung cau; cham theo lo SCORE_BATCH va luu vao <tag>-scores.jsonl sau moi lo (theo giam khao)."""
    path = RESULTS_DIR / f"{tag}-scores.jsonl"
    judge = judge_model or "llm-service"
    done = {row["question"]: row for row in read_jsonl(path) if row["service"] == service and row["judge"] == judge}
    rows, pending = [], []
    for row in questions:
        answer = answers[(service, row["question"])]
        contexts = [format_context(ContextItem(**context), CONTEXT_CHARS[service]) for context in answer.get("contexts", [])]
        answered = bool(contexts) and not answer.get("error")
        rows.append({"set": row["set"], "question": row["question"], "answered": answered, "covered": row["covered"],
                     "error": answer.get("error"), "tokens": answer.get("token_usage", {}).get("total_tokens")})
        if answered and row["question"] in done:
            rows[-1].update({name: done[row["question"]].get(name) for name in METRIC_NAMES})
        elif answered:
            sample = SingleTurnSample(user_input=row["question"], response=answer["answer"],
                                      retrieved_contexts=contexts, reference=references[row["question"]])
            pending.append((rows[-1], sample))
    if done:
        print(f"{service}: {len(rows) - len(pending)} cau da co diem, cham {len(pending)} cau", file=sys.stderr)
    for start in range(0, len(pending), SCORE_BATCH):
        batch = pending[start : start + SCORE_BATCH]
        result = evaluate(EvaluationDataset(samples=[sample for _, sample in batch]), metrics=metrics, llm=llm,
                          embeddings=embeddings, run_config=RunConfig(max_workers=workers, timeout=900, max_retries=3),
                          show_progress=False)
        check_fatal()  # lo bi het tien giua chung: diem thieu, khong luu
        for (row, _), record in zip(batch, result.to_pandas().to_dict("records")):
            row.update({name: _float(record.get(name)) for name in METRIC_NAMES})
        append_jsonl(path, [{"service": service, "question": row["question"], "judge": judge,
                             **{name: row[name] for name in METRIC_NAMES}} for row, _ in batch])
        print(f"{service}: da cham {min(start + SCORE_BATCH, len(pending))}/{len(pending)} cau", file=sys.stderr)
    return rows


def _float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def summarize(rows: list) -> dict:
    answered = [row for row in rows if row["answered"]]
    summary = {"n": len(rows), "answered": len(answered)}
    for name in METRIC_NAMES:
        values = [row[name] for row in answered if row.get(name) is not None]
        summary[name] = round(sum(values) / len(values), 4) if values else None
        summary[f"{name}_scored"] = len(values)
    # Cau khong co context thi khong tim lai duoc gi tu reference: tinh recall = 0 tren toan bo cau hoi.
    recalls = [row.get("context_recall") or 0.0 for row in rows]
    summary["context_recall_all"] = round(sum(recalls) / len(recalls), 4) if recalls else None
    tokens = [row["tokens"] for row in rows if row.get("tokens")]
    summary["answer_tokens_mean"] = round(sum(tokens) / len(tokens)) if tokens else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--set", default="all", choices=("golden", "synthesis", "all"))
    parser.add_argument("--limit", type=int, default=0, help="Chi lay N cau dau moi bo (0 = tat ca)")
    parser.add_argument("--services", default="vector,wiki")
    parser.add_argument("--gateway", default="http://api-gateway:7860")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--judge-model", default=None, help="Model giam khao (mac dinh: model cua llm-service)")
    parser.add_argument("--answers-only", action="store_true", help="Chi lay va cache cau tra loi, khong cham")
    args = parser.parse_args()
    global judge_model
    judge_model = args.judge_model

    services = args.services.split(",")
    questions = load_questions(args.set, args.limit)
    covered = wiki_source_files()
    for row in questions:
        row["covered"] = row["source_file"] in covered
    references = ensure_references(questions, args.workers)
    answers = ensure_answers(args.tag, questions, services, args.gateway, args.workers)
    if args.answers_only:
        return
    print(f"Token reference + tra loi: {dict(_usage)}", file=sys.stderr)
    _usage.clear()

    llm = LangchainLLMWrapper(LLMServiceChat())
    embeddings = LangchainEmbeddingsWrapper(get_embedding_model())
    metrics = build_metrics(llm, embeddings)

    report = {"tag": args.tag, "set": args.set, "limit": args.limit, "judge_model": judge_model, "summary": {}, "rows": {}}
    for service in services:
        rows = score(args.tag, service, questions, references, answers, metrics, llm, embeddings, args.workers)
        report["rows"][service] = rows
        for subset in sorted({row["set"] for row in rows}):
            report["summary"][f"{service}/{subset}"] = summarize([row for row in rows if row["set"] == subset])
        # Cau golden thuoc de muc co wiki: nhom so sanh cong bang giua hai service.
        golden_covered = [row for row in rows if row["set"] == "golden" and row["covered"]]
        if golden_covered:
            report["summary"][f"{service}/golden-wiki-de-muc"] = summarize(golden_covered)
    report["judge_usage"] = dict(_usage)
    (RESULTS_DIR / f"{args.tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    columns = ("n", "answered") + METRIC_NAMES + ("context_recall_all", "answer_tokens_mean")
    print("\n" + "group".ljust(20) + "".join(name[:14].rjust(16) for name in columns))
    for group, summary in report["summary"].items():
        print(group.ljust(20) + "".join(str(summary.get(name)).rjust(16) for name in columns))
    print(f"\nToken giam khao: {report['judge_usage']}")


if __name__ == "__main__":
    try:
        main()
    except Fatal as exc:
        sys.exit(f"Dung danh gia (da luu phan cache): {exc}")
