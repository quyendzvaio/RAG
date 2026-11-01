"""Cham lai bang Sonnet cac cau vector/golden nghi giam khao deepseek cham sai: context da co Dieu dap an nhung
faithfulness hoac context recall < 0.5. Cac cau con lai giu diem deepseek; ghi tong hop vao eval/results/<TAG>.json.

Diem da cham nam trong <TAG>-scores.jsonl nen chay lai khong ton token. Chay voi phu thuoc cua make eval-ragas:
  docker compose run --rm --no-deps -e PYTHONPATH=/deps:/app -v rag_eval_deps:/deps:ro -v ./scripts:/app/scripts:ro \
      -v ./eval:/app/eval vector-rag-service python eval/rejudge_sonnet.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_ragas as E  # noqa: E402
from eval_retrieval import first_hit_rank  # noqa: E402

SRC, TAG = "ragas-qwen14b-v2", "ragas-qwen14b-v2-sonnet"
BASE_JUDGE, JUDGE = "cl/deepseek/deepseek-v4-flash", "ag/claude-sonnet-4-6"
SUSPECT_BELOW = 0.5

gold = {row["question"]: row for row in E.read_jsonl(Path("eval/golden.jsonl"))}
answers = {(d["service"], d["question"]): d for d in E.read_jsonl(E.RESULTS_DIR / f"{SRC}-answers.jsonl")}
base = {d["question"]: d for d in E.read_jsonl(E.RESULTS_DIR / f"{SRC}-scores.jsonl")
        if d["service"] == "vector" and d["judge"] == BASE_JUDGE}


def suspect(question: str) -> bool:
    if first_hit_rank(answers[("vector", question)]["contexts"], gold[question]) is None:
        return False
    row = base[question]
    return (row["faithfulness"] or 0) < SUSPECT_BELOW or (row["context_recall"] or 0) < SUSPECT_BELOW


def base_row(question: str) -> dict:
    answer = answers[("vector", question)]
    return {**base[question], "answered": bool(answer["contexts"]) and not answer.get("error"),
            "tokens": answer.get("token_usage", {}).get("total_tokens")}


suspects = [q for q in gold if suspect(q)]
print(f"Cham lai {len(suspects)} cau", file=sys.stderr)

E.judge_model = JUDGE
questions = [{"set": "golden", "covered": False, **gold[q]} for q in suspects]
references = E.ensure_references(questions, 2)
llm = E.LangchainLLMWrapper(E.LLMServiceChat())
embeddings = E.LangchainEmbeddingsWrapper(E.get_embedding_model())
rejudged = {row["question"]: row for row in
            E.score(TAG, "vector", questions, references, answers, E.build_metrics(llm, embeddings), llm, embeddings, 2)}

old_rows = [base_row(q) for q in gold]
combined = [{**row, **{m: rejudged[row["question"]][m] for m in E.METRIC_NAMES}} if row["question"] in rejudged else row
            for row in old_rows]
summary = {"deepseek": E.summarize(old_rows), "ket_hop": E.summarize(combined)}
for q in suspects:
    print(f"{q[:60]:<60} F {base[q]['faithfulness']} -> {rejudged[q]['faithfulness']}  "
          f"CR {base[q]['context_recall']} -> {rejudged[q]['context_recall']}")
print(json.dumps(summary, indent=1))
out = E.RESULTS_DIR / f"{TAG}.json"
# Chay lai tu diem da luu thi khong goi Sonnet: giu token cua lan cham that.
usage = dict(E._usage) or (json.loads(out.read_text(encoding="utf-8"))["judge_usage"] if out.exists() else {})
out.write_text(json.dumps({"suspect": suspects, "summary": summary, "judge_usage": usage}, ensure_ascii=False, indent=1),
               encoding="utf-8")
