"""Danh gia retrieval cua vector va wiki service tren golden set.

Goi gateway (/ask/vector, /ask/wiki) voi use_llm=false. Mot cau tra loi dung khi context
co cung (source_file, article_code) voi Dieu goc. Chi dung thu vien chuan.

    python3 scripts/eval_retrieval.py --tag baseline
    python3 scripts/eval_retrieval.py --tag refactor --compare baseline
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SERVICES = {"vector": "/ask/vector", "wiki": "/ask/wiki"}
VARIANTS = {"accented": "question", "unaccented": "question_unaccented"}
RESULTS_DIR = Path("eval/results")


def ask(gateway: str, path: str, question: str, top_k: int, candidate_k: int) -> tuple:
    payload = {"question": question, "top_k": top_k, "candidate_k": candidate_k, "use_llm": False}
    request = urllib.request.Request(
        f"{gateway}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.load(response)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if data.get("error"):
        raise RuntimeError(f"{path}: {data['error']}")
    return data["contexts"], data.get("latency_ms", elapsed_ms)


def context_key(context: dict) -> list:
    metadata = context.get("metadata", {})
    return [metadata.get("source_file", ""), metadata.get("article_code", ""), context.get("id", "")]


def first_hit_rank(contexts: list, row: dict):
    for rank, context in enumerate(contexts, start=1):
        metadata = context.get("metadata", {})
        if metadata.get("source_file") == row["source_file"] and metadata.get("article_code") == row["article_code"]:
            return rank
    return None


def percentile(values: list, pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(round(pct / 100 * (len(ordered) - 1))), len(ordered) - 1)]


def summarize(ranks: list, latencies: list) -> dict:
    total = len(ranks)
    return {
        "n": total,
        "recall@1": round(sum(1 for r in ranks if r == 1) / total, 4),
        "recall@5": round(sum(1 for r in ranks if r is not None and r <= 5) / total, 4),
        "mrr": round(sum(1 / r for r in ranks if r) / total, 4),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
    }


def run(args) -> dict:
    rows = [json.loads(line) for line in Path(args.golden).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    result = {"tag": args.tag, "top_k": args.top_k, "candidate_k": args.candidate_k, "metrics": {}, "snapshot": {}}
    for service, path in SERVICES.items():
        for variant, field in VARIANTS.items():
            def evaluate(row):
                contexts, latency = ask(args.gateway, path, row[field], args.top_k, args.candidate_k)
                return first_hit_rank(contexts, row), latency, [context_key(c) for c in contexts]

            # Chay tuan tu de latency khong bi nhieu do tranh CPU.
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                outputs = list(executor.map(evaluate, rows))
            ranks = [o[0] for o in outputs]
            latencies = [o[1] for o in outputs]
            result["metrics"][f"{service}/{variant}"] = summarize(ranks, latencies)
            result["snapshot"][f"{service}/{variant}"] = [o[2] for o in outputs]
            print(f"{service}/{variant}: {result['metrics'][f'{service}/{variant}']}", flush=True)
    return result


def print_comparison(current: dict, previous: dict) -> None:
    print(f"\n{'metric':<34}{previous['tag']:>12}{current['tag']:>12}{'delta':>10}")
    for key, metrics in current["metrics"].items():
        old = previous["metrics"].get(key, {})
        for name, value in metrics.items():
            if name == "n":
                continue
            before = old.get(name)
            delta = f"{value - before:+.4g}" if isinstance(before, (int, float)) else ""
            print(f"{key + ' ' + name:<34}{str(before):>12}{str(value):>12}{delta:>10}")

    for key, snapshot in current["snapshot"].items():
        old = previous["snapshot"].get(key)
        if old is None:
            continue
        same = sum(1 for a, b in zip(snapshot, old) if a == b)
        print(f"snapshot {key}: {same}/{len(snapshot)} cau co top-k giong het")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--compare", help="Tag cu de so sanh")
    parser.add_argument("--golden", default="eval/golden.jsonl")
    parser.add_argument("--gateway", default="http://localhost:7860")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    result = run(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Da luu {RESULTS_DIR / (args.tag + '.json')}")

    if args.compare:
        previous_path = RESULTS_DIR / f"{args.compare}.json"
        if not previous_path.exists():
            sys.exit(f"Khong thay {previous_path}")
        print_comparison(result, json.loads(previous_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
