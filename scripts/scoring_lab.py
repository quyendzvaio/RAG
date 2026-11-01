"""Thu cac bien the scoring cua hybrid search tren golden set ma khong can restart service.

Load hybrid index mot lan, lay parent hits (vector + BM25) va lexical rerank cho moi cau mot lan,
roi cham tung bien the trong bo nho. Bien the "current" phai trung voi rag.query_engine.hybrid_search
(script kiem tra dieu nay). Chay trong image vi can rag/ va index:

    docker compose run --rm --no-deps -v ./scripts:/app/scripts:ro -v ./eval:/app/eval \\
        vector-rag-service python scripts/scoring_lab.py --tag lab-baseline
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.bm25_store import document_key  # noqa: E402
from rag.embeddings import get_embedding_model  # noqa: E402
from rag.query_engine import (  # noqa: E402
    BM25_WEIGHT,
    RERANK_WEIGHT,
    RRF_K,
    VECTOR_WEIGHT,
    RankedDocument,
    _lexical_rerank_score,
    _rerank_text,
    boost_article_intent,
    candidate_hits,
    context_text,
    expand_abbreviations,
    hybrid_search,
)
from rag.vector_store import load_hybrid_vectorstore  # noqa: E402

VARIANTS = {"accented": "question", "unaccented": "question_unaccented"}
RERANK_WEIGHTS = (0.0, 0.3, 0.5, 0.65, 0.8, 1.0)


def article_of(document) -> tuple:
    metadata = document.metadata or {}
    return metadata.get("source_file", ""), metadata.get("article_code", "")


def fuse(vector_hits, bm25_hits, vector_weight: float, bm25_weight: float) -> list:
    """Giong _merge_ranked_hits: tra ve [(key, document, fused_score)] theo thu tu sort cua hybrid_search."""
    scores, documents = {}, {}
    for hits, weight in ((vector_hits, vector_weight), (bm25_hits, bm25_weight)):
        for rank, (document, _raw) in enumerate(hits, start=1):
            key = document_key(document)
            documents.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rank + RRF_K))
    order = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [(key, documents[key], scores[key]) for key in order]


def score_variant(
    fused, rerank_scores: dict, rerank_weight: float, normalize_fused: bool, candidate_k: int, intent_question: str = ""
) -> list:
    """`intent_question`: cong INTENT_BOOST nhu hybrid_search (bien the "current")."""
    pool = fused[:candidate_k]
    if not pool:
        return []
    raw = [rerank_scores[key] for key, _doc, _score in pool]
    r_min, r_range = min(raw), max(max(raw) - min(raw), 1e-9)
    fused_values = [score for _key, _doc, score in pool]
    f_min, f_range = min(fused_values), max(max(fused_values) - min(fused_values), 1e-9)
    final = []
    for (key, document, fused_score), rerank_score in zip(pool, raw):
        fused_part = (fused_score - f_min) / f_range if normalize_fused else fused_score
        normalized = (rerank_score - r_min) / r_range
        final.append(RankedDocument(document=document, score=rerank_weight * normalized + (1 - rerank_weight) * fused_part))
    final.sort(key=lambda item: item.score, reverse=True)
    if intent_question:
        final = boost_article_intent(intent_question, final)
    return [item.document for item in final]


def collapse_articles(documents: list) -> list:
    seen, kept = set(), []
    for document in documents:
        article = article_of(document)
        if article not in seen:
            seen.add(article)
            kept.append(document)
    return kept


def first_hit_rank(documents: list, row: dict, top_k: int):
    target = (row["source_file"], row["article_code"])
    for rank, document in enumerate(documents[:top_k], start=1):
        if article_of(document) == target:
            return rank
    return None


def summarize(ranks: list) -> dict:
    total = len(ranks) or 1
    return {
        "recall@1": round(sum(1 for r in ranks if r == 1) / total, 4),
        "recall@5": round(sum(1 for r in ranks if r is not None and r <= 5) / total, 4),
        "mrr": round(sum(1.0 / r for r in ranks if r) / total, 4),
    }


def build_variants() -> dict:
    variants = {"current": dict(rerank_weight=RERANK_WEIGHT, normalize_fused=True, collapse=False, intent=True)}
    for weight in RERANK_WEIGHTS:
        for collapse in (False, True):
            name = f"minmax w={weight}" + (" +collapse" if collapse else "")
            variants[name] = dict(rerank_weight=weight, normalize_fused=True, collapse=collapse)
    variants["raw fused (truoc ban sua)"] = dict(rerank_weight=RERANK_WEIGHT, normalize_fused=False, collapse=False)
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--golden", default="eval/golden.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--check", type=int, default=20, help="So cau doi chieu 'current' voi hybrid_search")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.golden).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    store = load_hybrid_vectorstore(get_embedding_model())
    variants = build_variants()
    ranks = {(name, variant): [] for name in variants for variant in VARIANTS}

    mismatches = 0
    for index, row in enumerate(rows):
        for variant, field in VARIANTS.items():
            question = row[field]
            vector_hits, bm25_hits, matched = candidate_hits(question, store, args.candidate_k)
            fused = fuse(vector_hits, bm25_hits, VECTOR_WEIGHT, BM25_WEIGHT)
            rerank_scores = {}
            rerank_query = expand_abbreviations(question)  # nhu rag.query_engine.rerank
            for key, doc, _score in fused[: args.candidate_k]:
                ranked = RankedDocument(document=doc, score=0.0, matched_children=matched.get(key, []))
                text = _rerank_text(doc.metadata, context_text(rerank_query, store, ranked))
                rerank_scores[key] = _lexical_rerank_score(rerank_query, text, doc.metadata)

            for name, spec in variants.items():
                ordered = score_variant(
                    fused, rerank_scores, spec["rerank_weight"], spec["normalize_fused"], args.candidate_k,
                    question if spec.get("intent") else "",
                )
                if spec["collapse"]:
                    ordered = collapse_articles(ordered)
                ranks[(name, variant)].append(first_hit_rank(ordered, row, args.top_k))

            if index < args.check:
                expected = [document_key(item.document) for item in hybrid_search(question, store, args.top_k, args.candidate_k)]
                ours = score_variant(fused, rerank_scores, RERANK_WEIGHT, True, args.candidate_k, question)[: args.top_k]
                if [document_key(doc) for doc in ours] != expected:
                    mismatches += 1
        print(f"{index + 1}/{len(rows)}", end="\r", file=sys.stderr, flush=True)

    if mismatches:
        print(f"CANH BAO: 'current' lech hybrid_search o {mismatches} truy van", file=sys.stderr)

    result = {"tag": args.tag, "top_k": args.top_k, "candidate_k": args.candidate_k, "questions": len(rows),
              "check_mismatches": mismatches, "variants": {}}
    print(f"\n{'bien the':28} | {'co dau R@1  R@5   MRR':24} | {'khong dau R@1  R@5   MRR':24}")
    for name in variants:
        row_summary = {variant: summarize(ranks[(name, variant)]) for variant in VARIANTS}
        result["variants"][name] = row_summary
        cells = [f"{s['recall@1']:.3f} {s['recall@5']:.3f} {s['mrr']:.3f}" for s in row_summary.values()]
        print(f"{name:28} | {cells[0]:24} | {cells[1]:24}")

    out = Path("eval/results") / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Da luu {out}")


if __name__ == "__main__":
    main()
