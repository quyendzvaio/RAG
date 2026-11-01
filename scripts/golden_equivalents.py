"""Tim Dieu tuong duong voi Dieu goc cua moi cau golden, de eval retrieval khong tinh la truot khi he thong lay
mot Dieu khac cung noi dung (luat, nghi dinh, thong tu cung de muc thuong lap lai nhau).

Ung vien: cac Dieu trong top-k cua cac lan eval retrieval da luu (`--runs`) ma khong phai Dieu goc. LLM (qua
llm-service) doc cau hoi, cau tra loi mau (eval/ragas/references.jsonl) va tung ung vien, chon Dieu tu no tra loi
duoc cung noi dung VA dieu chinh dung doi tuong cau hoi hoi (khong nhan Dieu linh vuc khac chi trung cau chu).
Ket qua cache theo cau hoi trong eval/golden_equivalents.jsonl; chay lai chi hoi cau/ung vien moi. Chay trong image:

    docker compose run --rm --no-deps -v ./scripts:/app/scripts:ro -v ./eval:/app/eval \\
        vector-rag-service python scripts/golden_equivalents.py --runs meta-0 v4-intent --model cl/deepseek/deepseek-v4-flash
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import get_settings  # noqa: E402
from shared.llm import complete  # noqa: E402
from shared.schemas import ChatMessage  # noqa: E402

EVAL_DIR = Path("eval")
OUTPUT = EVAL_DIR / "golden_equivalents.jsonl"
ARTICLE_CHARS = 3000  # moi ung vien; Dieu dai hon bi cat (du de so noi dung voi cau tra loi mau)
MAX_TOKENS = 400
ANSWER_RE = re.compile(r"\[[^\[\]]*\]")
RETRY_DELAYS = (0, 5, 15, 30, 60, 120)
QUOTA_ERROR_MARKERS = ("402", "insufficient", "balance", "usage limit")


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def load_articles() -> dict:
    """(source_file, article_code) -> (tieu de, noi dung) cua phan "article" (bo phu luc)."""
    articles = {}
    for record in read_jsonl(Path(get_settings().paths.parents)):
        metadata = record["metadata"]
        if metadata.get("part") == "article":
            key = (metadata["source_file"], metadata["article_code"])
            articles.setdefault(key, (f"{metadata['article_code']} {metadata.get('article_title', '')}", record["page_content"]))
    return articles


def candidates(rows: list, runs: list) -> dict:
    """question -> [(source_file, article_code)] khac Dieu goc, theo thu tu xuat hien trong cac lan eval."""
    found = {row["question"]: [] for row in rows}
    for run in runs:
        snapshot = json.loads((EVAL_DIR / "results" / f"{run}.json").read_text(encoding="utf-8"))["snapshot"]
        for row, contexts in zip(rows, snapshot.get("vector/accented", [])):
            gold = (row["source_file"], row["article_code"])
            for source_file, article_code, _id in contexts:
                key = (source_file, article_code)
                if key != gold and key not in found[row["question"]]:
                    found[row["question"]].append(key)
    return found


def judge(row: dict, reference: str, keys: list, articles: dict, model: str) -> tuple:
    listing = "\n\n".join(
        f"[{index}] {articles[key][0]}\n{articles[key][1][:ARTICLE_CHARS]}" for index, key in enumerate(keys, start=1)
    )
    prompt = (
        f"Câu hỏi: {row['question']}\n\nCâu trả lời mẫu (rút từ {row['article_code']}): {reference}\n\n"
        f"Các Điều luật ứng viên:\n\n{listing}\n\n"
        "Chọn các Điều mà CHỈ RIÊNG Điều đó đã đủ để trả lời câu hỏi với cùng nội dung như câu trả lời mẫu "
        "(cùng đối tượng, cùng quy định, cùng con số nếu có) VÀ quy định đúng lĩnh vực, đối tượng mà câu hỏi hỏi. "
        "Không chọn Điều chỉ trùng câu chữ nhưng thuộc lĩnh vực hoặc văn bản điều chỉnh đối tượng khác, "
        "không chọn Điều chỉ liên quan một phần.\n"
        "Chỉ trả lời một danh sách JSON số thứ tự, ví dụ [2] hoặc [] nếu không có Điều nào."
    )
    for delay in RETRY_DELAYS:
        time.sleep(delay)
        completion = complete([ChatMessage(role="user", content=prompt)], MAX_TOKENS, temperature=0.0, model=model)
        error = (completion.error or "").lower()
        # Chi thu lai loi tam thoi (429 qua nhanh, 5xx); het tien/quota thi dung ngay.
        if not completion.fallback or any(marker in error for marker in QUOTA_ERROR_MARKERS):
            break
    if completion.fallback:
        raise RuntimeError(f"LLM loi: {completion.error}")
    match = ANSWER_RE.search(completion.content)
    picked = json.loads(match.group(0)) if match else []
    accepted = [keys[index - 1] for index in picked if isinstance(index, int) and 1 <= index <= len(keys)]
    return accepted, completion.token_usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, help="Tag eval retrieval lay ung vien (eval/results/<tag>.json)")
    parser.add_argument("--model", default=None, help="Model qua llm-service (mac dinh: model cua llm-service)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Chi hoi N cau dau (0 = tat ca)")
    args = parser.parse_args()

    rows = read_jsonl(EVAL_DIR / "golden.jsonl")
    references = {row["question"]: row["reference"] for row in read_jsonl(EVAL_DIR / "ragas" / "references.jsonl")}
    articles = load_articles()
    cached = {row["question"]: row for row in read_jsonl(OUTPUT)}
    found = candidates(rows, args.runs)

    jobs = []
    for row in rows:
        done = cached.get(row["question"], {"checked": []})
        checked = {tuple(key) for key in done["checked"]}
        pending = [key for key in found[row["question"]] if key not in checked and key in articles]
        if pending:
            jobs.append((row, pending))
    jobs = jobs[: args.limit or None]

    def run(job):
        row, pending = job
        accepted, usage = judge(row, references.get(row["question"], ""), pending, articles, args.model)
        return row, pending, accepted, usage

    total = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for count, (row, pending, accepted, usage) in enumerate(pool.map(run, jobs), start=1):
            entry = cached.setdefault(row["question"], {
                "question": row["question"], "source_file": row["source_file"], "article_code": row["article_code"],
                "equivalents": [], "checked": [],
            })
            entry["checked"] += [list(key) for key in pending]
            entry["equivalents"] += [list(key) for key in accepted]
            for name, value in usage.items():
                if isinstance(value, int):
                    total[name] = total.get(name, 0) + value
            print(f"{count}/{len(jobs)} {row['question'][:50]}: {len(accepted)}/{len(pending)}", file=sys.stderr)
            # Ghi lai ca file sau moi cau: dung giua chung khong mat phan da hoi.
            OUTPUT.write_text(
                "".join(json.dumps(cached[r["question"]], ensure_ascii=False) + "\n" for r in rows if r["question"] in cached),
                encoding="utf-8",
            )

    with_equivalents = sum(1 for entry in cached.values() if entry["equivalents"])
    print(f"Da hoi {len(jobs)} cau, token: {total}. {with_equivalents}/{len(rows)} cau co Dieu tuong duong -> {OUTPUT}")


if __name__ == "__main__":
    main()
