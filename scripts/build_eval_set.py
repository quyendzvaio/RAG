"""Xay tap kiem thu doc lap: cau hoi va cau tra loi chuan dung truc tiep tu van ban nguyen ban.

Muc dich: co mot bo du lieu rata thu cong duoc de danh gia truy xuat va chat luong tra loi,
khong phu thuoc vao tap golden sinh boi LLM (khi LLM sinh cau hoi va ca reference thi sai lech
cua LLM xuyen suot ca hai phan). Moi mau gom:
  question         : cau hoi do nguoi viet (o day sinh tu tieu de Dieu; co the sua thu cong sau)
  article_code     : ma Dieu dap an
  source_file      : file HTML goc
  article_heading  : tieu de day du cua Dieu
  answer           : cau tra loi chuan rata thu cong (mac dinh: noi dung Dieu)
  question_unaccented: cau hoi khong dau (de do truong hop nguoi dung go khong dau)

Chay: python3 scripts/build_eval_set.py --limit 30
Xong co the rata thu cong file eval/eval_standalone.jsonl (sua cau hoi, rut gon answer).
"""
import argparse
import json
import random
import re
from pathlib import Path

OUT = Path("eval/eval_standalone.jsonl")
ARTICLES = Path("wiki_data/raw_sources/articles.jsonl")

# Dieu qua dai thi chi lay phan dau lam answer: cau hoi van tra loi duoc tu phan dau.
ANSWER_MAX_CHARS = 1500
# Bo qua Dieu co noi dung qua ngan (chi danh muc, khong co y phap ly) va phu luc.
MIN_CONTENT_CHARS = 120
SKIP_CODE_RE = re.compile(r"part.*appendix|PHỤ LỤC", re.I)


def load_articles() -> list[dict]:
    rows = []
    with ARTICLES.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                meta = row.get("metadata", {})
                if meta.get("part") == "appendix":
                    continue
                content = row.get("content", "") or row.get("content_preview", "")
                if len(content) < MIN_CONTENT_CHARS:
                    continue
                rows.append(row)
    return rows


def question_from_heading(heading: str, title: str) -> str:
    """Cau hoi mo dau tu tieu de; danh de nguoi rata sua cau hoi tu sau."""
    return f"{heading} quy định những gì?" if not title else f"{title} được quy định thế nào?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=30, help="So mau sinh ra (0 = tat ca)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = load_articles()
    random.shuffle(rows)

    samples = []
    seen_codes = set()
    for row in rows:
        if args.limit and len(samples) >= args.limit:
            break
        meta = row.get("metadata", {})
        code = meta.get("article_code", "")
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)
        content = (row.get("content", "") or row.get("content_preview", "")).strip()
        heading = row.get("article_heading", "") or meta.get("article_heading", "")
        title = meta.get("article_title", "")
        answer = content if len(content) <= ANSWER_MAX_CHARS else content[:ANSWER_MAX_CHARS].rsplit(" ", 1)[0] + "…"
        samples.append(
            {
                "question": question_from_heading(heading, title),
                "question_unaccented": "",
                "article_code": code,
                "source_file": meta.get("source_file", ""),
                "article_heading": heading,
                "answer": answer,
                "kind": "manual_draft",
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in samples:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Da sinh {len(samples)} mau vao {OUT}. Rata thu cong truoc khi dung de danh gia.")


if __name__ == "__main__":
    main()
