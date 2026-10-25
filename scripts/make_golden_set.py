"""Sinh bo cau hoi danh gia retrieval tu wiki raw sources.

Mac dinh dung DeepSeek dat cau hoi tu nhien (golden set do chat luong). Voi --offline,
cau hoi = tieu de Dieu + mot doan than Dieu, khong goi LLM: chi dung lam bo hoi quy
(so snapshot top-k khi refactor), khong dung lam chi so chat luong vi thien vi BM25.

Chay tren host sau `make prepare-wiki-sources`. Chi dung thu vien chuan.
Moi dong output: {question, question_unaccented, source_file, article_code, article_heading, kind}.
"""
import argparse
import json
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPT = """Bạn là người dân đang tìm hiểu pháp luật Việt Nam. Dưới đây là một Điều luật.
Hãy viết MỘT câu hỏi tự nhiên, ngắn (dưới 25 từ) mà Điều luật này trả lời được.
Yêu cầu:
- Hỏi về nội dung cụ thể trong thân Điều, không chép lại nguyên tiêu đề Điều.
- Không nhắc số hiệu Điều, số văn bản hay mã pháp điển.
- Câu hỏi phải tự hiểu được khi đứng một mình: không viết "Thông tư này", "Nghị định này",
  "Luật này", "văn bản này"; thay vào đó nêu rõ lĩnh vực hoặc đối tượng cụ thể.
- Chỉ trả về câu hỏi, không giải thích.

{heading}
{content}"""

MAX_ARTICLE_CHARS = 20000  # bo cac Dieu bi parser gop nham phu luc


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def load_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    return env


def read_sections(markdown: str) -> dict:
    """Map heading -> noi dung, chi giu heading xuat hien dung mot lan."""
    sections = defaultdict(list)
    parts = re.split(r"^## ", markdown, flags=re.M)
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        sections[heading.strip()].append(body)
    return {heading: bodies[0] for heading, bodies in sections.items() if len(bodies) == 1}


def article_content(section: str) -> str:
    match = re.search(r"^### Content\n\n(.*?)(?=^### |\Z)", section, flags=re.M | re.S)
    return match.group(1).strip() if match else ""


def ask_deepseek(env: dict, prompt: str) -> str:
    base_url = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    url = f"{base_url}/chat/completions" if base_url.endswith("/v1") else f"{base_url}/v1/chat/completions"
    payload = {
        "model": env.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 120,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {env['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"},
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.load(response)
            return data["choices"][0]["message"]["content"].strip().strip('"')
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503) or attempt == 5:
                raise
            time.sleep(2 ** attempt * 3)  # 3, 6, 12, 24, 48 giay


def offline_question(record: dict, content: str) -> str:
    words = content.split()
    fragment = " ".join(words[10:22]) if len(words) > 22 else " ".join(words[:12])
    title = record["metadata"].get("article_title", "")
    return f"{title} {fragment}".strip()


def sample_articles(raw_dir: Path, size: int, seed: int) -> list:
    records_by_source = defaultdict(list)
    with (raw_dir / "articles.jsonl").open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record["metadata"].get("article_code"):
                records_by_source[record["source_markdown"]].append(record)

    candidates = []
    for source_markdown, records in records_by_source.items():
        sections = read_sections((raw_dir / source_markdown).read_text(encoding="utf-8"))
        for record in records:
            section = sections.get(record["article_heading"])
            content = article_content(section) if section else ""
            if 200 <= len(content) <= MAX_ARTICLE_CHARS:
                candidates.append((record, content))

    # Lay deu theo source_file de khong bi cac van ban lon chiem het mau.
    rng = random.Random(seed)
    by_source = defaultdict(list)
    for record, content in candidates:
        by_source[record["metadata"]["source_file"]].append((record, content))
    for items in by_source.values():
        rng.shuffle(items)
    sources = sorted(by_source)
    rng.shuffle(sources)
    picked = []
    while len(picked) < size and any(by_source.values()):
        for source in sources:
            if by_source[source] and len(picked) < size:
                picked.append(by_source[source].pop())
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-sources", default="wiki_data/raw_sources")
    parser.add_argument("--output", default="eval/golden.jsonl")
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--offline", action="store_true", help="Khong goi LLM, chi sinh bo hoi quy")
    args = parser.parse_args()

    env = load_env(Path(".env"))
    if not args.offline and not env.get("DEEPSEEK_API_KEY"):
        sys.exit("Thieu DEEPSEEK_API_KEY trong .env")

    picked = sample_articles(Path(args.raw_sources), args.size, args.seed)
    print(f"Da chon {len(picked)} Dieu, dang sinh cau hoi...")

    def build(item):
        record, content = item
        if args.offline:
            question = offline_question(record, content)
        else:
            question = ask_deepseek(env, PROMPT.format(heading=record["article_heading"], content=content[:4000]))
        return {
            "question": question,
            "question_unaccented": strip_accents(question).lower(),
            "source_file": record["metadata"]["source_file"],
            "article_code": record["metadata"]["article_code"],
            "article_heading": record["article_heading"],
            "kind": "offline_snippet" if args.offline else "llm",
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(build, picked))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Da ghi {len(rows)} cau hoi vao {output}")


if __name__ == "__main__":
    main()
