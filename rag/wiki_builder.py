"""Lop wiki do LLM viet (mau "LLM Wiki" cua Karpathy) tren lop raw source bat bien.

Doc `raw_sources/articles.jsonl`, goi llm-service va ghi vao `wiki_data/wiki/`:
- `de-muc/<ma>.md`: tong quan mot de muc phap dien, lien ket toi van ban va chu de.
- `van-ban/<ma>.md`: tom tat mot van ban (Luat, Nghi dinh, Thong tu...) cua de muc.
- `chu-de/<slug>.md`: chu de tong hop nhieu Dieu tu nhieu van ban.
- `index.md` (muc luc), `log.md` (nhat ky ingest, chi ghi them), `SCHEMA.md` (quy uoc).

Moi y trong trang trich `[Điều <ma>]`. Lien ket, muc luc va danh sach Dieu do code sinh nen luon dung;
`lint_wiki` kiem tra trich dan va lien ket ma LLM tu viet.
"""
import argparse
import html
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rag.wiki_source_preparer import _frontmatter
from shared.config import get_settings
from shared.http import LLM_TIMEOUT
from shared.llm import complete
from shared.log import setup_logging
from shared.schemas import ChatMessage


logger = logging.getLogger(__name__)

LLM_WORKERS = 4
PAGE_MAX_TOKENS = 4000  # het ngan sach thi shared.llm goi lai mot lan voi gap doi
PLAN_MAX_TOKENS = 3000
TOPIC_CONTEXT_CHARS = 60000  # Dieu vuot ngan sach nay chi gui tieu de
ARTICLES_PER_TOPIC = 8  # so chu de ~ so Dieu / hang so nay, gioi han [MIN_TOPICS, MAX_TOPICS]
MIN_TOPICS = 3
MAX_TOPICS = 8

DE_MUC_TITLE_RE = re.compile(r"<h3[^>]*>(.*?)</h3>", re.S)
LEGAL_NOTE_RE = re.compile(
    r"(?P<title>(?:Bộ luật|Luật|Pháp lệnh|Nghị định|Nghị quyết liên tịch|Nghị quyết|Thông tư liên tịch|Thông tư|"
    r"Quyết định|Chỉ thị|Sắc lệnh)\b.*?)\s+ngày\s+(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+của\s+(?P<issuer>[^,()]+)"
)
CITATION_RE = re.compile(r"\[Điều\s+([^\]]+)\]")
CODE_RE = re.compile(r"\d+(?:\.\d+)*\.[A-ZĐ]+(?:\.\d+[a-z]?)*")
LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
SUMMARY_PREFIX = "TÓM TẮT:"

SYSTEM_PROMPT = (
    "Bạn là biên tập viên wiki pháp luật Việt Nam. Chỉ viết điều có trong các Điều được cung cấp, "
    "không dùng kiến thức ngoài. Mỗi ý phải kèm trích dẫn dạng [Điều <mã>], mỗi ngoặc vuông một mã, "
    "đúng mã đã cho. Giữ nguyên số liệu, mức thuế suất, thời hạn như văn bản. Viết cô đọng: diễn đạt lại ý chính, "
    "không chép lại nguyên văn từng khoản, không lặp cùng một ý cho nhiều văn bản mà gộp trích dẫn. "
    "Viết Markdown tiếng Việt có dấu, "
    "không viết tiêu đề cấp 1, không viết frontmatter, không bọc trong ```."
)


@dataclass
class Article:
    code: str  # "33.9.LQ.5" (khong co "Điều " va dau cham cuoi)
    heading: str
    content: str


@dataclass
class LegalDocument:
    code: str  # "33.9.LQ", "33.9.NĐ.1"
    title: str
    issuer: str
    issued_date: str
    effective_date: str
    articles: List[Article] = field(default_factory=list)

    @property
    def path(self) -> str:
        return f"van-ban/{slugify(self.code)}"


@dataclass
class DeMuc:
    code: str  # "33.9"
    title: str
    source_file: str
    source_hash: str
    documents: List[LegalDocument]

    @property
    def path(self) -> str:
        return f"de-muc/{self.code}"

    @property
    def articles(self) -> List[Article]:
        return [article for document in self.documents for article in document.articles]


@dataclass
class Topic:
    title: str
    summary: str
    article_codes: List[str]
    de_muc: str

    @property
    def path(self) -> str:
        return f"chu-de/{slugify(self.title)}"


@dataclass
class Page:
    path: str  # khong co ".md", dung luon lam dich [[lien ket]]
    title: str
    kind: str
    summary: str
    body: str
    meta: Dict = field(default_factory=dict)


def slugify(text: str) -> str:
    folded = unicodedata.normalize("NFD", text.replace("Đ", "D").replace("đ", "d"))
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9.]+", "-", ascii_text.lower()).strip("-.")


def article_code(raw_code: str) -> str:
    return raw_code.replace("Điều", "").strip().rstrip(".")


def document_code(code: str) -> str:
    return code.rsplit(".", 1)[0]


# ---------- doc raw source ----------


def load_de_muc(source_prefix: str) -> DeMuc:
    paths = get_settings().paths
    records = [
        record
        for record in _read_records(Path(paths.raw_sources_dir) / "articles.jsonl")
        if record["metadata"]["source_file"].startswith(source_prefix)
    ]
    if not records:
        raise ValueError(f"Khong co Dieu nao trong raw sources cho source '{source_prefix}'")
    source_file = records[0]["metadata"]["source_file"]
    if any(record["metadata"]["source_file"] != source_file for record in records):
        raise ValueError(f"Tien to '{source_prefix}' khop nhieu file, hay ghi dai hon")

    documents: Dict[str, LegalDocument] = {}
    notes: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        metadata = record["metadata"]
        code = article_code(metadata["article_code"])
        doc_code = document_code(code)
        document = documents.setdefault(
            doc_code, LegalDocument(doc_code, doc_code, "", "", metadata.get("effective_date", ""))
        )
        document.articles.append(Article(code, record["article_heading"], record["content"]))
        match = LEGAL_NOTE_RE.search(metadata.get("legal_note", ""))
        if match:
            notes[doc_code][(match["title"].strip(), match["issuer"].strip(), match["date"])] += 1
    for doc_code, document in documents.items():
        if notes[doc_code]:
            document.title, document.issuer, document.issued_date = notes[doc_code].most_common(1)[0][0]

    title = _de_muc_title(Path(paths.data_dir) / source_file)
    code_match = re.match(r"\d+\.\d+", next(iter(documents)))
    return DeMuc(
        code=code_match.group(0) if code_match else Path(source_file).stem,
        title=re.sub(r"^Đề mục\s+[\d.]+\s*", "", title) or source_file,
        source_file=source_file,
        source_hash=records[0]["source_hash"],
        documents=list(documents.values()),
    )


def _read_records(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def _de_muc_title(html_path: Path) -> str:
    if not html_path.exists():
        return ""
    match = DE_MUC_TITLE_RE.search(html_path.read_text(encoding="utf-8-sig")[:5000])
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", match.group(1))).split()) if match else ""


# ---------- goi LLM ----------


@dataclass
class LLMClient:
    usage: Counter = field(default_factory=Counter)
    calls: int = 0
    model: str = ""  # model thuc su tra loi (llm-service doc DEEPSEEK_MODEL, container nay thi khong)

    def complete(self, prompt: str, max_tokens: int) -> str:
        messages = [ChatMessage(role="system", content=SYSTEM_PROMPT), ChatMessage(role="user", content=prompt)]
        completion = complete(messages, max_tokens, temperature=0.2, timeout=LLM_TIMEOUT)
        if completion.fallback or not completion.content.strip():
            # Khong bao gio ghi text fallback cua llm-service vao wiki.
            raise RuntimeError(f"LLM khong tra loi duoc: {completion.error}")
        self.calls += 1
        self.model = completion.model
        self.usage.update(completion.token_usage)
        if completion.truncated:
            raise RuntimeError("LLM van bi cat sau khi goi lai voi max_tokens gap doi, khong ghi trang do dang")
        return _strip_fences(completion.content)


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```[\w-]*\n(.*?)\n?```", text, re.S)
    return match.group(1).strip() if match else text


def _split_summary(text: str) -> Tuple[str, str]:
    first, _, rest = text.partition("\n")
    if first.strip().upper().startswith(SUMMARY_PREFIX):
        return first.strip()[len(SUMMARY_PREFIX) :].strip(), rest.strip()
    return "", text


def _articles_text(articles: List[Article], budget: int = TOPIC_CONTEXT_CHARS) -> str:
    parts, used = [], 0
    for article in articles:
        if used + len(article.content) <= budget:
            parts.append(f"### [Điều {article.code}] {article.heading}\n{article.content}")
            used += len(article.content)
        else:
            parts.append(f"### [Điều {article.code}] {article.heading}\n(nội dung lược bỏ vì quá dài)")
    return "\n\n".join(parts)


def _documents_text(de_muc: DeMuc) -> str:
    return "\n".join(f"- {document.code}: {document.title}" for document in de_muc.documents)


# ---------- cac buoc ingest ----------


def plan_topics(de_muc: DeMuc, llm: LLMClient) -> List[Topic]:
    """LLM chia de muc thanh cac chu de, moi chu de gom Dieu tu nhieu van ban."""
    count = max(MIN_TOPICS, min(MAX_TOPICS, len(de_muc.articles) // ARTICLES_PER_TOPIC))
    outline = "\n\n".join(
        f"### {document.title} ({document.code})\n"
        + "\n".join(f"- {article.code}: {article.heading}" for article in document.articles)
        for document in de_muc.documents
    )
    prompt = (
        f"Đề mục {de_muc.code} {de_muc.title} gồm các văn bản và Điều sau (mã: tiêu đề):\n\n{outline}\n\n"
        f"Hãy chia đề mục thành khoảng {count} chủ đề để làm trang wiki. Mỗi chủ đề là một vấn đề người dân "
        "hoặc doanh nghiệp hay hỏi (ví dụ: đối tượng áp dụng, căn cứ tính, miễn giảm, thủ tục, quyền và nghĩa vụ), "
        "gom các Điều liên quan TỪ NHIỀU VĂN BẢN (luật và văn bản hướng dẫn) về cùng một chỗ. Mỗi Điều nên thuộc "
        "ít nhất một chủ đề; một Điều có thể thuộc nhiều chủ đề. Có thể bỏ các Điều về hiệu lực, trách nhiệm thi hành.\n"
        'Chỉ trả về một mảng JSON: [{"title": "...", "summary": "<một câu>", "articles": ["<mã Điều>", ...]}]'
    )
    raw = llm.complete(prompt, PLAN_MAX_TOKENS)
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        raise ValueError(f"LLM khong tra ve JSON chu de cho de muc {de_muc.code}: {raw[:200]}")
    known = {article.code for article in de_muc.articles}
    topics, seen = [], set()
    for item in json.loads(match.group(0)):
        codes = [code for code in (article_code(str(code)) for code in item.get("articles", [])) if code in known]
        dropped = len(item.get("articles", [])) - len(codes)
        if dropped:
            logger.warning("Chu de '%s': bo %d ma Dieu khong ton tai", item.get("title"), dropped)
        topic = Topic(str(item.get("title", "")).strip(), str(item.get("summary", "")).strip(), codes, de_muc.code)
        if topic.title and codes and topic.path not in seen:
            seen.add(topic.path)
            topics.append(topic)
    logger.info("De muc %s: %d chu de", de_muc.code, len(topics))
    return topics


def write_document_page(de_muc: DeMuc, document: LegalDocument, topics: List[Topic], llm: LLMClient) -> Page:
    prompt = (
        f'Dưới đây là toàn bộ các Điều của văn bản "{document.title}" (mã {document.code}) '
        f"trong đề mục {de_muc.code} {de_muc.title}. Các văn bản khác cùng đề mục:\n{_documents_text(de_muc)}\n\n"
        f"Dòng đầu tiên: \"{SUMMARY_PREFIX} <một câu dưới 30 từ>\". Sau đó viết đúng các mục:\n"
        "## Tổng quan: 2-4 câu, văn bản quy định gì, áp dụng cho ai, hướng dẫn văn bản nào trong đề mục (nếu có).\n"
        "## Nội dung chính: gạch đầu dòng theo nhóm vấn đề, mỗi ý kèm [Điều <mã>].\n"
        "## Lưu ý: ngoại lệ, điểm dễ nhầm, mốc thời gian (bỏ mục này nếu không có).\n\n"
        f"Các Điều:\n\n{_articles_text(document.articles)}"
    )
    summary, body = _split_summary(llm.complete(prompt, PAGE_MAX_TOKENS))
    codes = {article.code for article in document.articles}
    related = [topic for topic in topics if codes & set(topic.article_codes)]
    header = [f"Thuộc đề mục: [[{de_muc.path}|{de_muc.code} {de_muc.title}]]"]
    if document.issuer:
        header.append(f"Ban hành: {document.issuer}, ngày {document.issued_date}")
    if document.effective_date:
        header.append(f"Hiệu lực: {document.effective_date}")
    footer = ["## Danh sách Điều", ""] + [f"- {article.heading}" for article in document.articles]
    if related:
        footer += ["", "## Chủ đề liên quan", ""] + [f"- [[{topic.path}|{topic.title}]]" for topic in related]
    return Page(
        path=document.path,
        title=document.title,
        kind="van_ban",
        summary=summary,
        body="\n\n".join(["  \n".join(header), body, "\n".join(footer)]),
        meta={"de_muc": de_muc.code, "document_code": document.code, "sources": sorted(codes)},
    )


def write_topic_page(de_muc: DeMuc, topic: Topic, catalog: List[Topic], llm: LLMClient) -> Page:
    by_code = {article.code: article for article in de_muc.articles}
    articles = [by_code[code] for code in topic.article_codes]
    others = "\n".join(f"- [[{other.path}|{other.title}]]" for other in catalog if other.path != topic.path)
    prompt = (
        f'Viết trang wiki chủ đề "{topic.title}" thuộc đề mục {de_muc.code} {de_muc.title}. '
        f"Phạm vi: {topic.summary}\n\n"
        f"Dòng đầu tiên: \"{SUMMARY_PREFIX} <một câu dưới 30 từ>\". Sau đó:\n"
        "- 2-3 câu mở đầu trả lời thẳng vào chủ đề.\n"
        "- Các mục ## theo vấn đề con. Tổng hợp theo vấn đề, không tóm tắt lần lượt từng Điều: nêu quy định "
        "ở luật trước, rồi chi tiết ở văn bản hướng dẫn, nói rõ văn bản nào hướng dẫn văn bản nào.\n"
        "- Mỗi ý kèm [Điều <mã>].\n"
        "- Nếu các văn bản quy định khác nhau hoặc chồng chéo về cùng vấn đề, nêu ở mục ## Điểm cần lưu ý.\n"
        "- Chỉ khi thật liên quan, liên kết sang trang chủ đề khác bằng đúng cú pháp trong danh sách sau:\n"
        f"{others or '(không có)'}\n\n"
        f"Các văn bản:\n{_documents_text(de_muc)}\n\nCác Điều:\n\n{_articles_text(articles)}"
    )
    summary, body = _split_summary(llm.complete(prompt, PAGE_MAX_TOKENS))
    header = f"Thuộc đề mục: [[{de_muc.path}|{de_muc.code} {de_muc.title}]]"
    footer = ["## Các Điều nguồn", ""] + [f"- {article.heading}" for article in articles]
    return Page(
        path=topic.path,
        title=topic.title,
        kind="chu_de",
        summary=summary or topic.summary,
        body="\n\n".join([header, body, "\n".join(footer)]),
        meta={"de_muc": de_muc.code, "sources": topic.article_codes},
    )


def write_de_muc_page(de_muc: DeMuc, documents: List[Page], topics: List[Page], llm: LLMClient) -> Page:
    listing = "\n".join(f"- {page.title}: {page.summary}" for page in documents)
    topic_listing = "\n".join(f"- {page.title}: {page.summary}" for page in topics)
    prompt = (
        f"Viết phần tổng quan cho trang wiki đề mục {de_muc.code} {de_muc.title} của Bộ pháp điển, "
        f"dựa trên tóm tắt các văn bản:\n{listing}\n\nvà các chủ đề:\n{topic_listing}\n\n"
        f"Dòng đầu tiên: \"{SUMMARY_PREFIX} <một câu dưới 30 từ>\". Sau đó viết:\n"
        "## Tổng quan: một đoạn, đề mục điều chỉnh vấn đề gì, cho ai.\n"
        "## Hệ thống văn bản: văn bản gốc là gì, văn bản nào hướng dẫn phần nào.\n"
        "Không cần trích dẫn Điều trong trang này. Không liệt kê lại danh sách chủ đề."
    )
    summary, body = _split_summary(llm.complete(prompt, PAGE_MAX_TOKENS))
    footer = (
        ["## Văn bản", ""]
        + [f"- [[{page.path}|{page.title}]]: {page.summary}" for page in documents]
        + ["", "## Chủ đề", ""]
        + [f"- [[{page.path}|{page.title}]]: {page.summary}" for page in topics]
    )
    return Page(
        path=de_muc.path,
        title=f"Đề mục {de_muc.code} {de_muc.title}",
        kind="de_muc",
        summary=summary,
        body="\n\n".join([f"Nguồn: `{de_muc.source_file}`", body, "\n".join(footer)]),
        meta={"de_muc": de_muc.code, "source_file": de_muc.source_file, "source_hash": de_muc.source_hash},
    )


def ingest_de_muc(de_muc: DeMuc, catalog: List[Topic], topics: List[Topic], llm: LLMClient) -> List[Page]:
    with ThreadPoolExecutor(LLM_WORKERS) as pool:
        document_jobs = [pool.submit(write_document_page, de_muc, document, topics, llm) for document in de_muc.documents]
        topic_jobs = [pool.submit(write_topic_page, de_muc, topic, catalog, llm) for topic in topics]
        documents = [job.result() for job in document_jobs]
        topic_pages = [job.result() for job in topic_jobs]
    return [write_de_muc_page(de_muc, documents, topic_pages, llm)] + documents + topic_pages


# ---------- ghi file, muc luc, lint ----------


def save_page(wiki_dir: Path, page: Page, model: str) -> None:
    path = wiki_dir / f"{page.path}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "type": page.kind,
        "title": page.title,
        "summary": page.summary,
        **page.meta,
        "generated_by": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(f"{_frontmatter(meta)}\n\n# {page.title}\n\n{page.body.strip()}\n", encoding="utf-8")


def read_pages(wiki_dir: Path) -> List[Tuple[str, dict, str]]:
    """(duong dan khong .md, frontmatter, than trang) cua moi trang noi dung."""
    pages = []
    for path in sorted(wiki_dir.glob("*/*.md")):
        text = path.read_text(encoding="utf-8")
        _, front, body = text.split("---", 2)
        meta = {}
        for line in front.strip().splitlines():
            key, _, value = line.partition(": ")
            meta[key] = json.loads(value)
        pages.append((str(path.relative_to(wiki_dir).with_suffix("")), meta, body))
    return pages


def write_index(wiki_dir: Path) -> None:
    groups = {"de_muc": "Đề mục", "chu_de": "Chủ đề", "van_ban": "Văn bản"}
    pages = read_pages(wiki_dir)
    lines = ["# Mục lục wiki", "", f"{len(pages)} trang. Đọc mục lục này trước, rồi mở trang liên quan.", ""]
    for kind, label in groups.items():
        lines += [f"## {label}", ""]
        lines += [
            f"- [[{path}|{meta['title']}]] ({meta['de_muc']}): {meta['summary']}"
            for path, meta, _ in pages
            if meta["type"] == kind
        ]
        lines.append("")
    (wiki_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def lint_wiki(wiki_dir: Path, known_codes: Iterable[str]) -> Dict[str, list]:
    """Trich dan toi Dieu khong ton tai, lien ket gay, trang khong ai tro toi, trang khong trich dan."""
    known = set(known_codes)
    pages = read_pages(wiki_dir)
    paths = {path for path, _, _ in pages}
    inbound: Counter = Counter()
    issues: Dict[str, list] = {"bad_citations": [], "broken_links": [], "orphans": [], "uncited": []}
    for path, meta, body in pages:
        codes = [code for group in CITATION_RE.findall(body) for code in CODE_RE.findall(group)]
        issues["bad_citations"] += [(path, code) for code in codes if code not in known]
        if meta["type"] != "de_muc" and not codes:
            issues["uncited"].append(path)
        for target in LINK_RE.findall(body):
            if target.strip() in paths:
                inbound[target.strip()] += 1 if target.strip() != path else 0
            else:
                issues["broken_links"].append((path, target))
    issues["orphans"] = [path for path in paths if not inbound[path] and not path.startswith("de-muc/")]
    return issues


def append_log(wiki_dir: Path, lines: List[str]) -> None:
    with (wiki_dir / "log.md").open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n\n")


def write_schema(wiki_dir: Path) -> None:
    (wiki_dir / "SCHEMA.md").write_text(WIKI_SCHEMA, encoding="utf-8")


WIKI_SCHEMA = """# Wiki Schema

Lớp wiki do LLM viết từ `wiki_data/raw_sources` (bất biến). Sinh lại bằng `build_wiki.py`; không sửa tay.

## Trang

- `de-muc/<mã>.md`: tổng quan một đề mục pháp điển, liên kết tới mọi văn bản và chủ đề của đề mục.
- `van-ban/<mã>.md`: một văn bản (Luật, Nghị định, Thông tư...) của đề mục: tổng quan, nội dung chính, danh sách Điều.
- `chu-de/<slug>.md`: một vấn đề tổng hợp từ nhiều Điều của nhiều văn bản.
- `index.md`: mục lục, mỗi trang một dòng tóm tắt. `log.md`: nhật ký ingest, chỉ ghi thêm.

## Quy ước

- Frontmatter: `type`, `title`, `summary`, `de_muc`, `sources` (mã Điều nguồn), `generated_by`, `generated_at`.
- Mỗi ý trích `[Điều <mã>]` theo mã pháp điển (vd `[Điều 33.9.LQ.5]`). Khi trả lời, trích dẫn cuối cùng
  phải lấy từ Điều gốc trong raw sources, không chỉ từ trang wiki.
- Liên kết nội bộ dạng `[[chu-de/<slug>|Tên]]`. Danh sách Điều, liên kết văn bản/chủ đề và mục lục do code sinh.
- `lint_wiki` báo trích dẫn tới Điều không tồn tại, liên kết gãy, trang mồ côi, trang không trích dẫn.
"""


def build_wiki(source_prefixes: List[str], wiki_dir: Path) -> Dict[str, object]:
    de_mucs = [load_de_muc(prefix) for prefix in source_prefixes]
    llm = LLMClient()
    wiki_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(LLM_WORKERS) as pool:
        planned = list(pool.map(lambda de_muc: plan_topics(de_muc, llm), de_mucs))
    # Danh muc chu de cua moi de muc trong lan chay de trang chu de co the lien ket cheo.
    catalog = [topic for topics in planned for topic in topics]

    model = llm.model
    log_lines = [f"## [{datetime.now(timezone.utc).isoformat(timespec='seconds')}] ingest ({model})", ""]
    for de_muc, topics in zip(de_mucs, planned):
        logger.info("Dang viet wiki cho de muc %s %s (%d van ban, %d chu de)",
                    de_muc.code, de_muc.title, len(de_muc.documents), len(topics))
        pages = ingest_de_muc(de_muc, catalog, topics, llm)
        for page in pages:
            save_page(wiki_dir, page, model)
        log_lines.append(
            f"- Đề mục {de_muc.code} {de_muc.title}: {len(de_muc.articles)} Điều, "
            f"{len(de_muc.documents)} văn bản, {len(topics)} chủ đề, {len(pages)} trang"
        )

    write_schema(wiki_dir)
    write_index(wiki_dir)
    known = {
        article_code(record["metadata"]["article_code"])
        for record in _read_records(Path(get_settings().paths.raw_sources_dir) / "articles.jsonl")
    }
    issues = lint_wiki(wiki_dir, known)
    log_lines.append(f"- LLM: {llm.calls} lần gọi, token {dict(llm.usage)}")
    log_lines.append("- Lint: " + ", ".join(f"{key} {len(value)}" for key, value in issues.items()))
    append_log(wiki_dir, log_lines)
    return {"llm_calls": llm.calls, "token_usage": dict(llm.usage), "lint": issues}


def main(argv: Optional[Iterable[str]] = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Viet lop wiki bang LLM cho cac de muc da chon.")
    parser.add_argument("sources", nargs="+", help="Tien to ten file HTML cua de muc, vd aec013b7")
    parser.add_argument("--output", default=get_settings().paths.wiki_dir, help="Thu muc wiki")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = build_wiki(args.sources, Path(args.output))
    print(f"LLM: {result['llm_calls']} lan goi, token {result['token_usage']}")
    for key, values in result["lint"].items():
        print(f"- {key}: {len(values)}")
        for value in values[:10]:
            print(f"    {value}")


if __name__ == "__main__":
    main()
