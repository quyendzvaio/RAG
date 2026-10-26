"""Parse HTML phap dien thanh Document: moi Dieu mot Document, moi Phu luc dinh kem mot Document rieng.

`page_content` chi la noi dung Dieu (khong gom tieu de, ghi chu, chi dan); cac phan do nam trong metadata.
"""
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import unquote

from bs4 import BeautifulSoup, NavigableString, Tag
from langchain_core.documents import Document


logger = logging.getLogger(__name__)

LEGAL_DATE_RE = re.compile(r"có hiệu lực(?: thi hành)?(?: kể từ ngày| từ ngày)?\s+(\d{1,2}/\d{1,2}/\d{4})", re.I)
ARTICLE_RE = re.compile(r"^(Điều\s+[\w.\-]+)(?:\.\s*)?(.*)$", re.I)
# Dong tieu de phu luc dung rieng mot dong, vd "PHỤ LỤC", "Phụ lục V", "PHỤ LỤC II.1".
# Cau dan chieu ("... theo Phụ lục II ban hành kèm theo ...") khong khop.
APPENDIX_HEADING_RE = re.compile(r"^(PHỤ LỤC|Phụ lục)(\s+(số\s+)?[IVXLC\d]+(\.\d+)*[a-zA-Z]?)?\s*[.:]?$")
# Ma loai van ban nam trong ma Dieu, vd "Điều 39.4.NĐ.8." -> NĐ.
DOC_TYPE_CODE_RE = re.compile(r"\.(LQ|NĐ|TT|TL|QĐ|PL|NQ|NL|CT|SL)\.")
DOC_TYPE_BY_CODE = {
    "LQ": "Luật",
    "NĐ": "Nghị định",
    "TT": "Thông tư",
    "TL": "Thông tư liên tịch",
    "QĐ": "Quyết định",
    "PL": "Pháp lệnh",
    "NQ": "Nghị quyết",
    "NL": "Nghị quyết liên tịch",
    "CT": "Chỉ thị",
    "SL": "Sắc lệnh",
}
APPENDIX_TITLE_CHARS = 160


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\xa0", " ").splitlines()]
    return "\n".join(line for line in lines if line)


def extract_text_from_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()
    return clean_text(soup.get_text(separator="\n"))


def _tag_text(tag: Tag) -> str:
    _flatten_tables(tag)
    return clean_text(tag.get_text(separator="\n"))


def _flatten_tables(tag: Tag) -> None:
    """Thay moi <table> bang cac dong "o | o | o" (bang long nhau xu ly tu trong ra ngoai)."""
    for table in reversed(tag.find_all("table")):
        rows = []
        for row in table.find_all("tr"):
            cells = [" ".join(cell.get_text(" ").split()) for cell in row.find_all(["td", "th"], recursive=False)]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(NavigableString("\n" + "\n".join(rows) + "\n"))


def _class_names(tag: Tag) -> List[str]:
    classes = tag.get("class", [])
    if isinstance(classes, str):
        return [classes]
    return classes


def _is_class(tag: Tag, class_name: str) -> bool:
    return class_name in _class_names(tag)


def _extract_urls(tag: Optional[Tag]) -> List[str]:
    if tag is None:
        return []
    urls = []
    for link in tag.find_all("a", href=True):
        href = link["href"].strip()
        if href and href != "#":
            urls.append(unquote(href))
    return urls


def _article_parts(article_text: str) -> Tuple[str, str]:
    match = ARTICLE_RE.match(article_text.strip())
    if not match:
        return article_text.strip(), ""
    return match.group(1).strip(), match.group(2).strip()


def article_heading(metadata: dict) -> str:
    """Tieu de hien thi cua mot Dieu, vd "Điều 10.1.QĐ.12.1. Phạm vi điều chỉnh"."""
    return " ".join(
        value for value in (metadata.get("article_code", ""), metadata.get("article_title", "")) if value
    ).strip()


def doc_type_from_code(article_code: str) -> str:
    match = DOC_TYPE_CODE_RE.search(article_code)
    return DOC_TYPE_BY_CODE[match.group(1)] if match else ""


def _extract_effective_date(legal_note: str) -> str:
    match = LEGAL_DATE_RE.search(legal_note)
    return match.group(1) if match else ""


def _split_appendices(content: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Tach noi dung Dieu thanh (than Dieu, [(tieu de phu luc, noi dung phu luc), ...])."""
    lines = content.split("\n")
    starts = [index for index, line in enumerate(lines) if APPENDIX_HEADING_RE.match(line)]
    if not starts:
        return content, []
    body = "\n".join(lines[: starts[0]])
    appendices = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        section = lines[start:end]
        # Tieu de = dong "PHỤ LỤC ..." + dong ten dau tien (bo dong "(Kèm theo ...)").
        name = next((line for line in section[1:4] if not line.startswith("(")), "")
        title = f"{section[0]} {name}".strip()[:APPENDIX_TITLE_CHARS]
        appendices.append((title, "\n".join(section)))
    return body, appendices


def _make_article_documents(
    *,
    source_file: str,
    chapter: str,
    chapter_title: str,
    article_text: str,
    legal_note: str,
    content: str,
    urls: List[str],
    related_text: str,
    related_urls: List[str],
    first_index: int,
) -> List[Document]:
    """Mot Dieu -> Document cua Dieu + moi Phu luc dinh kem mot Document (cung ma Dieu)."""
    article_code, article_title = _article_parts(article_text)
    content = clean_text(content)
    if not article_code or not content:
        return []

    body, appendices = _split_appendices(content)
    base_metadata = {
        "source": source_file,
        "source_file": source_file,
        "chapter": chapter,
        "chapter_title": chapter_title,
        "article_code": article_code,
        "article_title": article_title,
        "legal_note": legal_note,
        "doc_type": doc_type_from_code(article_code),
        "effective_date": _extract_effective_date(legal_note),
        "urls": urls,
        "related_text": related_text,
        "related_urls": related_urls,
    }
    parts = [("article", article_title, body)] if body else []
    parts += [("appendix", f"{article_title} - {title}" if article_title else title, text) for title, text in appendices]

    documents = []
    for offset, (part, title, text) in enumerate(parts):
        article_index = first_index + offset
        metadata = {
            **base_metadata,
            "article_title": title,
            "part": part,
            "article_index": article_index,
            "parent_id": f"{Path(source_file).stem}:{article_index}",
        }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def parse_legal_html(html_content: str, source_file: str) -> List[Document]:
    """Moi Dieu (va moi Phu luc) mot Document; file khong co Dieu nao (stub "Đề mục ...") tra ve []."""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    container = soup.select_one("div._content") or soup.body or soup
    blocks = [
        tag
        for tag in container.find_all("p")
        if any(name in _class_names(tag) for name in ("pChuong", "pDieu", "pGhiChu", "pNoiDung", "pChiDan"))
    ]

    documents: List[Document] = []
    chapter = ""
    chapter_title = ""
    current_article = ""
    current_note = ""
    current_urls: List[str] = []
    current_content: List[str] = []
    current_related: List[str] = []
    current_related_urls: List[str] = []

    def flush_article() -> None:
        nonlocal current_article, current_note, current_urls, current_content
        nonlocal current_related, current_related_urls
        if not current_article:
            return
        documents.extend(
            _make_article_documents(
                source_file=source_file,
                chapter=chapter,
                chapter_title=chapter_title,
                article_text=current_article,
                legal_note=current_note,
                content="\n".join(current_content),
                urls=current_urls,
                related_text="\n".join(current_related),
                related_urls=current_related_urls,
                first_index=len(documents) + 1,
            )
        )
        current_article = ""
        current_note = ""
        current_urls = []
        current_content = []
        current_related = []
        current_related_urls = []

    last_chapter_marker = ""
    for block in blocks:
        text = _tag_text(block)
        if not text:
            continue

        if _is_class(block, "pChuong"):
            flush_article()
            if text.lower().startswith("chương"):
                chapter = text
                last_chapter_marker = text
            elif last_chapter_marker:
                chapter_title = text
            continue

        if _is_class(block, "pDieu"):
            flush_article()
            current_article = text
            continue

        if _is_class(block, "pGhiChu"):
            current_note = text
            current_urls = _extract_urls(block)
            continue

        if _is_class(block, "pNoiDung"):
            current_content.append(text)
            continue

        if _is_class(block, "pChiDan"):
            current_related.append(text)
            current_related_urls.extend(_extract_urls(block))

    flush_article()

    if not documents:
        preview = " ".join(extract_text_from_html(html_content).split())[:80]
        logger.info("Bo qua %s: khong co Dieu nao (%s)", source_file, preview)
    return documents


def load_html_documents(directory: str) -> List[Document]:
    documents: List[Document] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".html"):
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            documents.extend(parse_legal_html(file.read(), filename))
    return documents
