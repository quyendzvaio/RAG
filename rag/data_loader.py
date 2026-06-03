import os
import re
from typing import List, Optional, Tuple
from urllib.parse import unquote

from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


LEGAL_DATE_RE = re.compile(r"có hiệu lực(?: thi hành)?(?: kể từ ngày| từ ngày)?\s+(\d{1,2}/\d{1,2}/\d{4})", re.I)
ARTICLE_RE = re.compile(r"^(Điều\s+[\w.\-]+)(?:\.\s*)?(.*)$", re.I)


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\xa0", " ").splitlines()]
    return "\n".join(line for line in lines if line)


def extract_text_from_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()
    return clean_text(soup.get_text(separator="\n"))


def _tag_text(tag: Tag) -> str:
    return clean_text(tag.get_text(separator="\n"))


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


def _infer_doc_type(article_code: str, legal_note: str) -> str:
    text = f"{article_code} {legal_note}".lower()
    if ".lq." in text or " luật " in f" {text} ":
        return "Luật"
    if ".nđ." in text or "nghị định" in text:
        return "Nghị định"
    if ".tt." in text or "thông tư" in text:
        return "Thông tư"
    if ".qđ." in text or "quyết định" in text:
        return "Quyết định"
    return ""


def _extract_effective_date(legal_note: str) -> str:
    match = LEGAL_DATE_RE.search(legal_note)
    return match.group(1) if match else ""


def _make_article_document(
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
    article_index: int,
) -> Optional[Document]:
    article_code, article_title = _article_parts(article_text)
    content = clean_text(content)
    if not article_code or not content:
        return None

    doc_type = _infer_doc_type(article_code, legal_note)
    effective_date = _extract_effective_date(legal_note)
    page_parts = [
        chapter,
        chapter_title,
        article_text,
        legal_note,
        content,
        related_text,
    ]
    page_content = clean_text("\n".join(part for part in page_parts if part))

    return Document(
        page_content=page_content,
        metadata={
            "source": source_file,
            "source_file": source_file,
            "chapter": chapter,
            "chapter_title": chapter_title,
            "article_code": article_code,
            "article_title": article_title,
            "legal_note": legal_note,
            "doc_type": doc_type,
            "effective_date": effective_date,
            "urls": urls,
            "related_text": related_text,
            "related_urls": related_urls,
            "article_index": article_index,
        },
    )


def parse_legal_html(html_content: str, source_file: str) -> List[Document]:
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
    article_index = 0

    def flush_article() -> None:
        nonlocal current_article, current_note, current_urls, current_content
        nonlocal current_related, current_related_urls, article_index
        if not current_article:
            return
        article_index += 1
        document = _make_article_document(
            source_file=source_file,
            chapter=chapter,
            chapter_title=chapter_title,
            article_text=current_article,
            legal_note=current_note,
            content="\n".join(current_content),
            urls=current_urls,
            related_text="\n".join(current_related),
            related_urls=current_related_urls,
            article_index=article_index,
        )
        if document:
            documents.append(document)
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

    if documents:
        return documents

    plain_text = extract_text_from_html(html_content)
    if not plain_text:
        return []
    return [Document(page_content=plain_text, metadata={"source": source_file, "source_file": source_file})]


def load_html_documents(directory: str) -> List[Document]:
    documents: List[Document] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".html"):
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            documents.extend(parse_legal_html(file.read(), filename))
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks
