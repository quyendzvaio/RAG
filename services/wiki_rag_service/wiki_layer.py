"""Truy xuat cua wiki-rag-service: LLM doc index.md cua lop wiki (build_wiki.py) va chon trang can doc.

Context chi gom cac trang wiki, khong co BM25 hay Dieu goc. Cau hoi ngoai cac de muc da co wiki
thi LLM khong chon trang nao va service tra loi la khong tim thay context.
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from rag.wiki_builder import CITATION_RE, CODE_RE, read_pages
from shared.llm import complete
from shared.schemas import ChatMessage, ContextItem


logger = logging.getLogger(__name__)

MAX_PAGES = 3  # so trang toi da moi cau hoi (con bi gioi han boi top_k cua request)
SELECT_MAX_TOKENS = 3000  # tran, khong phai muc dung: model co suy luan (glm) tieu 300-1300 token truoc khi tra loi
PAGE_PATH_RE = re.compile(r"(?:de-muc|van-ban|chu-de)/[\w.\-]+")
DE_MUC_ANSWER_RE = re.compile(r"ĐỀ MỤC:\s*(KHÔNG|\d+\.\d+)", re.I)
# Danh sach Dieu/lien ket cuoi trang do code sinh: bo khoi context, trich dan da nam trong than trang.
FOOTER_RE = re.compile(r"\n## (?:Các Điều nguồn|Danh sách Điều)\n.*", re.S)


@dataclass
class WikiPage:
    path: str
    meta: dict
    body: str
    source_file: str  # file HTML cua de muc, de eval doi chieu voi Dieu goc

    @property
    def cited_codes(self) -> List[str]:
        """Ma Dieu trich trong trang, theo thu tu xuat hien, khong trung."""
        codes = [code for group in CITATION_RE.findall(self.body) for code in CODE_RE.findall(group)]
        return list(dict.fromkeys(codes))

    def context(self) -> ContextItem:
        return ContextItem(
            id=self.path,
            source=f"wiki/{self.path}.md",
            title=self.meta.get("title", self.path),
            content=self.body,
            metadata={
                "type": self.meta.get("type"),
                "de_muc": self.meta.get("de_muc"),
                "source_file": self.source_file,
                # Cung dang `article_code` cua Dieu goc ("Điều 33.9.LQ.9.") de eval biet trang phu Dieu nao.
                "article_codes": [f"Điều {code}." for code in self.cited_codes],
            },
            citation=f"wiki/{self.path}",
        )


class WikiLayer:
    def __init__(self, wiki_dir: Path, index_text: str, version: float):
        self.wiki_dir = wiki_dir
        self.index_text = index_text
        self.version = version  # mtime cua index.md luc nap
        self.pages: Dict[str, WikiPage] = {}
        self.de_muc_titles: Dict[str, str] = {}
        if not index_text:
            return
        loaded = [(path, meta, _strip_page(body)) for path, meta, body in read_pages(wiki_dir)]
        de_mucs = [meta for _, meta, _ in loaded if meta.get("type") == "de_muc"]
        source_files = {meta.get("de_muc"): meta.get("source_file", "") for meta in de_mucs}
        self.de_muc_titles = {meta.get("de_muc"): meta.get("title", "") for meta in de_mucs}
        for path, meta, body in loaded:
            self.pages[path] = WikiPage(path, meta, body, source_files.get(meta.get("de_muc"), ""))

    @property
    def ready(self) -> bool:
        return bool(self.pages)

    def status(self) -> Dict:
        return {"wiki_dir": str(self.wiki_dir), "page_count": len(self.pages), "index_mtime": self.version}

    def select(self, question: str, max_pages: int) -> Tuple[List[WikiPage], Dict[str, int]]:
        """(trang LLM chon tu index.md, token da dung). Rong khi cau hoi ngoai cac de muc co wiki.

        LLM phai xac dinh de muc truoc roi moi chon trang, va chi trang thuoc de muc do duoc nhan: neu chi hoi
        LLM chon trang, no van ghep cau hoi ngoai pham vi vao trang co tu ngu giong (do tren golden set).
        """
        if not self.ready or max_pages <= 0:
            return [], {}
        de_mucs = "\n".join(f"- {code}: {title}" for code, title in sorted(self.de_muc_titles.items()))
        prompt = (
            f"Wiki pháp luật này CHỈ gồm các đề mục:\n{de_mucs}\n\nMục lục wiki:\n\n{self.index_text}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Bước 1: câu hỏi có thực sự thuộc phạm vi điều chỉnh của một đề mục trên không (cùng lĩnh vực, "
            "không phải chỉ trùng vài từ)? Dòng đầu tiên trả lời đúng dạng \"ĐỀ MỤC: <mã>\" hoặc \"ĐỀ MỤC: KHÔNG\".\n"
            f"Bước 2: nếu có đề mục, các dòng sau là tối đa {max_pages} trang của đề mục đó cần đọc để trả lời, "
            "ưu tiên trang chủ đề, mỗi dòng một đường dẫn (ví dụ chu-de/abc). Không giải thích."
        )
        completion = complete([ChatMessage(role="user", content=prompt)], SELECT_MAX_TOKENS, temperature=0.0)
        if completion.fallback:
            raise RuntimeError(f"Khong chon duoc trang wiki, LLM loi: {completion.error}")
        match = DE_MUC_ANSWER_RE.search(completion.content)
        de_muc = match.group(1).upper() if match else None
        if de_muc == "KHÔNG" or de_muc not in self.de_muc_titles:
            return [], completion.token_usage
        paths = [
            path
            for path in dict.fromkeys(PAGE_PATH_RE.findall(completion.content))
            if path in self.pages and self.pages[path].meta.get("de_muc") == de_muc
        ]
        return [self.pages[path] for path in paths[:max_pages]], completion.token_usage


class LatestWikiLayer:
    """Luon dung ban wiki moi nhat: nap lai khi index.md doi (build_wiki.py ghi index.md sau cac trang)."""

    def __init__(self, wiki_dir: str):
        self.wiki_dir = Path(wiki_dir)
        self._layer: Optional[WikiLayer] = None
        self._lock = Lock()

    def get(self) -> WikiLayer:
        index_path = self.wiki_dir / "index.md"
        version = index_path.stat().st_mtime if index_path.exists() else 0.0
        if self._layer is None or self._layer.version != version:
            with self._lock:
                if self._layer is None or self._layer.version != version:
                    text = index_path.read_text(encoding="utf-8") if version else ""
                    self._layer = WikiLayer(self.wiki_dir, text, version)
                    logger.info("Da nap lop wiki %s: %d trang", self.wiki_dir, len(self._layer.pages))
        return self._layer


def _strip_page(body: str) -> str:
    """Bo dong '# Tieu de' (da nam trong ContextItem.title) va phan danh sach cuoi trang."""
    body = FOOTER_RE.sub("", body.strip())
    lines = body.split("\n", 1)
    return lines[1].strip() if lines[0].startswith("# ") and len(lines) > 1 else body
