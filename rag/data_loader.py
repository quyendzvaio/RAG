import os
from bs4 import BeautifulSoup
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


def extract_text_from_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    clean_text = "\n".join(line for line in lines if line)
    return clean_text

def load_html_documents(directory: str) -> List[Document]:
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                html = file.read()
                text = extract_text_from_html(html)
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents


def split_by_dieu(text: str) -> List[str]:
    parts = re.split(r"(Điều\s+\d+[.:]?[^\n]*)", text)
    chunks = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""
        full = f"{title}\n{content}"
        chunks.append(full)
    return chunks

# Step 2: Nếu Điều dài quá thì chunk nhỏ tiếp
def split_documents(documents: List[Document], chunk_size=512, chunk_overlap=64) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunked_docs = []
    for doc in documents:
        dieu_chunks = split_by_dieu(doc.page_content)
        for dieu in dieu_chunks:
            if len(dieu) <= chunk_size:
                chunked_docs.append(Document(page_content=dieu, metadata=doc.metadata))
            else:
                # Chunk thêm nếu điều quá dài
                small_chunks = splitter.split_text(dieu)
                for chunk in small_chunks:
                    chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return chunked_docs

