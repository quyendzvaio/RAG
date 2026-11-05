"""Phan tich ingestion + chunking parent-child + embedding input + wiki records tren du lieu that.

Can langchain/transformers nen chay trong image (xem `make analyze-ingestion`).
"""
import collections, json, re, statistics, sys
sys.path.insert(0, "/app")
from rag.chunking import split_parent_child
from rag.data_loader import APPENDIX_HEADING_RE, DOC_TYPE_BY_CODE, DOC_TYPE_CODE_RE, load_html_documents
from rag.query_engine import PARENT_CONTEXT_CHARS

def pct(a, b): return f"{a} ({100*a/max(b,1):.1f}%)"
def q(vals, p): s = sorted(vals); return s[min(int(p/100*len(s)), len(s)-1)]

docs = load_html_documents("data_raw")
N = len(docs)
print("== INGESTION ==")
parts = collections.Counter(d.metadata.get("part") for d in docs)
print("parent:", N, dict(parts))
print("parent khong co article_code (fallback):", sum("article_code" not in d.metadata for d in docs))
left = [d for d in docs if d.metadata.get("part") == "article" and any(APPENDIX_HEADING_RE.match(l) for l in d.page_content.split("\n"))]
print("Dieu van con dong tieu de PHU LUC trong noi dung:", len(left))
apx = [d for d in docs if d.metadata.get("part") == "appendix"]
print("phu luc: ky tu median", statistics.median(len(d.page_content) for d in apx) if apx else 0,
      "| max", max((len(d.page_content) for d in apx), default=0))
mism = collections.Counter()
for d in docs:
    m = DOC_TYPE_CODE_RE.search(d.metadata.get("article_code", ""))
    expect = DOC_TYPE_BY_CODE[m.group(1)] if m else ""
    if d.metadata.get("doc_type", "") != expect:
        mism[(m.group(1) if m else "(khong ma)", d.metadata.get("doc_type", "") or "(rong)")] += 1
print("doc_type", collections.Counter(d.metadata.get("doc_type") or "(rong)" for d in docs).most_common())
print("doc_type sai so voi ma loai:", sum(mism.values()), mism.most_common(5))
rel = [d for d in docs if d.metadata.get("related_text")]
print("Dieu co related_text:", pct(len(rel), N), "| con nam trong page_content:",
      sum(d.metadata["related_text"][:80] in d.page_content for d in rel))
heads = collections.Counter((d.metadata.get("source_file"), d.metadata.get("article_code"), d.metadata.get("article_title")) for d in docs)
print("tieu de trung nhau trong cung file:", sum(c - 1 for c in heads.values() if c > 1))
lens = [len(d.page_content) for d in docs]
print("ky tu/parent: median", statistics.median(lens), "p90", q(lens, 90), "p99", q(lens, 99),
      f"| > {PARENT_CONTEXT_CHARS} (context chi lay child lien quan):", pct(sum(l > PARENT_CONTEXT_CHARS for l in lens), N))

print("\n== CHUNKING (parent-child) + EMBEDDING INPUT ==")
try:
    from rag.embeddings import get_embedding_model, token_counter
    count, max_tokens = token_counter(get_embedding_model())  # tokenizer cua model ONNX da export, khong can torch
except Exception as exc:
    print("khong tai duoc tokenizer, uoc luong theo ky tu:", exc)
    def count(text): return len(text) // 3 + 1
    max_tokens = 256
children = split_parent_child(docs, count, max_tokens)
C = len(children)
ntok = [count(c.page_content) + 2 for c in children]
print("max_seq_length:", max_tokens, "| child:", C, "| token/child median", statistics.median(ntok), "p90", q(ntok, 90),
      "max", max(ntok), "| > max_seq_length (bi cat duoi):", pct(sum(n > max_tokens for n in ntok), C))
per_parent = collections.Counter(c.metadata["parent_id"] for c in children)
print("child/parent: 1 child:", pct(sum(v == 1 for v in per_parent.values()), len(per_parent)),
      "| median", statistics.median(per_parent.values()), "| max", max(per_parent.values()))
print("parent khong co child:", N - len(per_parent))
nohdr = sum(not c.page_content.startswith(c.metadata["article_code"]) for c in children)
print("child KHONG bat dau bang ma Dieu:", pct(nohdr, C))
short = sum(n < 40 for n in ntok)
print("child < 40 token:", pct(short, C))
bad_end = sum(1 for c in children if not re.search(r"[.;:!?)\]”\"…]\s*$", c.page_content.strip()))
print("child ket thuc giua cau:", pct(bad_end, C))
meta_bytes = [len(json.dumps(c.metadata, ensure_ascii=False).encode()) for c in children]
text_bytes = [len(c.page_content.encode()) for c in children]
print(f"metadata/child: median {statistics.median(meta_bytes)} B | tong metadata {sum(meta_bytes)/1e6:.1f} MB vs text {sum(text_bytes)/1e6:.1f} MB")

print("\n== WIKI RECORDS ==")
try:
    recs = [json.loads(l) for l in open("wiki_data/raw_sources/articles.jsonl", encoding="utf-8")]
    print("record:", len(recs), "| co full content:", pct(sum(bool(r.get("content")) for r in recs), len(recs)),
          "| content dai hon preview 500 ky tu:", pct(sum(len(r.get("content", "")) > 500 for r in recs), len(recs)))
except FileNotFoundError:
    print("chua co wiki_data/raw_sources/articles.jsonl (make prepare-wiki-sources)")
