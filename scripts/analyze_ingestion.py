"""Phan tich ingestion + chunking + embedding input + wiki lookup tren du lieu that.

Can langchain/transformers nen chay trong image (xem `make analyze-ingestion`).
"""
import collections, json, re, statistics, sys
sys.path.insert(0, "/app")
from rag.data_loader import load_html_documents, split_documents

def pct(a, b): return f"{a} ({100*a/max(b,1):.1f}%)"
def q(vals, p): s = sorted(vals); return s[min(int(p/100*len(s)), len(s)-1)]

docs = load_html_documents("data_raw")
N = len(docs)
print("== INGESTION ==")
print("Dieu:", N)
fallback = [d for d in docs if "article_code" not in d.metadata]
print("document fallback (file khong co pDieu):", len(fallback), "| tong ky tu:", sum(len(d.page_content) for d in fallback))

app = [d for d in docs if re.search(r"^PHỤ LỤC\b|\nPHỤ LỤC\b|\nPhụ lục [IVX\d]+\n", d.page_content)]
print("Dieu chua PHU LUC dinh kem:", len(app), "| tong ky tu:", sum(len(d.page_content) for d in app),
      f"({100*sum(len(d.page_content) for d in app)/sum(len(d.page_content) for d in docs):.1f}% toan bo du lieu)")

# doc_type: so voi ma loai trong article_code (vd 39.4.NĐ.8 -> NĐ)
code_map = {"LQ": "Luật", "NĐ": "Nghị định", "TT": "Thông tư", "QĐ": "Quyết định"}
mism = collections.Counter(); seen = collections.Counter()
for d in docs:
    m = re.search(r"\.(LQ|NĐ|TT|QĐ|PL|TL|NQ|CT|BL|LT|TTLT)\.", d.metadata.get("article_code", ""))
    if not m: continue
    code = m.group(1); seen[code] += 1
    expect = code_map.get(code, "")
    if d.metadata.get("doc_type", "") != expect:
        mism[(code, d.metadata.get("doc_type", "") or "(rong)")] += 1
print("ma loai van ban trong article_code:", dict(seen))
print("doc_type sai so voi ma loai:", sum(mism.values()), mism.most_common(8))

dup = collections.Counter(d.page_content.split("\n", 3)[-1][:400] for d in docs if len(d.page_content) > 300)
print("Dieu co noi dung trung lap (cung 400 ky tu dau sau tieu de):", sum(c - 1 for c in dup.values() if c > 1))
rel = [d for d in docs if d.metadata.get("related_text")]
print("Dieu co related_text (pChiDan) nam trong page_content:", pct(len(rel), N))

print("\n== CHUNKING ==")
chunks = split_documents(docs)
C = len(chunks)
lens = [len(c.page_content) for c in chunks]
print("chunk:", C, "| ky tu median", statistics.median(lens), "p90", q(lens, 90), "| < 200 ky tu:", pct(sum(l < 200 for l in lens), C))
per_doc = collections.Counter((c.metadata["source_file"], c.metadata.get("article_index")) for c in chunks)
print("chunk/Dieu: 1 chunk:", pct(sum(v == 1 for v in per_doc.values()), len(per_doc)), "| >10 chunk:", sum(v > 10 for v in per_doc.values()), "| max:", max(per_doc.values()))
def has_header(c):
    code = c.metadata.get("article_code")
    return bool(code) and code in c.page_content
nohdr = sum(not has_header(c) for c in chunks)
print("chunk KHONG chua ma Dieu trong text (embedding khong biet thuoc Dieu nao):", pct(nohdr, C))
bad_end = sum(1 for c in chunks if not re.search(r"[.;:!?)\]”\"]\s*$", c.page_content.strip()))
print("chunk ket thuc giua cau:", pct(bad_end, C))
meta_bytes = [len(json.dumps(c.metadata, ensure_ascii=False).encode()) for c in chunks]
text_bytes = [len(c.page_content.encode()) for c in chunks]
print(f"metadata/chunk: median {statistics.median(meta_bytes)} B | tong metadata {sum(meta_bytes)/1e6:.1f} MB vs text {sum(text_bytes)/1e6:.1f} MB")
heavy = sum(len(json.dumps({k: c.metadata.get(k) for k in ('urls','related_urls','related_text','legal_note')}, ensure_ascii=False).encode()) for c in chunks)
print(f"  trong do urls/related_*/legal_note lap lai o moi chunk: {heavy/1e6:.1f} MB")

# BM25: tu khoa bi dem 2 lan vi metadata vua nam trong page_content vua duoc noi them
c0 = next(c for c in chunks if c.metadata.get("legal_note"))
print("legal_note xuat hien trong page_content cua chunk dau Dieu:", c0.metadata["legal_note"][:40] in c0.page_content)

print("\n== EMBEDDING INPUT ==")
try:
    from sentence_transformers import SentenceTransformer
    name = "bkai-foundation-models/vietnamese-bi-encoder"
    model = SentenceTransformer(name, device="cpu")
    print("max_seq_length:", model.max_seq_length)
    tok = model.tokenizer
    sample = chunks[:: max(C // 2000, 1)]
    ntok = [len(tok(c.page_content, add_special_tokens=True)["input_ids"]) for c in sample]
    print(f"token/chunk (mau {len(sample)}): median {statistics.median(ntok)} p90 {q(ntok,90)} | > max_seq_length (bi cat duoi):",
          pct(sum(n > model.max_seq_length for n in ntok), len(sample)))
    kept = [min(1.0, model.max_seq_length / n) for n in ntok]
    print(f"ty le noi dung chunk thuc su duoc embed: trung binh {100*statistics.mean(kept):.0f}%")
    s = "Đối tượng áp dụng hiến lấy ghép mô"
    print("tokenize khong tach tu:", tok.tokenize(s))
    print("tokenize da tach tu   :", tok.tokenize("Đối_tượng áp_dụng hiến lấy ghép mô"))
    unk = sum(t == tok.unk_token for c in sample[:300] for t in tok.tokenize(c.page_content[:1000]))
    print("so token <unk> trong 300 chunk mau:", unk)
except Exception as exc:
    print("khong phan tich duoc tokenizer:", exc)

print("\n== WIKI LOOKUP (## heading) ==")
recs = [json.loads(l) for l in open("wiki_data/raw_sources/articles.jsonl")]
cache = {}; wrong = 0; missing = 0; truncated = 0
for r in recs:
    path = "wiki_data/raw_sources/" + r["source_markdown"]
    text = cache.setdefault(path, open(path, encoding="utf-8").read())
    marker = f"## {r['article_heading']}"
    start = text.find(marker)
    if start < 0: missing += 1; continue
    line_end = text.find("\n", start)
    if text[start:line_end].strip() != marker.strip(): wrong += 1; continue
    idx = re.search(r"- Article index: (\d+)", text[start:start + 600])
    if idx and int(idx.group(1)) != r["metadata"]["article_index"]: wrong += 1; continue
    nxt = text.find("\n## ", start + len(marker)); sec = text[start: nxt if nxt >= 0 else len(text)]
    if len(sec) > 3000: truncated += 1
print("Dieu wiki tra ve NHAM Dieu khac:", wrong, "| khong tim thay:", missing, "| bi cat 3000 ky tu:", pct(truncated, len(recs)))
print("wiki BM25 chi index content_preview (500 ky tu) — Dieu dai hon 500 ky tu:", pct(sum(len(d.page_content) > 500 for d in docs), N))
