# Structure-aware Hybrid RAG cho du lieu phap luat

Du an nay xu ly HTML phap luat theo cau truc Dieu/Chuong, sau do truy xuat bang:

- BM25 lexical search
- FAISS vector search
- Reranking tren tap ung vien hop nhat

## Cai dependency

```bash
python3 -m pip install -r requirements.txt
```

## Cau hinh .env va DeepSeek

Tao file `.env` tu template:

```bash
cp .env.example .env
```

Sua `.env` va dien API key:

```env
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

Kiem tra app da nhan cau hinh chua:

```bash
python3 check_env.py
```

Lenh nay chi in trang thai `configured`, khong in API key. Hien tai UI van dang tra ket qua retrieval; cau hinh DeepSeek duoc chuan bi san cho buoc them generation/answer synthesis.

`sentence-transformers` la tuy chon nhung nen dung cho chat luong semantic tot:

```bash
python3 -m pip install sentence-transformers
```

Embedding mac dinh cho tieng Viet:

```text
bkai-foundation-models/vietnamese-bi-encoder
```

Model nay phu hop hon model multilingual MiniLM cu cho du lieu tieng Viet/phap ly. Co the doi nhanh bang `EMBEDDING_PROFILE`:

```bash
EMBEDDING_PROFILE=vietnamese python3 create_faiss.py
EMBEDDING_PROFILE=bge-m3 python3 create_faiss.py
EMBEDDING_PROFILE=multilingual-e5-base python3 create_faiss.py
```

Luu y: moi lan doi embedding model/profile, can build lai `faiss_index/` va `hybrid_index/`.

## Chuan hoa data_raw cho wiki-service

HTML goc trong `data_raw/` la source of truth va khong bi sua. De wiki-service ingest on dinh hon, tao lop normalized raw sources:

```bash
make prepare-wiki-sources
```

Lenh nay sinh:

```text
wiki_data/raw_sources/
  index.md
  manifest.json
  articles.jsonl
  sources/*.md
```

Y nghia:

- `sources/*.md`: moi file HTML thanh mot Markdown source, giu Dieu/Chuong/Ghi chu/Noi dung/Lien quan.
- `articles.jsonl`: moi dong la mot dieu khoan co metadata, hash va preview, phu hop cho worker/wiki-service doc theo stream ma khong nhan doi full content.
- `manifest.json`: checksum, article count, source count, metadata tong quat.
- `index.md`: catalog Obsidian-friendly cho lop raw source.

Wiki-service duoc doc `wiki_data/raw_sources/` nhung khong sua. Cac page LLM sinh ra nen nam o lop rieng, vi du `wiki_data/wiki/`.

Neu can JSONL chua full content tung dieu khoan, chay:

```bash
python3 prepare_wiki_sources.py --clean --include-jsonl-content
```

## Luu y cho GPU Quadro K2100M

Quadro K2100M co khoang 2GB VRAM va compute capability doi cu. PyTorch/sentence-transformers hien dai thuong khong chay CUDA on dinh tren GPU nay. Code mac dinh chay CPU cho embedding va reranker.

Kiem tra cach code chon thiet bi:

```bash
python3 check_device.py
```

Khuyen dung cho may nay:

```bash
python3 create_faiss.py
```

Neu van muon ep CUDA de thu nghiem:

```bash
FORCE_CUDA=1 EMBEDDING_DEVICE=cuda python3 create_faiss.py
```

Nhung voi K2100M, cach ep CUDA co rui ro loi `no kernel image`, `CUDA out of memory`, hoac model chay cham/khong on dinh.

## Tao hybrid index

Chay tu thu muc `RAG_TTCSN`:

```bash
python3 create_faiss.py
```

Lenh nay tao:

- `faiss_index/`
- `hybrid_index/bm25_store.pkl`
- `hybrid_index/index_manifest.json`

Manifest luu embedding backend/model/dimension. Neu config hien tai khong khop index, ung dung se yeu cau build lai de tranh load sai vector index.

Neu chua cai `sentence-transformers`, code se fallback sang hashing embedding de smoke test. Fallback nay khong phai embedding semantic chat luong cao.

De build nhanh bang hashing:

```bash
EMBEDDING_BACKEND=hashing python3 create_faiss.py
```

De dung embedding semantic:

```bash
EMBEDDING_BACKEND=sentence-transformers \
EMBEDDING_PROFILE=vietnamese \
python3 create_faiss.py
```

## Chay UI

```bash
python3 -m rag.app
```

Neu index duoc tao bang hashing, chay UI cung can cung backend:

```bash
EMBEDDING_BACKEND=hashing python3 -m rag.app
```

## Bat neural reranker tuy chon

Mac dinh reranker la lexical reranker nhe. Neu da cai `sentence-transformers`, co the bat cross-encoder:

```bash
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 python3 -m rag.app
```

Tren Quadro K2100M reranker mac dinh chay CPU. Co the giam batch size neu RAM yeu:

```bash
RERANKER_BATCH_SIZE=4 RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 python3 -m rag.app
```

## Chay kien truc microservice bang Docker CPU-only

Mot Docker image CPU-only duoc dung lai cho nhieu service:

- Dung `python:3.10-slim`
- Cai PyTorch CPU-only tu index cua PyTorch
- Khong copy `data_raw/`, `wiki_data/`, `faiss_index/`, `hybrid_index/` vao image
- Mount raw data/read-only sources read-only
- Moi service co storage rieng qua Docker volumes

Services:

- `api-gateway`: compare UI va `/ask/compare`
- `vector-rag-service`: BM25 + FAISS tren raw legal chunks
- `wiki-rag-service`: query doc/wiki DB rieng tu `wiki_data/raw_sources`
- `llm-service`: DeepSeek adapter stateless
- `evaluator-service`: metrics/logs SQLite

Endpoints chinh:

- `api-gateway`: `POST /ask/compare`, `POST /ask/vector`, `POST /ask/wiki`, UI `/ui`
- `vector-rag-service`: `POST /search`, `POST /answer`, `GET /index/status`
- `wiki-rag-service`: `POST /wiki/ingest`, `POST /wiki/query`, `POST /wiki/answer`, `GET /wiki/status`
- `llm-service`: `POST /chat`
- `evaluator-service`: `POST /eval/run`, `GET /eval/runs`

Build image:

```bash
docker compose build
```

Chuan bi wiki raw sources tu `data_raw`:

```bash
make prepare-wiki-sources
```

Build vector index va ingest wiki DB bang job rieng:

```bash
make docker-build-vector-index
make docker-ingest-wiki
```

Hoac chay tat ca buoc data prep:

```bash
make docker-rebuild-all-data
```

Chay he thong:

```bash
docker compose up
```

Mo UI:

```text
http://localhost:7860
```

Runtime app mac dinh khong tu build index (`AUTO_BUILD_INDEX=false`). Neu chua co `hybrid_index/index_manifest.json`, container se thoat va bao can chay:

```bash
make docker-build-vector-index
```

Cach nay tranh cold start trong app runtime. Model cache, vector index, wiki DB va eval DB duoc giu trong named volumes.

Xem logs:

```bash
make docker-logs
```

Dung container:

```bash
docker compose down
```

Xoa ca index/model cache trong Docker volumes:

```bash
make docker-clean-volumes
```

Neu can image nho hon nua va chi test BM25/hash embedding, build khong cai embedding model:

```bash
docker compose build --build-arg INSTALL_EMBEDDINGS=false
EMBEDDING_BACKEND=hashing docker compose up
```

## Cach du lieu duoc xu ly

Parser uu tien cac class HTML:

- `pChuong`: Chuong va ten chuong
- `pDieu`: Dieu va tieu de dieu
- `pGhiChu`: can cu van ban, ngay hieu luc, URL nguon
- `pNoiDung`: noi dung dieu
- `pChiDan`: cac dieu/van ban lien quan

Moi chunk co metadata nhu `article_code`, `article_title`, `doc_type`, `effective_date`, `legal_note`, `urls`, `related_text`, `related_urls`, `source`.

Thu muc `demuc/` neu co chi la ban sao cua `data_raw/` va khong duoc pipeline su dung.
