# Legal RAG Compare System

Du an nay xay dung he thong RAG cho du lieu phap luat Viet Nam. Muc tieu la so sanh song song hai cach truy xuat:

- Vector RAG: Structure-aware Hybrid RAG voi BM25 + FAISS vector search + reranking.
- Wiki RAG: Truy xuat tren lop Markdown wiki/raw-source da chuan hoa tu HTML phap luat.

He thong duoc tach theo kien truc microservice de moi service co database/index rieng, de benchmark chat luong cau tra loi, latency, citation va cac metric danh gia RAG.

## Kien Truc

```text
Frontend / UI
    |
API Gateway
    |-- vector-rag-service
    |-- wiki-rag-service
    |-- evaluator-service
    |-- llm-service
```

Thanh phan chinh:

- `api-gateway`: UI compare mode va cac API `/ask/compare`, `/ask/vector`, `/ask/wiki`.
- `vector-rag-service`: BM25 + FAISS + reranking tren chunk phap ly co cau truc.
- `wiki-rag-service`: truy xuat tu Markdown wiki/raw-source va SQLite DB rieng.
- `llm-service`: adapter goi DeepSeek API.
- `evaluator-service`: luu logs va tinh metrics danh gia RAG.

## Cong Nghe Su Dung

- Python 3.10
- FastAPI + Uvicorn cho microservices
- Gradio cho UI compare mode
- FAISS cho vector search
- BM25 lexical search
- Sentence Transformers cho embedding tieng Viet
- DeepSeek API cho answer synthesis
- SQLite cho wiki DB va evaluation logs
- Docker Compose cho dong goi va chay he thong

Embedding mac dinh:

```text
bkai-foundation-models/vietnamese-bi-encoder
```

Mac dinh he thong chay CPU-only de phu hop may 16GB RAM va GPU doi cu.

## Danh Gia Hai Phuong Phap

He thong co `evaluator-service` de ghi lai tung lan compare va tinh metric cho ca `vector-rag-service` va `wiki-rag-service`. Metrics duoc hien thi trong UI compare mode va luu vao SQLite DB rieng.

Bang metric hien tai:

| Metric | Y nghia |
| --- | --- |
| `vector_latency_ms`, `wiki_latency_ms` | Thoi gian xu ly cua tung service. |
| `vector_context_count`, `wiki_context_count` | So context truy xuat duoc. |
| `vector_citation_count`, `wiki_citation_count` | So citation/can cu phap ly tra ve. |
| `vector_question_context_overlap`, `wiki_question_context_overlap` | Do phu lexical giua cau hoi va context, dung lam proxy cho retrieval relevance. |
| `vector_question_answer_overlap`, `wiki_question_answer_overlap` | Do phu lexical giua cau hoi va cau tra loi, dung lam proxy cho answer relevance. |
| `vector_error`, `wiki_error` | 0 neu service thanh cong, 1 neu co loi. |
| `latency_delta_ms_vector_minus_wiki` | Gia tri duong neu Vector RAG cham hon Wiki RAG, am neu Vector RAG nhanh hon. |

Ket qua benchmark gan nhat tren CPU 16GB RAM, `top_k=2`, `candidate_k=20`:

| Che do | Tong thoi gian | Vector RAG | Wiki RAG | Loi |
| --- | ---: | ---: | ---: | --- |
| Retrieval-only | ~484 ms | ~436 ms | ~230 ms | Khong |
| Retrieval + DeepSeek LLM | ~3.47 s | ~2.81 s | ~3.42 s | Khong |

Nhan xet:

- Vector RAG phu hop khi can semantic search tren cac cach dien dat khac nhau.
- Wiki RAG nhanh va de trace hon khi cau hoi gan voi tu khoa/dieu khoan trong van ban.
- Compare mode giup chay hai phuong phap song song tren cung mot cau hoi de doi chieu latency, context, citation va chat luong cau tra loi.

## Cau Truc Du Lieu

- `data_raw/`: HTML phap luat goc, la source of truth.
- `wiki_data/raw_sources/`: Markdown/raw-source sinh tu `data_raw`.
- Docker volumes:
  - `vector_faiss_index`: FAISS index.
  - `vector_hybrid_index`: BM25 index va manifest.
  - `wiki_db`: SQLite DB cho wiki service.
  - `eval_data`: logs va metrics.
  - `model_cache`: cache model embedding.

Du lieu raw va index khong duoc copy vao Docker image. Chung duoc mount hoac luu trong Docker volumes.

## Cai Dat

Tao file `.env`:

```bash
cp .env.example .env
```

Sua `.env` va dien API key DeepSeek:

```env
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

Kiem tra cau hinh:

```bash
python3 check_env.py
```

## Chay Bang Docker

Build image:

```bash
make docker-build
```

Chuan hoa `data_raw` thanh wiki raw sources (chay trong container, ghi ra `wiki_data/raw_sources`):

```bash
make prepare-wiki-sources
```

Build vector index:

```bash
make docker-build-vector-index
```

Ingest wiki DB:

```bash
make docker-ingest-wiki
```

Chay toan bo he thong:

```bash
make docker-up
```

Mo UI:

```text
http://localhost:7860/ui
```

Dung he thong:

```bash
make docker-down
```

Xem logs:

```bash
make docker-logs
```

## API Chinh

Gateway public o cong `7860`:

```text
GET  /health
POST /ask/compare
POST /ask/vector
POST /ask/wiki
GET  /ui
```

Payload mau:

```json
{
  "question": "doi tuong ap dung hien lay ghep mo bo phan co the nguoi la gi",
  "top_k": 2,
  "candidate_k": 20,
  "use_llm": true
}
```

## Danh Gia Retrieval Tren Golden Set

```bash
make eval-golden                                   # sinh eval/golden.jsonl bang DeepSeek
make eval-retrieval EVAL_TAG=baseline              # can stack dang chay
make eval-retrieval EVAL_TAG=moi ARGS="--compare baseline"
```

Script goi `/ask/vector` va `/ask/wiki` voi `use_llm=false`, tinh Recall@1, Recall@5, MRR va latency cho cau hoi co dau va khong dau, luu ket qua vao `eval/results/<tag>.json`.

## Ghi Chu Van Hanh

- Lan khoi dong dau co the mat vai giay de preload vector index va wiki cache.
- Retrieval-only thuong nhanh hon nhieu so voi cau hoi co LLM.
- Neu khong muon goi DeepSeek, dat `use_llm=false` khi goi API.
- Khong commit file `.env` len GitHub.
