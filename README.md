# Legal RAG Compare System

Hệ thống RAG hỏi đáp pháp luật Việt Nam (dữ liệu Pháp điển). Cùng một câu hỏi được trả lời theo hai cách rồi so sánh song song:

- **Vector RAG**: hybrid parent-child. FAISS tìm các đoạn con (cụm khoản ≤ 256 token, có tiêu đề Điều), BM25 chấm nguyên Điều, hai bảng xếp hạng được gộp rồi rerank theo Điều.
- **Wiki RAG**: chỉ trả lời từ lớp wiki do LLM viết (kiểu "LLM Wiki" của Karpathy). LLM đọc `index.md`, chọn đề mục và trang wiki phù hợp rồi trả lời từ các trang đó, không dùng BM25.

Câu trả lời, context truy xuất và metric từng lần chạy hiển thị cạnh nhau trên giao diện Gradio `http://localhost:7860/ui`.

## Kết quả đánh giá (bản chuẩn cuối cùng)

Đo trên bộ golden 150 câu (`eval/golden.jsonl`, mỗi câu có một Điều đáp án) và 6 câu tổng hợp nhiều Điều (`eval/synthesis.jsonl`). Bản truy xuất được đo là commit `3df8dc2`: viết lại câu hỏi, mở rộng từ viết tắt, ưu tiên loại Điều.

### Truy xuất (vector, top 5)

| Bộ câu hỏi | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| Có dấu, trước cải thiện | 0.653 | 0.913 | 0.757 |
| **Có dấu, bản hiện tại** | **0.673** | **0.953** | **0.780** |
| Có dấu, tính cả Điều trùng nội dung (`--equivalents`) | 0.747 | 0.980 | 0.841 |
| Không dấu, không viết lại câu hỏi | 0.053 | 0.147 | 0.085 |
| **Không dấu, có LLM viết lại câu hỏi** (20 câu) | **0.700** | **0.900** | **0.783** |

Thời gian truy xuất (không gọi LLM trả lời): p50 khoảng 1,5 s, p95 khoảng 2,2 s trên CPU. Bước viết lại câu hỏi thêm khoảng 1 s và khoảng 2 nghìn token cho mỗi câu hỏi mới.

### Chất lượng câu trả lời (RAGAS)

Model trả lời là `qwen3-14b` (tắt suy luận). Giám khảo là `deepseek-v4-flash` (tắt suy luận). Kết quả trên 150 câu golden của vector RAG:

| Chỉ số | Giám khảo deepseek | **Bản chuẩn cuối cùng** |
| --- | ---: | ---: |
| Faithfulness: câu trả lời có bám context không | 0.688 | **0.795** |
| Answer relevancy: có trả lời đúng câu hỏi không | 0.908 | **0.907** |
| Context precision: context đúng có đứng đầu không | 0.831 | **0.840** |
| Context recall: context có đủ ý câu trả lời mẫu không | 0.819 | **0.902** |

**Cách ra bản chuẩn cuối cùng:**

- **Câu được chấm lại:** 143/150 câu lấy được đúng Điều đáp án. Trong số đó, 29 câu bị deepseek chấm faithfulness hoặc context recall dưới 0.5. Đọc lại thì phần lớn là giám khảo chấm sai:
  - Lúc đo, script chấm chỉ đưa nội dung Điều cho giám khảo, không đưa dòng `Citation` chứa số hiệu và ngày ban hành văn bản. Câu trả lời trích "Thông tư 188/2014/TT-BTC" vì thế bị coi là bịa. Lỗi này đã được sửa (xem phần RAGAS bên dưới).
  - Có câu deepseek chấm recall 0 dù Điều lấy được chứa nguyên văn câu trả lời mẫu.
- **Cách chấm lại:** 29 câu đó được `claude-sonnet-4-6` chấm lại đủ 4 chỉ số, với context giống hệt phần qwen thấy khi trả lời (tiêu đề + `Citation` + nội dung). Các câu còn lại giữ điểm deepseek.
- **Lỗi thật vẫn bị tính:**
  - 7 câu truy xuất trượt giữ nguyên điểm thấp.
  - Các câu qwen đọc đúng Điều nhưng hiểu sai con số vẫn bị trừ, ví dụ trả lời "ngày 05 hàng tháng" trong khi luật ghi "thứ 2 hàng tuần".
- **Giới hạn:** bảng này ghép điểm của hai giám khảo và chỉ chấm lại các câu bị chấm thấp. Nếu deepseek chấm quá dễ ở câu nào thì sai số đó không được sửa, nên con số có thể hơi lạc quan.
- **Tóm lại:** hệ thống vector đạt khoảng **faithfulness 0.80, relevancy 0.91, precision 0.84, recall 0.90**. Phần còn thiếu chủ yếu đến từ khoảng 5% câu truy xuất trượt và vài câu qwen3-14b hiểu sai số liệu.

**Các nhóm nhỏ** (deepseek chấm, chưa hiệu chỉnh, mỗi nhóm chỉ 6 câu nên sai số lớn):

| Nhóm | Hệ thống | Faithfulness | Relevancy | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Câu tổng hợp nhiều Điều | vector | 0.806 | 0.892 | 0.645 | 0.519 |
| Câu tổng hợp nhiều Điều | wiki | 0.891 | 0.886 | 0.681 | 0.732 |
| Câu golden thuộc đề mục đã có wiki | vector | 0.782 | 0.940 | 0.736 | 0.583 |
| Câu golden thuộc đề mục đã có wiki | wiki | 0.944 | 0.976 | 0.750 | 0.833 |

Wiki mới được xây cho 3 đề mục nên chỉ trả lời 14/150 câu golden. Trong phạm vi đã có wiki, wiki trả lời tốt hơn vector, nhất là với câu cần tổng hợp nhiều Điều.

**File kết quả** trong `eval/results/`:

| Lần đo | File |
| --- | --- |
| Truy xuất | `meta-0.json`, `v4-intent.json`, `v5-rewrite-20.json` |
| RAGAS | `ragas-qwen14b-v2.json` và các file `-answers.jsonl`, `-scores.jsonl` |
| Chấm lại bằng Sonnet | `ragas-qwen14b-v2-sonnet.json` (script `eval/rejudge_sonnet.py`) |

Chi phí giám khảo:

| Giám khảo | Token | Số lượt gọi |
| --- | ---: | ---: |
| deepseek-v4-flash | 7,08 triệu | 1.718 |
| Sonnet | 1,44 triệu | 319 |

## Kiến trúc

```text
api-gateway (FastAPI + Gradio tại /ui)
  /ask/compare ─┬─> vector-rag-service /answer ─┐
                └─> wiki-rag-service /wiki/answer ─┴─> llm-service /chat
                rồi ─> evaluator-service /eval/run (metric → SQLite)
```

| Service | Cổng | Vai trò |
| --- | ---: | --- |
| `api-gateway` | 7860 | UI so sánh, API `/ask/compare`, `/ask/vector`, `/ask/wiki` |
| `vector-rag-service` | 8001 | Truy xuất hybrid và trả lời |
| `wiki-rag-service` | 8002 | Chọn trang wiki bằng LLM và trả lời; tự nạp lại khi wiki được build lại |
| `llm-service` | 8003 | Adapter gọi API chat tương thích OpenAI (DeepSeek, hoặc proxy như 9router) |
| `evaluator-service` | 8004 | Lưu log và metric từng lần so sánh |

Các bước truy xuất của vector RAG:

1. **Viết lại câu hỏi** (`services/vector_rag_service/query_rewrite.py`): LLM thêm dấu cho từ gõ thiếu dấu. FAISS và BM25 coi "thue" và "thuế" là hai từ khác nhau.
   - Chỉ bước truy xuất dùng bản viết lại; LLM trả lời vẫn nhận câu hỏi gốc.
   - Nếu LLM lỗi hoặc đổi quá 30% số từ, hệ thống dùng lại câu gốc.
2. **Mở rộng từ viết tắt** theo cả hai chiều (CMND ↔ chứng minh nhân dân, UBND, BHXH…) trước BM25 và rerank.
3. **Lấy ứng viên:** FAISS lấy các đoạn con rồi gom theo Điều; BM25 xếp hạng nguyên Điều.
4. **Gộp hai bảng xếp hạng** bằng reciprocal rank fusion có trọng số (0.55 / 0.45).
5. **Rerank:** điểm cuối = 0.65 × rerank + 0.35 × điểm gộp, cả hai chuẩn hóa min-max.
6. **Ưu tiên loại Điều:** cộng điểm cho Điều có tiêu đề khớp loại câu hỏi. Ví dụ "những ai…" → "Đối tượng áp dụng", "phạm vi…" → "Phạm vi điều chỉnh", "có hiệu lực khi nào" → "Hiệu lực".

Công nghệ:
- Python 3.10, FastAPI, Gradio, Docker Compose
- FAISS và BM25
- Embedding `bkai-foundation-models/vietnamese-bi-encoder`, chạy bằng ONNX Runtime trên CPU
- Quản lý dependency bằng uv (`pyproject.toml`, `uv.lock`)
- SQLite cho log đánh giá
- RAGAS 0.4 để chấm câu trả lời

Mặc định chạy trên CPU (máy 16 GB RAM, GPU cũ). Việc build index nặng nên chạy trên máy có GPU rồi chép index về.

**Embedding bằng ONNX Runtime** (`EMBEDDING_RUNTIME=onnx`, mặc định):
- **Cách hoạt động:** lần đầu chạy, model sentence-transformers được export sang ONNX, tối ưu đồ thị, rồi cache trong volume `model_cache` (`/cache/onnx`).
- **Kết quả trên 150 câu golden:**
  - vector khớp bản PyTorch (sai khác < 10⁻⁶);
  - top-5 truy xuất giống hệt cả 150 câu;
  - thời gian nhúng một câu hỏi giảm từ 77 ms xuống 34 ms.
- **Không lượng tử hóa int8:** với model này trên CPU chỉ có AVX2, int8 làm vector lệch quá nhiều (cosine chỉ còn 0.71–0.84).
- **Khi nào dùng PyTorch:** khi có GPU, hoặc khi đặt `EMBEDDING_RUNTIME=torch`. Cả hai runtime dùng chung một index.

## Cấu trúc dữ liệu

- `data_raw/`: HTML pháp luật gốc, là nguồn sự thật.
- `wiki_data/raw_sources/`: Markdown sinh từ `data_raw`, là đầu vào để viết wiki.
- `wiki_data/wiki/`: các trang wiki do LLM viết (`make build-wiki-layer`).
- `eval/`: bộ golden, câu tổng hợp, Điều tương đương, cache reference và prompt RAGAS, kết quả đo.
- Docker volume:
  - `vector_faiss_index`: FAISS
  - `vector_hybrid_index`: BM25 và manifest
  - `eval_data`: log và metric
  - `model_cache`: cache model embedding
  - `rag_eval_deps`: thư viện RAGAS, không cài vào image service

Dữ liệu raw và index không được chép vào Docker image mà được mount hoặc lưu trong volume.

## Cài đặt

Dependency được quản lý bằng [uv](https://docs.astral.sh/uv/): `pyproject.toml` khai báo, `uv.lock` ghim phiên bản. Image Docker cài bằng `uv sync --frozen`.

```bash
cp .env.example .env
python3 check_env.py
uv sync --extra embeddings     # chỉ cần khi chạy ngoài Docker; torch lấy từ index CPU của PyTorch
```

Thêm hoặc đổi dependency: sửa `pyproject.toml`, chạy `uv lock`, rồi `make docker-build`. Nhóm `eval` (ragas) không vào image mà được cài vào volume riêng bằng `make eval-ragas-deps`.

Sửa `.env`: điền API key và endpoint chat tương thích OpenAI.

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
# Tắt suy luận nếu provider cho phép (tên tham số tùy provider):
# DEEPSEEK_EXTRA_BODY={"thinking":{"type":"disabled"}}
```

Kết quả ở trên dùng `DEEPSEEK_MODEL=cl/qwen/qwen3-14b` qua proxy 9router, với `DEEPSEEK_EXTRA_BODY={"reasoning":{"enabled":false}}`. Không commit file `.env`.

## Chạy bằng Docker

```bash
make docker-build
make prepare-wiki-sources                    # data_raw -> wiki_data/raw_sources
make docker-build-vector-index               # FAISS + BM25
make build-wiki-layer SOURCES="aec013b7 059ac230 62aa0284"   # LLM viết wiki cho các đề mục này (tốn token)
make docker-up                               # mở http://localhost:7860/ui
make docker-logs                             # hoặc docker-vector-logs, docker-wiki-logs
make docker-down
```

Phải build dữ liệu trước `docker-up`. Thiếu index thì healthcheck của vector và wiki service lỗi, và gateway không khởi động.

## API chính

```text
GET  /health
POST /ask/compare
POST /ask/vector
POST /ask/wiki
GET  /ui
```

```bash
curl -s localhost:7860/ask/compare -H 'content-type: application/json' \
  -d '{"question":"đối tượng áp dụng hiến lấy ghép mô","top_k":2,"candidate_k":20,"use_llm":false}'
```

Với `use_llm: false`, câu trả lời được trích thẳng từ context, không gọi LLM để viết. LLM vẫn được gọi ở hai bước: wiki service chọn trang, và vector service viết lại câu hỏi.

## Đánh giá

**Truy xuất trên bộ golden** (Recall@1/@5, MRR, latency cho câu có dấu và không dấu):

```bash
make eval-golden                                          # sinh eval/golden.jsonl bằng LLM
make eval-retrieval EVAL_TAG=moi ARGS="--compare baseline"
make eval-retrieval EVAL_TAG=moi ARGS="--services vector --equivalents eval/golden_equivalents.jsonl"
```

Các tùy chọn:
- `--compare baseline`: báo số danh sách top-k giống hệt lần đo trước. Một lần refactor thuần phải giữ nguyên tất cả.
- `--services vector`: bỏ qua wiki, vì chọn trang wiki tốn token.
- `--equivalents`: tính trúng cả Điều khác có cùng nội dung với Điều đáp án. File này do LLM dựng (`scripts/golden_equivalents.py`), cần rà lại trước khi tin con số.

**Chất lượng câu trả lời bằng RAGAS** (cần stack đang chạy):

```bash
make eval-ragas-deps                                      # một lần: cài RAGAS vào volume rag_eval_deps
make eval-ragas EVAL_TAG=thu ARGS="--limit 5"             # thử nhanh
make eval-ragas EVAL_TAG=ragas-qwen14b-v2 ARGS="--judge-model cl/deepseek/deepseek-v4-flash"
```

Cách RAGAS chạy:
- **Giám khảo:** nên dùng một model mạnh, khác họ với model trả lời, và giữ cố định khi so sánh các lần đo (`--judge-model`). Nếu để model tự chấm câu trả lời của chính nó, faithfulness và relevancy bị thiên lệch.
- **Cache:**
  - Reference, câu trả lời và điểm được cache trong `eval/`.
  - Điểm được lưu sau mỗi lô 12 câu, nên chạy lại sau khi bị ngắt sẽ tiếp tục từ chỗ dừng.
  - `--answers-only` chỉ sinh câu trả lời, không chấm.
- **Context của giám khảo:** giám khảo thấy đúng phần LLM trả lời đã thấy: tiêu đề, dòng `Citation` và nội dung (`shared/answering.format_context`). Điểm cũ trong `eval/results/` chấm trước thay đổi này chỉ có nội dung; điểm được cache theo tag và giám khảo, nên muốn chấm lại bản mới thì dùng tag mới.
- **Hết tiền:** khi gặp lỗi hết tiền hoặc hết quota (402, "insufficient balance"), script dừng ngay thay vì thử lại.
- **Chi phí:** khoảng 40–47 nghìn token giám khảo cho mỗi câu, tức 7–8 triệu token cho một lần đo đủ.

## Ghi chú vận hành

- Lần khởi động đầu mất vài giây để nạp trước vector index và lớp wiki.
- Wiki service chỉ trả lời được các đề mục đã build; câu ngoài phạm vi nhận "không tìm thấy trang".
- Khi kết quả bất thường, xem log service và các trường `error`, `fallback` trước khi nghi phần truy xuất. Service trả lỗi bằng HTTP 200 kèm `error`, và llm-service trả văn bản tạm khi gọi LLM lỗi.
- Chi tiết kỹ thuật cho người phát triển xem `CLAUDE.md`.
