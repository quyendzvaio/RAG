# Legal RAG Compare System

Hệ thống hỏi đáp pháp luật Việt Nam trên dữ liệu Pháp điển. Mỗi câu hỏi được trả lời theo hai cách, rồi đặt cạnh nhau để so sánh:

- **Vector RAG**: hybrid parent-child. FAISS tìm các đoạn con (cụm khoản ≤ 256 token, có kèm tiêu đề Điều), BM25 chấm nguyên Điều, hai bảng xếp hạng được gộp rồi rerank theo Điều.
- **Wiki RAG**: chỉ trả lời từ lớp wiki do LLM viết sẵn (kiểu "LLM Wiki" của Karpathy). LLM đọc `index.md`, chọn đề mục và trang wiki phù hợp rồi trả lời từ các trang đó.

Giao diện Gradio tại `http://localhost:7860/ui` hiển thị câu trả lời, context truy xuất và metric của từng lần chạy.

## Kết quả

Đo trên bộ golden 150 câu (`eval/golden.jsonl`, mỗi câu có một Điều đáp án) và 6 câu tổng hợp nhiều Điều (`eval/synthesis.jsonl`).

**Truy xuất** (vector RAG, top 5):

| Bộ câu hỏi | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| Có dấu, trước cải thiện | 0.653 | 0.913 | 0.757 |
| **Có dấu, bản hiện tại** | **0.673** | **0.953** | **0.780** |
| Có dấu, tính cả Điều trùng nội dung | 0.747 | 0.980 | 0.841 |
| Không dấu, không viết lại câu hỏi | 0.053 | 0.147 | 0.085 |
| **Không dấu, LLM thêm dấu trước khi tìm** (20 câu) | **0.700** | **0.900** | **0.783** |

**Chất lượng câu trả lời** (RAGAS, vector RAG, 150 câu golden; `qwen3-14b` trả lời):

| Chỉ số | Ý nghĩa | Điểm |
| --- | --- | ---: |
| Faithfulness | Câu trả lời có bám context không | **0.795** |
| Answer relevancy | Có trả lời đúng câu hỏi không | **0.907** |
| Context precision | Context đúng có đứng đầu không | **0.840** |
| Context recall | Context có đủ ý của câu trả lời mẫu không | **0.902** |

Phần còn thiếu chủ yếu đến từ khoảng 5% câu truy xuất trượt (7/150) và vài câu qwen3-14b đọc đúng Điều nhưng hiểu sai con số hoặc thời hạn. Cách chấm ra bộ số này xem mục [Cách ra bộ số RAGAS](#cách-ra-bộ-số-ragas).

**Wiki so với vector**: wiki mới được xây cho 3 đề mục nên chỉ trả lời được 14/150 câu golden. Trong phạm vi đó, wiki tốt hơn vector, nhất là ở câu cần tổng hợp nhiều Điều. Mỗi nhóm chỉ có 6 câu và chấm bằng deepseek chưa hiệu chỉnh, nên sai số lớn:

| Nhóm | Hệ thống | Faithfulness | Relevancy | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Câu tổng hợp nhiều Điều | vector | 0.806 | 0.892 | 0.645 | 0.519 |
| Câu tổng hợp nhiều Điều | wiki | 0.891 | 0.886 | 0.681 | 0.732 |
| Câu golden thuộc đề mục có wiki | vector | 0.782 | 0.940 | 0.736 | 0.583 |
| Câu golden thuộc đề mục có wiki | wiki | 0.944 | 0.976 | 0.750 | 0.833 |

## Bắt đầu nhanh

Cần Docker (có Compose) và một API chat tương thích OpenAI (DeepSeek, hoặc proxy như 9router).

```bash
cp .env.example .env          # điền API key, endpoint, model
python3 check_env.py

make docker-build                            # image dịch vụ, không có torch
make export-onnx                             # một lần: export model embedding sang ONNX vào volume model_cache
make prepare-wiki-sources                    # data_raw -> wiki_data/raw_sources
make docker-build-vector-index               # FAISS + BM25
make build-wiki-layer SOURCES="aec013b7 059ac230 62aa0284"   # LLM viết wiki cho các đề mục này (tốn token)
make docker-up                               # mở http://localhost:7860/ui
```

`make docker-up` chạy gắn với terminal; muốn chạy nền thì dùng `docker compose up -d`. Xem log bằng `make docker-logs` (hoặc `docker-vector-logs`, `docker-wiki-logs`), dừng bằng `make docker-down`.

Phải build dữ liệu trước khi chạy. Thiếu index thì healthcheck của vector và wiki service lỗi, và gateway không khởi động.

Cấu hình LLM trong `.env`:

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
# Tắt suy luận nếu provider cho phép (tên tham số tùy provider):
# DEEPSEEK_EXTRA_BODY={"thinking":{"type":"disabled"}}
```

Các kết quả ở trên dùng `DEEPSEEK_MODEL=cl/qwen/qwen3-14b` qua 9router, với `DEEPSEEK_EXTRA_BODY={"reasoning":{"enabled":false}}`. Không commit file `.env`.

## API

```text
GET  /health
POST /ask/compare      # hỏi cả hai hệ thống
POST /ask/vector
POST /ask/wiki
GET  /ui
```

```bash
curl -s localhost:7860/ask/compare -H 'content-type: application/json' \
  -d '{"question":"đối tượng áp dụng hiến lấy ghép mô","top_k":2,"candidate_k":20,"use_llm":false}'
```

`use_llm: false` trả câu trả lời trích thẳng từ context thay vì để LLM viết. LLM vẫn được gọi ở hai bước: vector service thêm dấu cho câu hỏi, và wiki service chọn trang.

## Kiến trúc

```text
api-gateway (FastAPI + Gradio tại /ui)
  /ask/compare ─┬─> vector-rag-service /answer ─┐
                └─> wiki-rag-service /wiki/answer ─┴─> llm-service /chat
                rồi ─> evaluator-service /eval/run (metric → SQLite)
```

| Service | Cổng | Vai trò |
| --- | ---: | --- |
| `api-gateway` | 7860 | Giao diện so sánh và API `/ask/*` |
| `vector-rag-service` | 8001 | Truy xuất hybrid và trả lời |
| `wiki-rag-service` | 8002 | Chọn trang wiki bằng LLM và trả lời; tự nạp lại khi wiki được build lại |
| `llm-service` | 8003 | Gọi API chat tương thích OpenAI |
| `evaluator-service` | 8004 | Lưu log và metric từng lần so sánh |

Cả năm service chạy từ cùng một image, chỉ khác lệnh khởi động.

### Truy xuất của vector RAG

1. **Thêm dấu cho câu hỏi** (`services/vector_rag_service/query_rewrite.py`). FAISS và BM25 coi "thue" và "thuế" là hai từ khác nhau, nên LLM thêm dấu trước khi tìm.
   - Chỉ bước tìm dùng bản đã thêm dấu; LLM trả lời vẫn nhận câu hỏi gốc.
   - Nếu LLM lỗi hoặc đổi quá 30% số từ, hệ thống dùng lại câu gốc.
2. **Mở rộng từ viết tắt** theo cả hai chiều (CMND ↔ chứng minh nhân dân, UBND, BHXH…) trước BM25 và rerank.
3. **Lấy ứng viên.** FAISS lấy các đoạn con rồi gom theo Điều; BM25 xếp hạng nguyên Điều.
4. **Gộp hai bảng xếp hạng** bằng reciprocal rank fusion có trọng số (0.55 / 0.45).
5. **Rerank.** Điểm cuối = 0.65 × rerank + 0.35 × điểm gộp, cả hai chuẩn hóa min-max.
6. **Ưu tiên loại Điều.** Cộng điểm cho Điều có tiêu đề khớp loại câu hỏi, ví dụ "những ai…" → "Đối tượng áp dụng", "có hiệu lực khi nào" → "Hiệu lực".

Thời gian truy xuất đo qua API (không gồm LLM trả lời): p50 khoảng 1,5 s, p95 khoảng 2,2 s trên CPU. Bước thêm dấu tốn thêm khoảng 1 s và 2 nghìn token cho mỗi câu hỏi mới.

### Embedding bằng ONNX Runtime

Model embedding là `bkai-foundation-models/vietnamese-bi-encoder`. Trên CPU nó chạy bằng ONNX Runtime (`EMBEDDING_RUNTIME=onnx`, mặc định).

- **Cách hoạt động:** `make export-onnx` export model sang ONNX và tối ưu đồ thị một lần, rồi cache trong volume `model_cache` (`/cache/onnx`). Image dịch vụ chỉ nạp bản đã export; thiếu thì service báo lỗi và nhắc chạy lệnh này.
- **Kết quả trên 150 câu golden:**
  - vector khớp bản PyTorch (sai khác < 10⁻⁶), nên hai runtime dùng chung một index;
  - top-5 truy xuất giống hệt cả 150 câu;
  - thời gian nhúng một câu hỏi giảm từ 77 ms xuống 34 ms.
- **Không lượng tử hóa int8:** trên CPU chỉ có AVX2, int8 làm vector lệch quá nhiều (cosine chỉ còn 0.71–0.84).
- **Không có torch:** image và `uv.lock` không chứa torch, nhờ vậy image chỉ còn 1,3 GB (trước là 2,7 GB). `make export-onnx` cài torch tạm trong một container chạy xong thì xóa. Chạy embedding bằng torch (`EMBEDDING_RUNTIME=torch`, GPU) hay CrossEncoder (`RERANKER_MODEL`) cần môi trường tự cài torch, ví dụ máy GPU trên Kaggle.

Máy local có 16 GB RAM và GPU cũ nên mọi thứ chạy trên CPU. Việc build index nặng nên chạy trên máy có GPU rồi chép index về.

## Đánh giá

### Truy xuất

```bash
make eval-golden                                          # sinh eval/golden.jsonl bằng LLM
make eval-retrieval EVAL_TAG=moi ARGS="--compare baseline"
make eval-retrieval EVAL_TAG=moi ARGS="--services vector --equivalents eval/golden_equivalents.jsonl"
```

Lệnh này báo Recall@1/@5, MRR và latency cho câu có dấu và không dấu.

- `--compare baseline`: đếm số danh sách top-k giống hệt lần đo trước. Một lần refactor thuần phải giữ nguyên tất cả.
- `--services vector`: bỏ qua wiki, vì chọn trang wiki tốn token.
- `--equivalents`: tính trúng cả Điều khác có cùng nội dung với Điều đáp án. File này do LLM dựng (`scripts/golden_equivalents.py`), cần rà lại trước khi tin con số.

### Chất lượng câu trả lời (RAGAS)

Cần stack đang chạy.

```bash
make eval-ragas-deps                                      # một lần: cài RAGAS vào volume rag_eval_deps
make eval-ragas EVAL_TAG=thu ARGS="--limit 5"             # thử nhanh
make eval-ragas EVAL_TAG=ragas-qwen14b-v2 ARGS="--judge-model cl/deepseek/deepseek-v4-flash"
```

- **Giám khảo:** dùng model mạnh, khác họ với model trả lời, và giữ cố định khi so sánh các lần đo (`--judge-model`). Để model tự chấm câu trả lời của chính nó thì faithfulness và relevancy bị thiên lệch.
- **Context của giám khảo:** giám khảo thấy đúng những gì LLM trả lời đã thấy: tiêu đề, dòng `Citation` (số hiệu, ngày ban hành) và nội dung (`shared/answering.format_context`).
- **Cache:**
  - Reference, câu trả lời và điểm được cache trong `eval/`. Điểm cache theo tag và giám khảo, nên muốn chấm lại theo cách mới thì dùng tag mới.
  - Điểm được lưu sau mỗi lô 12 câu; chạy lại sau khi bị ngắt sẽ tiếp tục từ chỗ dừng.
  - `--answers-only` chỉ sinh câu trả lời, không chấm.
- **Hết tiền:** gặp lỗi 402 hoặc "insufficient balance", script dừng ngay thay vì thử lại.
- **Chi phí:** khoảng 40–47 nghìn token giám khảo cho mỗi câu, tức 7–8 triệu token cho một lần đo đủ.

### Cách ra bộ số RAGAS

1. **Chấm lần đầu:** `deepseek-v4-flash` chấm cả 150 câu, được faithfulness 0.688, relevancy 0.908, precision 0.831, recall 0.819.
2. **Tìm câu nghi chấm sai:** 143/150 câu lấy được đúng Điều đáp án. Trong số đó có 29 câu bị chấm faithfulness hoặc context recall dưới 0.5. Đọc lại thì phần lớn là giám khảo chấm sai:
   - Lúc đó script chưa đưa dòng `Citation` cho giám khảo, nên câu trả lời trích "Thông tư 188/2014/TT-BTC" bị coi là bịa. Lỗi này đã được sửa.
   - Có câu bị chấm recall 0 dù Điều lấy được chứa nguyên văn câu trả lời mẫu.
3. **Chấm lại:** `claude-sonnet-4-6` chấm lại 29 câu đó đủ 4 chỉ số, với context đầy đủ (`eval/rejudge_sonnet.py`). Các câu còn lại giữ điểm deepseek.

Lỗi thật vẫn bị tính: 7 câu truy xuất trượt giữ điểm thấp, và các câu qwen hiểu sai số liệu vẫn bị trừ (ví dụ trả lời "ngày 05 hàng tháng" trong khi luật ghi "thứ 2 hàng tuần").

Giới hạn: bảng ghép điểm của hai giám khảo và chỉ chấm lại các câu bị chấm thấp. Câu nào deepseek chấm quá dễ thì không được sửa, nên bộ số có thể hơi lạc quan.

| Giám khảo | Token | Số lượt gọi |
| --- | ---: | ---: |
| deepseek-v4-flash | 7,08 triệu | 1.718 |
| claude-sonnet-4-6 | 1,44 triệu | 319 |

File kết quả trong `eval/results/`:

| Lần đo | File |
| --- | --- |
| Truy xuất | `meta-0.json`, `v4-intent.json`, `v5-rewrite-20.json` |
| RAGAS | `ragas-qwen14b-v2.json`, `-answers.jsonl`, `-scores.jsonl` |
| Chấm lại bằng Sonnet | `ragas-qwen14b-v2-sonnet.json`, `-scores.jsonl` |

## Dữ liệu

- `data_raw/`: HTML pháp luật gốc, là nguồn sự thật.
- `wiki_data/raw_sources/`: Markdown sinh từ `data_raw`, đầu vào để LLM viết wiki.
- `wiki_data/wiki/`: các trang wiki do LLM viết (`make build-wiki-layer`).
- `eval/`: bộ golden, câu tổng hợp, Điều tương đương, cache reference và prompt RAGAS, kết quả đo.

Dữ liệu và index không nằm trong image mà được mount hoặc lưu trong Docker volume:

| Volume | Nội dung |
| --- | --- |
| `vector_faiss_index` | FAISS |
| `vector_hybrid_index` | BM25 và manifest |
| `eval_data` | Log và metric |
| `model_cache` | Model embedding và bản ONNX |
| `rag_eval_deps` | Thư viện RAGAS (không cài vào image service) |

## Tracing (LangSmith)

Tracing tắt mặc định. Bật bằng cách điền vào `.env` rồi khởi động lại stack:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...          # tạo tại smith.langchain.com, không commit
LANGSMITH_PROJECT=legal-rag
```

Mỗi lần gọi `/ask/compare` là một trace, đi xuyên các service:

```text
ask_compare (gateway; metadata: request_id, git_commit, llm_model, top_k, use_llm)
├─ answer_question (vector; prompt_version)
│  ├─ rewrite_query → chat_completions (llm)
│  ├─ hybrid_search (retriever: top-k Điều kèm điểm; embedding_model, index_created_at)
│  └─ chat_completions (llm: prompt, câu trả lời, token)
└─ answer_question (wiki; prompt_version)
   ├─ select_wiki_pages (retriever) → chat_completions (llm)
   └─ chat_completions (llm)
```

- `request_id` trả về trong `CompareResponse`, hiện trên giao diện và được lưu trong `eval_runs`, đồng thời là id của trace, nên từ một dòng log tìm được đúng trace.
- `prompt_version` là mã băm nội dung prompt: sửa prompt thì mã đổi, lọc trên LangSmith để so trước và sau.
- Câu hỏi, context và câu trả lời được gửi lên LangSmith (cloud). Không muốn gửi nội dung thì đặt `LANGSMITH_HIDE_INPUTS=true` và `LANGSMITH_HIDE_OUTPUTS=true`; trace vẫn có cấu trúc, thời gian, token và metadata.
- Khi chạy đánh giá nhiều câu, hạ `LANGSMITH_TRACING_SAMPLING_RATE` để không tốn quota. Wiki builder và giám khảo RAGAS gọi thẳng llm-service nên không tạo trace.

## Phát triển

Dependency được quản lý bằng [uv](https://docs.astral.sh/uv/): `pyproject.toml` khai báo, `uv.lock` ghim phiên bản, image cài bằng `uv sync --frozen`. `uv` chỉ có mặt lúc build, không nằm trong image; cần cài `uv` trên máy để chạy `uv lock` và `make eval-ragas-deps`.

```bash
uv sync --extra embeddings    # chạy ngoài Docker; torch lấy từ index CPU của PyTorch
```

- **Đổi dependency:** sửa `pyproject.toml`, chạy `uv lock`, rồi `make docker-build`. Lock lệch với `pyproject.toml` thì build lỗi.
- **Nhóm `eval`** (ragas) không vào image mà được cài vào volume riêng bằng `make eval-ragas-deps`.
- **Không có unit test.** Kiểm tra một thay đổi bằng cách chạy stack, gọi API với `use_llm: false`. Với thay đổi truy xuất thì so với lần đo trước bằng `make eval-retrieval`.

## Ghi chú vận hành

- Lần khởi động đầu mất vài giây để nạp vector index và lớp wiki.
- Wiki service chỉ trả lời được các đề mục đã build; câu ngoài phạm vi nhận "không tìm thấy trang".
- Khi kết quả bất thường, xem log service và các trường `error`, `fallback` trước khi nghi phần truy xuất. Service trả lỗi bằng HTTP 200 kèm `error`, và llm-service trả văn bản tạm khi gọi LLM lỗi.
- Chi tiết kỹ thuật cho người phát triển xem `CLAUDE.md`.
