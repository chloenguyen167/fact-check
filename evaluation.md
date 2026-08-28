# Kế hoạch & Kết quả Đánh giá Hệ thống Fact-Check Y tế

Tài liệu tổng hợp **phương pháp đánh giá**, **cách chạy thực nghiệm** và **kết quả mới nhất** của hệ thống. Chi tiết lệnh CLI xem thêm tại [`evaluation/README.md`](evaluation/README.md); báo cáo số liệu đầy đủ tại [`evaluation/ket_qua_danh_gia_day_du.md`](evaluation/ket_qua_danh_gia_day_du.md).

**Cập nhật:** 02/08/2026  
**Cấu hình chạy chính:** Embedding `BAAI/bge-m3`, Reranker `bge-reranker-v2-m3`, NLI `mDeBERTa-v3-base-xnli`, LLM giải thích `qwen3:8b` (Ollama)

---

## Mục lục

1. [Bài toán & phương pháp](#1-bài-toán--phương-pháp)
2. [Tập dữ liệu](#2-tập-dữ-liệu)
3. [Kế hoạch 4 giai đoạn](#3-kế-hoạch-4-giai-đoạn)
4. [Kết quả thực nghiệm](#4-kết-quả-thực-nghiệm)
5. [So sánh baseline & kết luận](#5-so-sánh-baseline--kết-luận)
6. [Phân tích lỗi & hướng cải thiện](#6-phân-tích-lỗi--hướng-cải-thiện)
7. [Hướng dẫn chạy nhanh](#7-hướng-dẫn-chạy-nhanh)

---

## 1. Bài toán & phương pháp

### 1.1. Đánh giá Retrieval trên kho cục bộ

Hệ thống tìm kiếm trên kho bài viết tiếng Việt (SKDS, UMC, bài scrape). **Không có benchmark public** map trực tiếp claim → `article_id` trong Qdrant, nên nhóm xây **Golden Dataset A (Viet-MedFact)** bằng phương pháp **sinh dữ liệu tự động + human-in-the-loop**:

1. Random 100–200 bài từ `articles_skds.csv` và `articles.csv`
2. Dùng LLM sinh 3 loại claim/bài: **ĐÚNG**, **SAI** (bóp méo), **KHÔNG ĐỦ BẰNG CHỨNG** (liên quan chủ đề nhưng không đủ căn cứ)
3. Lưu `article_id` ground truth cho mỗi claim
4. Đo **Hit@K** / **MRR**: bài gốc có trong top-K sau retrieve (và sau rerank) hay không

### 1.2. Hai bối cảnh đánh giá

| Bối cảnh | Dataset | Evidence | Ý nghĩa |
|----------|---------|----------|---------|
| **Oracle context** | B (ViFactCheck-Health) | `Context` vàng đầy đủ | Trần năng lực NLI / LLM explanation |
| **Thực tế pipeline** | A (Viet-MedFact) | Top chunk sau retrieve + rerank | Hiệu năng E2E người dùng thấy |

> **Lưu ý:** Module **Guard 3 tầng** (`guard/`) lọc input trước pipeline — dùng trong UI, **không** tính vào metric E2E/baseline (eval chạy trực tiếp claim hợp lệ trong dataset).

---

## 2. Tập dữ liệu

### Dataset A — Viet-MedFact

| Thuộc tính | Chi tiết |
|------------|----------|
| Nguồn | Sinh từ bài SKDS + UMC đã index Qdrant |
| Kích thước | ~300 claim (100 bài × 3 claim/bài trong eval chính) |
| Nhãn | `true` / `false` / `unproven` → ĐÚNG / SAI / KHÔNG ĐỦ BẰNG CHỨNG |
| Ground truth retrieval | `article_id` trong payload Qdrant |
| Script tạo | `evaluation/generate_viet_medfact.py` hoặc `create_offline_dataset.py` |
| File | `evaluation/data/viet_medfact.jsonl` |

### Dataset B — ViFactCheck-Health (mặc định)

| Thuộc tính | Chi tiết |
|------------|----------|
| Nguồn | [tranthaihoa/vifactcheck](https://huggingface.co/datasets/tranthaihoa/vifactcheck) |
| Lọc | Topic *Sức khỏe / Y tế* |
| Kích thước | ~438 mẫu tiếng Việt sẵn |
| Nhãn | Supported / Refuted / NEI → ĐÚNG / SAI / KHÔNG ĐỦ BẰNG CHỨNG |
| Evidence | Trường `Context` (bài báo gốc) |
| Script tạo | `evaluation/prepare_vifactcheck.py` |
| File | `evaluation/data/vifactcheck_health.jsonl` |

**Đánh giá RAG công bằng trên ViFactCheck:** index `Context` vào collection riêng `vifactcheck_health_bge_m3` (`evaluation/index_vifactcheck.py`) để baseline RAG+NLI / RAG+LLM retrieve trên cùng corpus với claim.

### Dataset B (tùy chọn) — PUBHEALTH-Vi

| Thuộc tính | Chi tiết |
|------------|----------|
| Nguồn | [ImperialCollegeLondon/health_fact](https://huggingface.co/datasets/ImperialCollegeLondon/health_fact) |
| Xử lý | Dịch tiếng Việt (~500 mẫu) |
| Script | `evaluation/prepare_pubhealth.py` |
| Dùng khi | Muốn so sánh thêm với chuẩn quốc tế |

---

## 3. Kế hoạch 4 giai đoạn

### Giai đoạn 1 — Chuẩn bị dữ liệu

- Sinh / tải Dataset A và B
- Re-index Qdrant cục bộ (`python ingest.py`) — bắt buộc có `article_id` trong payload
- (Tuỳ chọn) Index ViFactCheck cho baseline RAG công bằng

### Giai đoạn 2 — Đánh giá thành phần

| Module | Dataset | Script | Metrics |
|--------|---------|--------|---------|
| Retrieval & Rerank | A | `eval_retrieval.py` | Hit@1/5/10/20, MRR; ablation có/không reranker |
| NLI classification | B | `eval_nli.py` | Accuracy, F1-macro, Precision/Recall, Confusion Matrix |
| LLM explanation | B | `eval_llm.py` | ROUGE-L, BERTScore F1 |
| Fine-tune NLI | B | `finetune_nli.py` | Acc/F1 trên val set |

**Ablation Retrieval:** so sánh chỉ embedding `bge-m3` vs embedding + cross-encoder `bge-reranker-v2-m3`.

**NLI:** đưa trực tiếp `claim` + `Context` vào mDeBERTa, bỏ qua retrieval — đo trần năng lực suy luận.

**LLM explanation:** so sánh output `explain()` với gold `Evidence`/`Context`; ưu tiên BERTScore hơn ROUGE vì khác genre (giải thích vs trích dẫn).

### Giai đoạn 3 — End-to-End & Baseline

| Thí nghiệm | Dataset | Script | Mô tả |
|------------|---------|--------|-------|
| E2E RAG+NLI | A | `eval_e2e.py` | Retrieve → Rerank → NLI → verdict (không LLM) |
| Baseline | A hoặc B | `eval_baselines.py` | LLM-only / RAG+LLM / RAG+NLI / RAG+NLI+LLM |

**Baseline so sánh:**

- **LLM-only:** LLM phân loại không retrieval
- **RAG + LLM:** retrieve + rerank, LLM phân loại từ evidence
- **RAG + NLI (đề xuất):** retrieve + rerank, NLI quyết định verdict — **pipeline chính của hệ thống**
- **RAG + NLI + LLM:** thêm bước giải thích (cùng verdict với RAG+NLI)

Metrics E2E: **Accuracy**, **F1-macro**, **Latency** (mean / P50 / P95).

### Giai đoạn 4 — Phân tích lỗi

Script `evaluation/error_analysis.py` — trích case sai, phân loại nguyên nhân:

- DB thiếu thông tin → NEI đúng logic nhưng người dùng kỳ vọng ĐÚNG/SAI
- Retrieval miss (bài gốc ngoài top-20)
- NLI nhầm entailment/contradiction trên evidence nhiễu
- Phủ định kép / cấu trúc câu phức tạp
- LLM hallucination trong giải thích (verdict đúng nhưng giải thích bịa)

---

## 4. Kết quả thực nghiệm

*Nguồn: `evaluation/results/*.json` — chạy 30–31/07/2026, n=100 mẫu/module (trừ khi ghi chú khác).*

### 4.1. Retrieval & Reranking (Dataset A)

| Cấu hình | Hit@1 | Hit@5 | Hit@20 | MRR |
|----------|------:|------:|-------:|----:|
| Embedding only | 0.47 | 0.70 | 0.84 | 0.575 |
| **Embedding + Reranker** | **0.63** | **0.80** | 0.84 | **0.703** |

- Reranker cải thiện rõ top nhỏ: Hit@1 **+16 điểm**, Hit@5 **+10 điểm**, MRR **+0.128**
- Hit@20 không đổi → rerank chỉ sắp xếp lại trong tập đã retrieve
- **16/100** mẫu không có bài đúng trong top-20

### 4.2. NLI trên ViFactCheck-Health (Dataset B, oracle context)

| Metric | Giá trị |
|--------|--------:|
| Accuracy | **0.660** |
| F1-macro | **0.654** |
| Strategy | `weighted_max` |

| Nhãn | Precision | Recall | F1 |
|------|----------:|-------:|---:|
| ĐÚNG | 0.74 | **0.38** | 0.50 |
| SAI | 0.74 | 0.82 | **0.78** |
| KHÔNG ĐỦ BẰNG CHỨNG | 0.58 | 0.83 | 0.68 |

→ Model **thiên NEI**; lỗi phổ biến: ĐÚNG bị gán NEI (17/34 lỗi).

### 4.3. LLM explanation (Dataset B)

| Metric | Giá trị |
|--------|--------:|
| ROUGE-L | 0.328 |
| BERTScore F1 | **0.727** |
| Mẫu rỗng | 0/100 |

Dùng `qwen3:8b` (Ollama). ROUGE thấp hơn BERTScore vì reference là đoạn evidence gốc, không phải câu giải thích.

### 4.4. End-to-End RAG+NLI (Dataset A)

| Metric | Giá trị |
|--------|--------:|
| Accuracy | **0.500** |
| F1-macro | 0.440 |
| Latency mean / p50 / p95 | 10.6s / 8.3s / 24.8s |

| Nhãn | Recall |
|------|-------:|
| ĐÚNG | 0.68 |
| SAI | 0.74 |
| KHÔNG ĐỦ BẰNG CHỨNG | **0.09** |

→ E2E **thiên ĐÚNG/SAI**, gần như không dự đoán NEI — ngược chiều với NLI oracle context.

### 4.5. Baseline trên ViFactCheck (RAG công bằng, n=100)

Index `Context` vào `vifactcheck_health_bge_m3`:

| Phương pháp | Accuracy | F1-macro | Latency TB |
|-------------|---------:|---------:|-----------:|
| LLM-only | 0.44 | 0.39 | **1.3s** |
| RAG + LLM | 0.33 | 0.22 | 48.6s |
| **RAG + NLI** | **0.49** | **0.47** | 10.0s |
| RAG + NLI + LLM | 0.49 | 0.47 | 59.6s |

→ Trên ViFactCheck, **RAG+NLI tốt nhất**; RAG+LLM kém nhất (bias ĐÚNG mạnh).

---

## 5. So sánh baseline & kết luận

### 5.1. Dataset A — Viet-MedFact (n=100)

| Phương pháp | Accuracy | F1-macro | Latency TB |
|-------------|---------:|---------:|-----------:|
| LLM-only | **0.51** | **0.495** | **1.2s** |
| RAG + LLM | 0.44 | 0.318 | 47.2s |
| **RAG + NLI (đề xuất)** | 0.50 | 0.440 | 9.3s |

### 5.2. Nhận định tổng hợp

| Khía cạnh | Kết luận |
|-----------|----------|
| Retrieval + Rerank | Hoạt động tốt (Hit@5 = 0.80); rerank có đóng góp đo được |
| NLI (context vàng) | Acc 0.66 — chấp nhận được; yếu recall ĐÚNG |
| E2E thực tế | Acc 0.50; NEI gần như fail (recall 0.09) |
| LLM-only | Acc cao trên A nhưng **không có bằng chứng** → không phù hợp triển khai |
| RAG + LLM | Bias ĐÚNG nghiêm trọng, chậm → **không dùng làm classifier** |
| **RAG + NLI** | Cân bằng nhãn hơn, có evidence, latency chấp nhận → **pipeline đề xuất** |

**Kết luận triển khai:** Hệ thống dùng **RAG (bge-m3) + Reranker + NLI (mDeBERTa)** cho verdict; **LLM chỉ sinh giải thích**, không quyết định ĐÚNG/SAI.

---

## 6. Phân tích lỗi & hướng cải thiện

### Điểm mạnh

1. Reranker nâng Hit@5 từ 0.70 → 0.80
2. NLI tiếng Việt trên ViFactCheck đạt Acc 0.66 không cần dịch
3. LLM explanation ổn định với `qwen3:8b` (BERTScore F1 0.73)
4. RAG+NLI bắt SAI tốt hơn rõ so với RAG+LLM

### Điểm yếu

1. E2E Acc 0.50; NEI recall 0.09 trên Dataset A
2. NLI oracle thiên NEI; recall ĐÚNG thấp (0.38)
3. 16% claim không retrieve đúng bài trong top-20
4. Gap NLI oracle (0.66) vs E2E (0.50) → evidence retrieve nhiễu là nút thắt chính

### Hướng cải thiện

1. **NLI:** fine-tune trên ViFactCheck-Health; thử aggregate / threshold khác
2. **E2E NEI:** abstain khi confidence thấp; thêm relevance score trước NLI
3. **Retrieval:** hybrid BM25 + dense; mở rộng corpus (scraper)
4. **LLM:** giữ vai trò giải thích, không làm verdict chính
5. **Đánh giá:** luôn nêu rõ Dataset A vs B khi báo cáo số liệu

---

## 7. Hướng dẫn chạy nhanh

```bash
# Chuẩn bị
python ingest.py
python -m evaluation.prepare_vifactcheck
python -m evaluation.generate_viet_medfact --num-articles 100

# Đánh giá từng phần
python -m evaluation.eval_retrieval
python -m evaluation.eval_nli
python -m evaluation.eval_e2e
python -m evaluation.eval_baselines --max-samples 100

# Baseline công bằng ViFactCheck (cần index trước)
python -m evaluation.index_vifactcheck --recreate
python -m evaluation.eval_baselines --dataset vifactcheck --max-samples 100

# Toàn bộ pipeline
python -m evaluation.run_all

# Thử nhanh offline (không API, không Dataset B)
python -m evaluation.run_all --offline --skip-dataset-b --max-samples 20
```

**Kết quả JSON/PNG:** `evaluation/results/`  
**Báo cáo chi tiết:** [`evaluation/ket_qua_danh_gia_day_du.md`](evaluation/ket_qua_danh_gia_day_du.md)
