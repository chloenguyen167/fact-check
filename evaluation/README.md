# Hướng dẫn Đánh giá Hệ thống

Tài liệu này bổ sung cho `evaluation.md`, mô tả cách chạy pipeline đánh giá đã triển khai.

## Tổng quan

| Tập dữ liệu | Script tạo | Mục đích |
|-------------|-----------|----------|
| **Viet-MedFact** (Dataset A) | `generate_viet_medfact.py` | Retrieval, Rerank, E2E |
| **ViFactCheck-Health** (Dataset B, mặc định) | `prepare_vifactcheck.py` | NLI, LLM explanation |
| **PUBHEALTH-Vi** (Dataset B, tùy chọn) | `prepare_pubhealth.py` | NLI, LLM explanation |

- **ViFactCheck-Health**: lọc topic *Sức khỏe / Y tế* từ [tranthaihoa/vifactcheck](https://huggingface.co/datasets/tranthaihoa/vifactcheck) — tiếng Việt sẵn, **không cần dịch** (~438 mẫu, nhãn cân bằng).
- **PUBHEALTH-Vi**: [ImperialCollegeLondon/health_fact](https://huggingface.co/datasets/ImperialCollegeLondon/health_fact) dịch Việt (cần LLM).

## Chuẩn bị

### 1. Cài dependencies

```bash
pip install -r requirements.txt
```

### 2. Re-index Qdrant (bắt buộc cho đánh giá Retrieval)

Vector DB cần lưu `article_id` trong payload. Chạy lại:

```bash
python ingest.py
```

### 3. Cấu hình API

Đặt `OPENROUTER_API_KEY` trong `config.py` hoặc biến môi trường (dùng cho sinh claim, dịch PUBHEALTH nếu dùng, đánh giá LLM).

## Chạy từng bước

### Dataset A — Viet-MedFact

```bash
# Cách 1: Sinh claim bằng LLM (khuyến nghị cho luận văn)
python -m evaluation.generate_viet_medfact --num-articles 100

# Cách 2: Tạo nhanh offline (thử pipeline, không cần API)
python -m evaluation.create_offline_dataset --num-articles 30
```

### Dataset B — ViFactCheck-Health (khuyến nghị)

```bash
# Tải HF + lọc topic sức khỏe / y tế (~438 mẫu)
python -m evaluation.prepare_vifactcheck

# Index Context vào RAG (collection riêng, 62 bài unique)
# Lưu ý: tắt streamlit/app trước khi index (Qdrant local 1 process)
python -m evaluation.index_vifactcheck --recreate

# Baseline công bằng trên ViFactCheck (cùng nhãn + evidence đã index)
python -m evaluation.eval_baselines --dataset vifactcheck --max-samples 100

# Chỉ RAG methods (bỏ LLM giải thích cho nhanh)
python -m evaluation.eval_baselines --dataset vifactcheck --methods llm_only rag_llm rag_nli --max-samples 100

# Hit@K retrieval trên ViFactCheck
python -m evaluation.eval_retrieval --dataset vifactcheck --max-samples 100
```

Kết quả baseline ViFactCheck: `evaluation/results/baseline_comparison_vifactcheck.json`

### Dataset B — PUBHEALTH-Vi (tùy chọn, cần dịch)

```bash
python -m evaluation.prepare_pubhealth --sample-size 500
python -m evaluation.prepare_pubhealth --skip-translate
```

### Đánh giá thành phần

```bash
# Retrieval & Reranking (Hit@K, MRR, ablation)
python -m evaluation.eval_retrieval

# NLI trên ViFactCheck-Health (mặc định)
python -m evaluation.eval_nli

# NLI trên PUBHEALTH-Vi
python -m evaluation.eval_nli --dataset-b pubhealth

# LLM explanation
python -m evaluation.eval_llm --max-samples 20
```

### Đánh giá toàn trình & phân tích lỗi

```bash
python -m evaluation.eval_e2e
python -m evaluation.error_analysis
python -m evaluation.plot_confusion_matrix
```

### Ablation cải tiến ESS (dev/test tách biệt)

```bash
# Cache một lần (nặng): retrieval + rerank + NLI mức chunk và mức câu
python -m evaluation.ablation_cache --dataset viet_medfact --mode e2e --max-samples 250
python -m evaluation.ablation_cache --dataset vifactcheck --mode oracle --max-samples 250
python -m evaluation.ablation_cache --dataset vifactcheck --mode e2e --max-samples 250

# So sánh các thiết kế luật quyết định CHỈ trên dev (idx >= 100)
python -m evaluation.ablation_design_dev --dataset viet_medfact --mode e2e

# Biến thể trên test (100 mẫu đầu) + bootstrap CI + McNemar + 5×5-fold CV
python -m evaluation.ablation_analyze --dataset viet_medfact --mode e2e
python -m evaluation.ablation_analyze --dataset vifactcheck --mode oracle
python -m evaluation.ablation_analyze --dataset vifactcheck --mode e2e

# Guard theo tầng (882 claim hợp lệ + 60 input không hợp lệ)
python -m evaluation.eval_guard
```

## Chạy toàn bộ

```bash
# Đầy đủ với ViFactCheck-Health (không dịch)
python -m evaluation.run_all

# Thử nhanh offline (chỉ Viet-MedFact + retrieval/e2e)
python -m evaluation.run_all --offline --skip-dataset-b --max-samples 20

# Dùng PUBHEALTH-Vi thay vì ViFactCheck
python -m evaluation.run_all --dataset-b pubhealth

# Bỏ đánh giá LLM explanation
python -m evaluation.run_all --skip-llm-eval
```

## Kết quả

Tất cả output nằm trong `evaluation/results/`:

| File | Nội dung |
|------|----------|
| `retrieval_results.json` | Hit@5/10/20, MRR, ablation reranker |
| `nli_results.json` | Accuracy, F1, confusion matrix |
| `nli_confusion_matrix.png` | Biểu đồ ma trận nhầm lẫn |
| `llm_results.json` | ROUGE-L, BERTScore |
| `e2e_results.json` | E2E accuracy, latency |
| `error_analysis.json` | Phân loại lỗi + ví dụ |

## Metrics

### Retrieval (Dataset A)

- **Hit@K**: Bài viết gốc (`article_id`) có trong top K không
- **MRR**: Vị trí trung bình của bài đúng
- **Ablation**: So sánh `retrieval_only` vs `with_reranker`

### NLI (Dataset B)

Ánh xạ nhãn → hệ thống:

| Nguồn | Nhãn gốc | Hệ thống |
|-------|----------|----------|
| ViFactCheck | 0 Supported / 1 Refuted / 2 NEI | ĐÚNG / SAI / KHÔNG ĐỦ BẰNG CHỨNG |
| PUBHEALTH | true / false / unproven,mixture | ĐÚNG / SAI / KHÔNG ĐỦ BẰNG CHỨNG |

File jsonl dùng chung schema `label ∈ {true,false,unproven}` nên `eval_nli` / `eval_llm` dùng chung.

### E2E (Dataset A)

- **Accuracy**: Kết luận cuối khớp nhãn gốc
- **Latency**: Thời gian trung bình / P50 / P95 (giây)

## Cấu hình

Chỉnh trong `evaluation/config_eval.py`:

- `DATASET_B_DEFAULT` — `"vifactcheck"` hoặc `"pubhealth"`
- `VIET_MEDFACT_NUM_ARTICLES`, `PUBHEALTH_SAMPLE_SIZE`
- `VIFACTCHECK_HEALTH_TOPICS` — topic được giữ lại
- `RETRIEVAL_K_VALUES`, `RERANK_K_VALUES`
- `EVAL_MAX_SAMPLES` — giới hạn mẫu khi chạy thử
