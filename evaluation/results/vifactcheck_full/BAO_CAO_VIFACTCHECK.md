# Báo cáo đánh giá trên ViFactCheck-Health (full 438 mẫu)

**Ngày chạy:** 19/10/2026, 18:29–22:49 (giờ GMT+7) · **Trạng thái:** đã xong toàn bộ 15/15 bước.
**Phạm vi:** chỉ đánh giá trên ViFactCheck-Health. Không bao gồm Viet-MedFact, Guard, PUBHEALTH.

---

## 1. Thiết lập

| Mục | Giá trị |
|-----|---------|
| Dataset | `evaluation/data/vifactcheck_health.jsonl` — 438 claim, 62 bài báo (topic Sức khỏe / Y tế của [tranthaihoa/vifactcheck](https://huggingface.co/datasets/tranthaihoa/vifactcheck)) |
| Phân bố nhãn | ĐÚNG 148 · SAI 146 · KHÔNG ĐỦ BẰNG CHỨNG (NEI) 144 |
| Corpus RAG | Collection riêng `vifactcheck_health_bge_m3` — chỉ chứa `main_text` của 62 bài (`index_vifactcheck.py`) |
| Embedding / Reranker | `BAAI/bge-m3` / `BAAI/bge-reranker-v2-m3` (GPU) |
| NLI | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (không fine-tune) |
| LLM | Ollama `qwen3:8b`, tắt thinking |
| Verdict pipeline | `VERDICT_STRATEGY="ess"`, `ESS_TOP_M=5`, `TOP_K=20`, `TOP_RERANK=3` |
| Phần cứng | NVIDIA RTX 4000 Ada 20 GB, container 5 CPU |
| Chia tập ablation | test = 100 mẫu đầu, dev = 338 mẫu còn lại (chỉ dùng chọn tham số); CV 5-fold × 5 lần trên toàn bộ 438 |

Hai bối cảnh đánh giá:
- **Oracle:** đưa thẳng `main_text` (bài báo gốc chứa đáp án) vào NLI, không qua retrieval → trần năng lực của NLI.
- **E2E:** retrieve → rerank → NLI trên corpus 62 bài → hiệu năng thực tế của pipeline.

---

## 2. Tóm tắt số liệu chính (n=438)

| Thực nghiệm | Metric chính | Giá trị |
|-------------|--------------|--------:|
| Retrieval + rerank | Hit@1 / Hit@5 / MRR | **0.943 / 0.991 / 0.964** |
| NLI oracle (`weighted_max`) | Acc / F1-macro | **0.616 / 0.608** |
| **RAG + NLI (ESS) — pipeline triển khai** | Acc / F1-macro / latency | **0.571 / 0.576 / 1.10 s** |
| RAG + NLI + LLM (kèm giải thích) | Acc / F1-macro / latency | 0.571 / 0.576 / 4.85 s |
| Baseline RAG + LLM | Acc / F1-macro / latency | 0.434 / 0.348 / 1.26 s |
| Baseline LLM-only | Acc / F1-macro / latency | 0.372 / 0.317 / 0.47 s |
| LLM explanation | ROUGE-L / BERTScore F1 | 0.321 / 0.723 |

→ **RAG + NLI tốt nhất**, hơn RAG+LLM **+0.23 F1** và LLM-only **+0.26 F1**. Thêm bước giải thích bằng LLM không đổi nhãn, chỉ tăng latency từ 1.1 s lên 4.85 s.

---

## 3. Retrieval & Reranking

| Chế độ | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR |
|--------|------:|------:|------:|-------:|-------:|----:|
| Dense (bge-m3) | 0.863 | 0.945 | 0.970 | 0.984 | 0.991 | 0.910 |
| + Rerank | **0.943** | **0.986** | **0.991** | 0.991 | 0.991 | **0.964** |

- Rerank tăng Hit@1 thêm +0.08 và MRR thêm +0.054. Hit@20 không đổi vì rerank chỉ sắp xếp lại trong tập đã retrieve.
- Chỉ 4/438 claim không có bài đúng trong top-5 sau rerank.
- **Lưu ý:** corpus chỉ có 62 bài nên bài toán retrieval dễ hơn nhiều so với corpus chính SKDS+UMC (trên Viet-MedFact, Hit@5 là 0.80). Không nên dùng số này để kết luận về retrieval trên corpus lớn.

---

## 4. NLI trên context vàng (oracle)

Cấu hình `eval_nli`: tách `main_text` thành chunk 800 ký tự, chạy NLI từng chunk, gộp bằng `weighted_max`.

| Nhãn | Precision | Recall | F1 | Support |
|------|----------:|-------:|---:|--------:|
| ĐÚNG | 0.637 | **0.392** | 0.485 | 148 |
| SAI | 0.709 | 0.719 | **0.714** | 146 |
| NEI | 0.538 | 0.743 | 0.624 | 144 |
| **Macro** | 0.628 | 0.618 | **0.608** | 438 |

**Ma trận nhầm lẫn** (hàng = nhãn đúng, cột = dự đoán):

|  | ĐÚNG | SAI | NEI |
|--|-----:|----:|----:|
| **ĐÚNG** | 58 | 27 | **63** |
| **SAI** | 12 | 105 | 29 |
| **NEI** | 21 | 16 | 107 |

**Phân bố lỗi** (168 lỗi, phân loại theo ô của ma trận nhầm lẫn — `error_analysis.py`):

| Loại | Số lỗi |
|------|-------:|
| ĐÚNG → NEI | 62 |
| SAI ↔ ĐÚNG (nhầm hướng) | 38 |
| NEI → ĐÚNG/SAI (quá tự tin) | 36 |
| SAI → NEI | 29 |
| Câu phủ định kép | 3 |

**Nhận định:**
- Mô hình **thiên về NEI**: 199/438 dự đoán là NEI, trong khi chỉ có 144 mẫu NEI thật.
- Điểm nghẽn là **recall của nhãn ĐÚNG (0.39)**: tín hiệu entailment bị pha loãng khi context dài có nhiều câu không liên quan.
- SAI là nhãn tốt nhất.

---

## 5. So sánh baseline (E2E, n=438)

| Phương pháp | Accuracy | F1-macro | Latency TB | P95 | Phân bố dự đoán (Đ / S / NEI) |
|-------------|---------:|---------:|-----------:|----:|------------------------------:|
| LLM-only | 0.372 | 0.317 | 0.47 s | 0.72 s | 60 / 28 / **350** |
| RAG + LLM | 0.434 | 0.348 | 1.26 s | 1.50 s | **370** / 25 / 43 |
| **RAG + NLI (ESS)** | **0.571** | **0.576** | 1.10 s | 1.23 s | 121 / 110 / 207 |
| RAG + NLI + LLM | 0.571 | 0.576 | 4.85 s | 6.20 s | 121 / 110 / 207 (trùng 438/438 nhãn với RAG+NLI) |

**F1 theo nhãn:**

| Phương pháp | ĐÚNG | SAI | NEI |
|-------------|-----:|----:|----:|
| LLM-only | 0.308 | 0.172 | 0.470 |
| RAG + LLM | 0.571 | 0.257 | 0.214 |
| **RAG + NLI** | 0.498 | **0.688** | **0.541** |

**Ma trận nhầm lẫn RAG + NLI** (hàng = nhãn đúng):

|  | ĐÚNG | SAI | NEI |
|--|-----:|----:|----:|
| **ĐÚNG** | 67 | 7 | **74** |
| **SAI** | 20 | 88 | 38 |
| **NEI** | 34 | 15 | 95 |

**Nhận định:**
- **Hai baseline LLM bị lệch về hai phía ngược nhau:**
  - LLM-only gán NEI cho 80% claim (350/438) và gần như không nhận ra SAI (recall SAI chỉ 0.10).
  - RAG+LLM gán ĐÚNG cho 84% claim (370/438): đạt recall ĐÚNG = 1.00, nhưng recall SAI chỉ 0.15 và recall NEI 0.14. Nghĩa là khi thấy bằng chứng cùng chủ đề, LLM coi đó là bằng chứng ủng hộ claim.
- **RAG + NLI cân bằng nhất giữa 3 nhãn.**
  - SAI có precision cao: 0.80.
  - Điểm yếu vẫn là ĐÚNG → NEI: 74/148 claim ĐÚNG bị đoán là NEI.
- Trên GPU, latency của RAG+NLI (1.1 s) tương đương RAG+LLM; LLM-only nhanh nhất.
- RAG+NLI+LLM cho nhãn trùng 100% với RAG+NLI (nhãn do NLI quyết định). Phần giải thích LLM tốn thêm khoảng 3.75 s/claim; không có giải thích nào bị rỗng.

---

## 6. Ablation bộ ra kết luận (ESS)

### 6.1. Kết quả trên tập test (100 mẫu đầu) và CV 5×5 (438 mẫu)

| Biến thể | Oracle F1 test | Oracle F1 CV | E2E Acc test | E2E F1 test | E2E F1 CV |
|----------|---------------:|-------------:|-------------:|------------:|----------:|
| **FULL** — rerank + NLI + ESS (đề xuất, m=5) | 0.674 | 0.628 | 0.600 | **0.606** | 0.573 |
| B0 — bỏ ESS (RAG+NLI gốc) | 0.654 | 0.608 | 0.560 | 0.547 | 0.526 |
| Bỏ rerank chunk (dense top-3) | **0.685** | **0.637** | 0.610 | **0.610** | 0.561 |
| Bỏ ESS + bỏ rerank | 0.654 | 0.608 | 0.560 | 0.542 | 0.524 |
| + Hiệu chỉnh ngưỡng (+CALIB) | 0.629 | 0.618 | 0.590 | 0.597 | **0.596** |
| + Cổng liên quan (+GATE) | 0.674 | 0.627 | 0.600 | 0.606 | 0.571 |
| + CALIB + GATE | 0.629 | 0.616 | 0.590 | 0.592 | 0.588 |
| Tổng hợp trung bình (MEAN) | 0.512 | 0.463 | 0.480 | 0.440 | 0.468 |
| *LLM-only (run mới, 100 mẫu test)* | — | — | 0.440 | 0.392 | — |
| *RAG+LLM (run mới, 100 mẫu test)* | — | — | 0.450 | 0.337 | — |

Khoảng tin cậy 95% (bootstrap) cho F1 của FULL trên test: oracle [0.58, 0.76], e2e [0.505, 0.694].

### 6.2. Kiểm định FULL so với các biến thể (test, E2E)

| So sánh | ΔF1 CI95 | p (bootstrap F1) | McNemar (chỉ A đúng / chỉ B đúng, p) |
|---------|---------|-----------------:|------------------------------------:|
| FULL vs B0 (bỏ ESS) | [−0.005, +0.128] | **0.034** | 6 / 2, p = 0.29 |
| FULL vs bỏ rerank | [−0.076, +0.074] | 0.52 | 6 / 7, p = 1.0 |
| FULL vs +CALIB | [−0.074, +0.097] | 0.41 | 13 / 12, p = 1.0 |
| FULL vs MEAN | [+0.068, +0.268] | < 0.001 | 17 / 5, p = 0.017 |
| FULL vs LLM-only | [+0.089, +0.330] | < 0.001 | 28 / 12, **p = 0.017** |
| FULL vs RAG+LLM | [+0.142, +0.397] | < 0.001 | 37 / 22, p = 0.067 |
| B0 vs LLM-only | [+0.038, +0.273] | 0.005 | 24 / 12, p = 0.065 |
| B0 vs RAG+LLM | [+0.067, +0.340] | 0.003 | 39 / 28, p = 0.22 |

Hai dòng LLM dùng kết quả run full mới (`baseline_comparison_vifactcheck.json`), lấy trên đúng 100 mẫu test.

Ở bối cảnh **oracle**, FULL vs B0 có ΔF1 CI [−0.047, +0.084] với p = 0.29, tức là không có ý nghĩa thống kê.

### 6.3. Độ nhạy theo số câu bằng chứng m (test)

| m | 1 | 2 | 3 | **5 (chọn trên dev)** | tất cả |
|---|--:|--:|--:|--:|--:|
| Oracle F1 | 0.658 | 0.658 | 0.667 | 0.674 | 0.685 |
| E2E F1 | 0.614 | 0.614 | 0.589 | 0.606 | 0.564 |

### 6.4. Nhận định

1. **ESS có ích rõ ở E2E:** F1 tăng +0.059 trên test (bootstrap p = 0.034), tăng +0.047 khi CV và thắng cả 5/5 lần lặp. Tuy vậy kiểm định McNemar chưa có ý nghĩa (p = 0.29), vì chỉ có 8 claim mà hai biến thể dự đoán khác nhau. Khi viết báo cáo nên diễn đạt thận trọng: *"cải thiện nhất quán nhưng biên độ nhỏ"*.
2. **Ở oracle, mức tăng nhỏ** (+0.02 F1, không có ý nghĩa thống kê). ESS hữu ích chủ yếu khi bằng chứng là các chunk nhiễu sau retrieval.
3. **Rerank chunk không giúp verdict trên tập này.** Bỏ rerank cho kết quả bằng hoặc hơn ở cả hai bối cảnh, vì corpus chỉ có 62 bài nên bài đúng gần như luôn có sẵn trong dense top-3 (Hit@3 dense = 0.945). Rerank vẫn có ích cho retrieval (mục 3).
4. **+CALIB đạt F1 CV cao nhất ở E2E (0.596)**, nhưng trên test không hơn FULL (p = 0.41). Kết luận trước đây "+CALIB overfit trên dev" được rút ra trên Viet-MedFact; trên ViFactCheck thì chưa thấy overfit. Đây là kết quả phụ thuộc dataset, nên được ghi nhận trong báo cáo.
5. **MEAN kém rõ rệt** (p < 0.001), xác nhận lựa chọn `weighted_max` / max-aggregation là đúng.
6. **FULL hơn hẳn hai baseline LLM về F1** (bootstrap p < 0.001 cho cả hai). Theo McNemar thì chỉ chênh lệch với LLM-only là có ý nghĩa (p = 0.017); với RAG+LLM thì p = 0.067. Lý do là RAG+LLM đúng nhiều ở nhãn ĐÚNG (do đoán ĐÚNG gần như toàn bộ), nên số cặp bất đồng giữa hai bên khá cân.

---

## 6b. LLM explanation (n=438)

Cấu hình `eval_llm`: verdict lấy từ một lần NLI trên context vàng (cắt còn 1500 ký tự); `explain()` dùng qwen3:8b sinh giải thích; so với trường `explanation` của dataset.

| Metric | Giá trị | Cũ (n=100) |
|--------|--------:|-----------:|
| ROUGE-L | 0.321 | 0.328 |
| BERTScore P / R / F1 | 0.693 / 0.758 / **0.723** | 0.694 / 0.765 / 0.727 |
| Giải thích bị rỗng | 0/438 | 0/100 |
| Độ dài TB (sinh / tham chiếu) | 390 / 183 ký tự | 406 / 184 |

- Kết quả ổn định giữa n=100 và n=438 (chênh lệch ≤ 0.007), nên đây là module ít biến động nhất.
- ROUGE-L thấp hơn BERTScore vì giải thích được sinh ra dài gấp khoảng 2 lần tham chiếu và khác văn phong. BERTScore phản ánh độ tương đồng ngữ nghĩa tốt hơn.
- Verdict đưa vào bước giải thích lệch về NEI (244/438), do lấy từ NLI một lần trên context dài. Nhãn này khác với nhãn của pipeline ESS, nên metric ở đây đánh giá **chất lượng văn bản giải thích**, không phải độ đúng của kết luận.

---

## 7. Độ ổn định giữa 100 mẫu đầu và 338 mẫu còn lại

| Phương pháp | F1 trên 100 mẫu đầu | F1 trên 338 mẫu còn lại | F1 toàn bộ 438 |
|-------------|--------------:|-----------------:|----------:|
| RAG + NLI (ESS) | 0.606 | 0.566 | 0.576 |
| LLM-only | 0.392 | 0.295 | 0.317 |
| RAG + LLM | 0.337 | 0.349 | 0.348 |
| NLI oracle | 0.654 (run cũ) | — | 0.608 |

→ **100 mẫu đầu dễ hơn phần còn lại.** Vì vậy các số trước đây đo trên n=100 đánh giá hơi cao. Nên dùng số trên 438 mẫu làm số chính, còn tập test 100 mẫu chỉ dùng cho ablation (vì dev/test đã tách sẵn).

---

## 8. So sánh với kết quả cũ (n=100, máy khác)

| Thực nghiệm | Cũ (n=100) | Mới (n=438) | Ghi chú |
|-------------|-----------:|------------:|---------|
| NLI oracle Acc / F1 | 0.66 / 0.654 | 0.616 / 0.608 | Cùng cấu hình; khác biệt do 100 mẫu đầu dễ hơn |
| RAG+NLI Acc / F1 | 0.49 / 0.47 | **0.571 / 0.576** | Bản cũ dùng `weighted_max`, bản mới dùng ESS |
| LLM-only Acc / F1 | 0.44 / 0.39 | 0.372 / 0.317 | Trên 100 mẫu đầu, run mới cho đúng 0.44 / 0.392 |
| RAG+LLM Acc / F1 | 0.33 / 0.22 | 0.434 / 0.348 | Run mới thiên mạnh về ĐÚNG; run cũ (theo cache ablation) thiên về NEI — hành vi LLM không ổn định giữa các lần chạy |
| Latency RAG+NLI | ~10 s (Apple M2, CPU) | 1.10 s (RTX 4000 Ada) | Không so sánh trực tiếp được — khác phần cứng |

---

## 9. Thời gian chạy

| Phase | Bước | Tổng thời gian |
|-------|------|---------------:|
| Không cần LLM | NLI, retrieval, RAG+NLI, ablation cache + phân tích | ~35 phút |
| Cần LLM | LLM-only, RAG+LLM, RAG+NLI+LLM, LLM explanation, gộp, ablation e2e | ~80 phút |

Chi tiết từng bước nằm trong `PROGRESS.md`.

---

## 10. Hạn chế

- **Corpus RAG nhỏ (62 bài):** retrieval gần như bão hòa, nên kết quả E2E ở đây gần với oracle hơn so với thực tế triển khai trên corpus lớn.
- **NLI chưa fine-tune:** mDeBERTa gốc thiên về NEI và recall ĐÚNG thấp. Fine-tune trên ViFactCheck (`finetune_nli.py`) là hướng cải thiện rõ nhất, nhưng phải tách train/test cẩn thận để tránh rò rỉ dữ liệu.
- **Tập test ablation chỉ có 100 mẫu:** CI rộng (±0.09 F1), nên phần lớn các khác biệt giữa biến thể không có ý nghĩa thống kê. CV 5×5 trên 438 mẫu bổ sung độ tin cậy cho kết luận.
- **Baseline LLM phụ thuộc vào lần chạy/phiên bản model:** hành vi RAG+LLM thay đổi mạnh giữa run cũ và run mới.

---

## 11. File kết quả

Tất cả nằm trong `evaluation/results/vifactcheck_full/`:

| File | Nội dung |
|------|----------|
| `nli_results.json`, `nli_results_report.txt`, `nli_confusion_matrix.png` | NLI oracle |
| `error_analysis_nli.json` | Phân loại lỗi NLI |
| `retrieval_results.json` | Hit@K, MRR |
| `baseline_{llm_only,rag_llm,rag_nli,rag_nli_llm}.json` | Baseline (chi tiết từng claim) |
| `baseline_comparison_vifactcheck.json` | Gộp 4 baseline |
| `llm_results.json` | LLM explanation (ROUGE-L, BERTScore, từng giải thích) |
| `ablation/cache_vifactcheck_{oracle,e2e}.jsonl` | Cache điểm số NLI / rerank |
| `ablation/summary_vifactcheck_{oracle,e2e}.json` | Kết quả ablation, CI, kiểm định, CV |
| `PROGRESS.md`, `logs/` | Tiến độ và log từng bước |

Tái lập: `bash scripts/run_vifactcheck_full.sh nollm` rồi `bash scripts/run_vifactcheck_full.sh llm`.
