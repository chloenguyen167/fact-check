# Báo cáo kết quả đánh giá hệ thống Fact-Check Y tế

**Ngày tổng hợp:** 31/07/2026  
**Cấu hình chạy:** 100 mẫu / module  
**Model chính:** Embedding `BAAI/bge-m3`, Reranker `bge-reranker-v2-m3`, NLI `mDeBERTa-v3-base-xnli`, LLM local `qwen3:8b` (Ollama)

---

## 1. Nguồn kết quả

| File | Module | Dataset | Thời điểm |
|------|--------|---------|-----------|
| `evaluation/results/retrieval_results.json` | Retrieval + Rerank | Viet-MedFact (A) | 30/07/2026 |
| `evaluation/results/nli_results.json` | NLI classification | ViFactCheck-Health (B) | 30/07/2026 |
| `evaluation/results/llm_results.json` | LLM explanation | ViFactCheck-Health (B) | 30/07/2026 |
| `evaluation/results/e2e_results.json` | End-to-End (RAG+NLI) | Viet-MedFact (A) | 30/07/2026 |
| `evaluation/results/baseline_comparison.json` | LLM-only / RAG+LLM / RAG+NLI | Viet-MedFact (A) | 31/07/2026 |

**Ghi chú dataset**

- **Dataset A (Viet-MedFact):** claim gắn `article_id` trong Qdrant cục bộ → dùng cho Retrieval, E2E, Baseline.
- **Dataset B (ViFactCheck-Health):** 438 mẫu topic Sức khỏe/Y tế từ [tranthaihoa/vifactcheck](https://huggingface.co/datasets/tranthaihoa/vifactcheck); đánh giá lấy 100 mẫu đầu → dùng cho NLI và LLM explanation (claim + `Context`/`Evidence` vàng, không qua retrieval).

---

## 2. Tóm tắt số liệu chính

| Giai đoạn | Metric chính | Giá trị | Đánh giá ngắn |
|-----------|--------------|--------:|---------------|
| Retrieval (+rerank) | Hit@5 / MRR | **0.80 / 0.703** | Tốt; rerank có ích rõ |
| NLI (ViFactCheck) | Acc / F1-macro | **0.66 / 0.654** | Khá; yếu recall ĐÚNG |
| LLM explanation | ROUGE-L / BERTScore F1 | **0.328 / 0.727** | Dùng được sau khi chuyển qwen3:8b |
| E2E (RAG+NLI) | Acc / F1-macro | **0.50 / 0.440** | Trung bình; lệch ĐÚNG/SAI |
| Baseline LLM-only | Acc / F1-macro | **0.51 / 0.495** | Nhanh; Acc nhỉnh hơn RAG+NLI trên tập A |
| Baseline RAG+LLM | Acc / F1-macro | **0.44 / 0.318** | Kém nhất; thiên ĐÚNG mạnh |
| Baseline RAG+NLI | Acc / F1-macro / Latency | **0.50 / 0.440 / 9.3s** | Cân bằng hơn RAG+LLM; cùng pipeline E2E |

---

## 3. Retrieval & Reranking (Dataset A, n=100)

### 3.1. Bảng metric

| Cấu hình | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR |
|----------|------:|------:|------:|-------:|-------:|----:|
| Embedding only (`bge-m3`) | 0.47 | 0.63 | 0.70 | 0.77 | 0.84 | 0.575 |
| **Embedding + Reranker** | **0.63** | **0.76** | **0.80** | **0.82** | 0.84 | **0.703** |

### 3.2. Phân tích

- Reranker cải thiện rõ ở top nhỏ: **Hit@1 +16 điểm**, **Hit@5 +10 điểm**, **MRR +0.128**.
- Hit@20 không đổi (0.84) → rerank chỉ sắp xếp lại trong tập đã retrieve, không “gỡ” bài ngoài top-20.
- **16/100** mẫu không có bài đúng trong top-20 (lỗi retrieval cứng).
- Khi tìm được: `retrieval_rank` trung vị = 1; sau rerank trung vị vẫn = 1, trung bình giảm 3.26 → 1.86.

**Kết luận thành phần:** Kiến trúc Retrieval + Cross-encoder Reranker là hợp lý và có đóng góp đo được.

---

## 4. NLI trên ViFactCheck-Health (Dataset B, n=100)

### 4.1. Tổng quan

| Metric | Giá trị |
|--------|--------:|
| Accuracy | 0.660 |
| Precision (macro) | 0.686 |
| Recall (macro) | 0.676 |
| F1-macro | 0.654 |
| Strategy | `weighted_max`, chunk 800 ký tự |

### 4.2. Theo nhãn

| Nhãn | Precision | Recall | F1 | Support |
|------|----------:|-------:|---:|--------:|
| ĐÚNG | 0.74 | **0.38** | 0.50 | 37 |
| SAI | 0.74 | 0.82 | **0.78** | 28 |
| KHÔNG ĐỦ BẰNG CHỨNG | 0.58 | 0.83 | 0.68 | 35 |

### 4.3. Ma trận nhầm lẫn (hàng = gold, cột = pred)

|  | ĐÚNG | SAI | NEI |
|--|-----:|----:|----:|
| **ĐÚNG** | 14 | 6 | **17** |
| **SAI** | 1 | 23 | 4 |
| **NEI** | 4 | 2 | 29 |

### 4.4. Phân tích lỗi

- Phân bố dự đoán: NEI **50**, SAI 31, ĐÚNG chỉ 19 → **thiên NEI**.
- Lỗi phổ biến nhất: **ĐÚNG → NEI (17/34 lỗi)**.
- Nhãn **SAI** tốt nhất (F1 0.78).
- Điểm tin cậy trung bình đúng/sai gần nhau (~0.87 vs ~0.85) → khó lọc lỗi chỉ bằng threshold đơn giản.

**Giải thích hợp lý:** NLI nhận full `Context` dài; tín hiệu entailment bị pha loãng giữa nhiều câu không liên quan → model chọn NEI an toàn.

**Kết luận thành phần:** NLI đạt baseline chấp nhận được trên claim–context y tế tiếng Việt; điểm nghẽn là recall ĐÚNG.

---

## 5. LLM explanation (Dataset B, n=100, qwen3:8b)

### 5.1. Metric

| Metric | Giá trị |
|--------|--------:|
| ROUGE-L (trung bình) | 0.328 |
| BERTScore P / R / F1 | 0.694 / 0.765 / **0.727** |
| Số mẫu chấm được | 100/100 |
| Generated rỗng | **0/100** |

### 5.2. Phân bố chất lượng

- ROUGE-L trung vị ≈ 0.330; **94/100** mẫu > 0.20; **66/100** > 0.30.
- Độ dài sinh ≈ 406 ký tự; reference (gold Evidence) ≈ 184 ký tự.

### 5.3. Phân tích

- Trước đó `qwen3:4b` trả ~99% content rỗng (thinking chiếm token). Chuyển `qwen3:8b` + tắt thinking → eval ổn định.
- ROUGE thấp hơn BERTScore một phần vì **khác genre**: model sinh *giải thích*; reference là *đoạn evidence* gốc — overlap từ vựng hạn chế dù ngữ nghĩa liên quan.
- Verdict đưa vào `explain` vẫn lệch NEI (56/100) vì lấy từ NLI phía trước.

**Kết luận thành phần:** Module giải thích đủ điều kiện báo cáo; nên ưu tiên BERTScore (hoặc đánh giá thủ công) hơn ROUGE khi reference là evidence.

---

## 6. End-to-End (Dataset A, n=100)

Pipeline: Retrieval → Rerank → NLI multi-evidence (`with_explanation=False`).

### 6.1. Tổng quan

| Metric | Giá trị |
|--------|--------:|
| Accuracy | 0.500 |
| F1-macro | 0.440 |
| Precision / Recall (macro) | 0.449 / 0.504 |
| Latency mean / p50 / p95 | **10.56s / 8.26s / 24.77s** |

### 6.2. Theo nhãn

| Nhãn | Precision | Recall | F1 | Support |
|------|----------:|-------:|---:|--------:|
| ĐÚNG | 0.57 | 0.68 | **0.62** | 44 |
| SAI | 0.45 | 0.74 | 0.56 | 23 |
| KHÔNG ĐỦ BẰNG CHỨNG | 0.33 | **0.09** | 0.14 | 33 |

### 6.3. Ma trận nhầm lẫn

|  | ĐÚNG | SAI | NEI |
|--|-----:|----:|----:|
| **ĐÚNG** | 30 | 9 | 5 |
| **SAI** | 5 | 17 | 1 |
| **NEI** | **18** | **12** | 3 |

### 6.4. Phân tích

- E2E **thiên ĐÚNG/SAI**, gần như không dự đoán NEI (chỉ 9/100) — **ngược chiều** với NLI trên ViFactCheck (thiên NEI).
- Lỗi lớn: gold NEI bị gán ĐÚNG (18) hoặc SAI (12). Khi retrieve được đoạn “có vẻ liên quan”, NLI dễ entail/contradict thay vì NEI.
- Gap Retrieval Hit@5 0.80 → E2E Acc 0.50 cho thấy phần lớn lỗi đến từ **giai đoạn suy luận trên evidence đã retrieve** (và 16% miss retrieval).

**Kết luận E2E:** Hệ thống dùng được ở mức tham khảo; điểm yếu là phân biệt NEI khi evidence nhiễu / không đủ.

---

## 7. So sánh baseline (Dataset A, n=100)

| Phương pháp | Accuracy | F1-macro | P-macro | R-macro | Latency TB |
|-------------|---------:|---------:|--------:|--------:|-----------:|
| **LLM-only** | **0.51** | **0.495** | 0.531 | 0.481 | **1.15s** |
| RAG + LLM | 0.44 | 0.318 | 0.445 | 0.360 | 47.17s |
| RAG + NLI (đề xuất) | 0.50 | 0.440 | 0.449 | 0.504 | 9.30s |

### 7.1. Chi tiết từng phương pháp

**LLM-only**

- CM: ĐÚNG khá ổn (29/44); SAI yếu (recall 0.39); NEI trung bình.
- Ưu: nhanh, Acc/F1 cao nhất trên tập này.
- Nhược: không có bằng chứng trích dẫn → kém tin cậy khi triển khai thực tế.

**RAG + LLM (kém nhất)**

- Dự đoán ĐÚNG tới **77/100**; SAI chỉ **2/100** (recall SAI = 0.04).
- Evidence retrieved có xu hướng “dẫn dắt” LLM khẳng định ĐÚNG.
- Latency rất cao (~47s/mẫu) do retrieval + gọi LLM classify.

**RAG + NLI (pipeline đề xuất)**

- Trùng số liệu E2E (cùng pipeline): Acc 0.50, F1 0.44.
- Bắt SAI tốt hơn nhiều so với RAG+LLM (recall SAI 0.74 vs 0.04).
- Nhanh hơn RAG+LLM ~5×; chậm hơn LLM-only ~8×.
- Cho phép trả evidence + score → phù hợp yêu cầu hệ thống fact-check.

### 7.2. Nhận định so sánh

1. Trên Viet-MedFact (claim sinh từ bài đã index), **LLM-only cạnh tranh về Accuracy** vì kiến thức tham số đôi khi đủ.
2. **RAG+LLM không cải thiện** so với LLM-only; thậm chí làm xấu do bias ĐÚNG.
3. **RAG+NLI** không dẫn đầu Acc thuần, nhưng:
   - phân lớp SAI/ĐÚNG cân hơn RAG+LLM,
   - có evidence + điểm NLI,
   - latency chấp nhận được hơn RAG+LLM.
4. Kết quả baseline cũ (n=10, Acc RAG+NLI=0.80) **không còn đại diện**; số liệu n=100 mới đáng dùng trong luận văn.

---

## 8. Đối chiếu hai bối cảnh đánh giá

| Khía cạnh | ViFactCheck NLI (B) | Viet-MedFact E2E (A) |
|-----------|---------------------|----------------------|
| Evidence | Context vàng (đầy đủ bài) | Top chunk sau retrieve+rerank |
| Accuracy | 0.66 | 0.50 |
| Bias | Thiên **NEI** | Thiên **ĐÚNG/SAI** |
| Ý nghĩa | Trần năng lực NLI khi có đủ ngữ cảnh | Hiệu năng thực tế pipeline |

→ Cải thiện retrieval chưa đủ; cần cải NLI/calibration trên evidence ngắn và nhiễu.

---

## 9. Điểm mạnh / điểm yếu

### Điểm mạnh

1. Reranker mang lại lợi ích rõ (Hit@5 0.70→0.80, MRR 0.57→0.70).
2. NLI tiếng Việt trên ViFactCheck đạt Acc 0.66 không cần dịch PUBHEALTH.
3. Module giải thích LLM ổn định với `qwen3:8b` (BERTScore F1 0.73).
4. RAG+NLI bắt nhãn SAI tốt hơn rõ so với RAG+LLM trên cùng evidence.

### Điểm yếu

1. E2E Acc chỉ 0.50; NEI gần như fail (recall 0.09).
2. NLI (gold context) recall ĐÚNG thấp (0.38), thiên NEI.
3. RAG+LLM bias ĐÚNG nghiêm trọng → không nên dùng làm classifier chính.
4. 16% claim không retrieve đúng bài trong top-20.

---

## 10. Hướng cải thiện đề xuất

1. **NLI:** fine-tune trên ViFactCheck-Health / hard negatives; thử aggregate khác; cân nhắc dùng gold Evidence khi train.
2. **E2E NEI:** ngưỡng tin cậy (abstain khi max-prob thấp); đưa “evidence relevance score” vào quyết định NEI.
3. **Retrieval:** mở rộng corpus / hybrid BM25+dense; phân tích 16 case miss.
4. **LLM:** giữ vai trò *giải thích*, không dùng làm verdict chính (tránh lỗi RAG+LLM).
5. **Đánh giá:** báo cáo đồng thời Acc/F1 theo nhãn + latency; nêu rõ Dataset A vs B khi trình bày.

---

## 11. Kết luận tổng thể

Hệ thống **RAG (bge-m3) + Reranker + NLI (mDeBERTa) + LLM giải thích (qwen3:8b)** hoạt động end-to-end ổn định trên dữ liệu y tế tiếng Việt.

- Thành phần **truy xuất/rerank** và **NLI trên context vàng** cho tín hiệu tích cực.
- Hiệu năng **toàn trình** còn khoảng trống lớn so với NLI oracle-context (0.50 vs 0.66), chủ yếu do evidence retrieve nhiễu và calibration NEI.
- So với các baseline: **RAG+NLI là lựa chọn triển khai hợp lý hơn RAG+LLM** (công bằng nhãn hơn, có bằng chứng, nhanh hơn), dù Accuracy thuần trên Viet-MedFact chưa vượt LLM-only.

Các số liệu trong báo cáo này lấy từ file JSON mới nhất trong `evaluation/results/` (30–31/07/2026) và phù hợp để đưa vào phần thực nghiệm của luận văn.
