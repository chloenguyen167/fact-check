## File kết quả & độ mới

| File | Vai trò | Thời điểm |
|------|---------|-----------|
| `retrieval_results.json` | Retrieval / Rerank (Viet-MedFact) | 30/07 ✅ |
| `nli_results.json` (+ `_report.txt`) | NLI trên ViFactCheck | 30/07 ✅ |
| `llm_results.json` | Giải thích LLM (sau fix qwen3:8b) | 30/07 ✅ |
| `e2e_results.json` | End-to-end | **13/07 — cũ** |
| `baseline_comparison_10.json` | So sánh baseline | **27/06 — cũ, chỉ 10 mẫu** |

Chạy gần đây đủ 3 phần: Retrieval + NLI + LLM. E2E/baseline chưa chạy lại.

---

## 1. Retrieval (100 mẫu Viet-MedFact)

| | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR |
|--|------:|------:|------:|-------:|-------:|----:|
| Embedding only | 0.47 | 0.63 | 0.70 | 0.77 | 0.84 | 0.575 |
| **+ Reranker** | **0.63** | **0.76** | **0.80** | **0.82** | 0.84 | **0.703** |

- Reranker đẩy bài đúng lên cao rõ (Hit@1 +16đ, MRR +0.13).
- 16/100 không nằm trong top-20 → lỗi retrieval cứng, rerank không cứu được.
- Rank trung vị = 1 → khi tìm được thì thường đã ở vị trí tốt.

**Kết luận:** Module truy xuất + rerank ổn, đúng hướng kiến trúc đề xuất.

---

## 2. NLI trên ViFactCheck-Health (100 mẫu) — kết quả lõi

**Accuracy 0.66 · F1-macro 0.65 · P 0.69 · R 0.68**

| Nhãn | P | R | F1 | Support |
|------|--:|--:|---:|--------:|
| ĐÚNG | 0.74 | **0.38** | 0.50 | 37 |
| SAI | 0.74 | 0.82 | **0.78** | 28 |
| KHÔNG ĐỦ BẰNG CHỨNG | 0.58 | 0.83 | 0.68 | 35 |

**Ma trận nhầm lẫn (hàng = gold):**
- ĐÚNG → đoán đúng 14; nhầm NEI **17**; nhầm SAI 6  
- SAI → đúng 23 (tốt nhất)  
- NEI → đúng 29  

**Điểm yếu:** model thiên **NEI** (đoán 50/100). Nửa số claim ĐÚNG bị đẩy sang “không đủ bằng chứng” — thường gặp khi đưa full `Context` dài vào NLI (tín hiệu entailment bị pha loãng).

Score đúng/sai gần nhau (0.87 vs 0.85) → chưa tách chắc bằng ngưỡng tin cậy đơn giản.

---

## 3. LLM explanation (100 mẫu, qwen3:8b) — đã sửa

| Metric | Trước (4b) | Sau (8b) |
|--------|------------|----------|
| Empty output | 99/100 | **0/100** |
| ROUGE-L | ~0.003 | **0.328** |
| BERTScore F1 | ~1 mẫu | **0.727** (100/100) |

- Độ dài gen ~406 ký tự; ref (gold Evidence) ~184 → ROUGE thấp một phần vì **khác kiểu văn bản** (giải thích vs đoạn evidence), không chỉ vì chất lượng kém.
- 94/100 mẫu có ROUGE-L > 0.2; BERTScore ~0.73 là tín hiệu ngữ nghĩa ổn hơn ROUGE.
- Verdict trong bước LLM vẫn lệch NEI (56/100) — cùng bias với NLI vì `explain` dùng verdict từ NLI.

**Kết luận:** Fix qwen3:8b thành công; LLM eval đã dùng được cho báo cáo.

---

## 4. E2E & Baseline (file cũ — mang tính tham khảo)

**E2E (13/07, Viet-MedFact, 100 mẫu):** Acc **0.50**, F1 0.44, ~10s/mẫu.  
Nhãn NEI gần như fail (recall 0.09) — khác pattern NLI ViFactCheck (ở đó NEI lại mạnh). Cho thấy lỗi E2E đến nhiều từ **retrieval sai / pipeline ghép**, không chỉ NLI.

**Baseline 10 mẫu (27/06):**

| Method | Acc | F1 | Latency |
|--------|----:|---:|--------:|
| LLM-only | 0.30 | 0.22 | 1.1s |
| RAG + LLM | 0.30 | 0.26 | 25.7s |
| **RAG + NLI** | **0.80** | 0.53 | 19.5s |

Hướng RAG+NLI thắng rõ, nhưng n=10 quá nhỏ → cần chạy lại 50–100 mẫu.

---
