# Tiến độ đánh giá ViFactCheck-Health full (n=438)

- Script: `scripts/run_vifactcheck_full.sh <nollm|llm>` · tmux session: `vifc-nollm` / `vifc-llm`
- Dataset: `evaluation/data/vifactcheck_health.jsonl` · output: `evaluation/results/vifactcheck_full/`
- Giờ hiển thị: GMT+7 · cập nhật lúc: 2026-10-19 22:48:57

| # | Phase | Bước | Trạng thái | Bắt đầu | Thời gian | Kết quả / tiến độ |
|---|-------|------|-----------|---------|-----------|-------------------|
| 1 | nollm | NLI oracle context (weighted_max, chunk 800) (`nli`) | ✅ xong | 2026-10-19 18:29:29 | 0h01m41s | n=438 · Acc 0.616 · F1 0.608 |
| 2 | nollm | Confusion matrix NLI (PNG) (`nli_plot`) | ✅ xong | 2026-10-19 18:31:10 | 0h00m20s |  |
| 3 | nollm | Phân tích lỗi NLI (`nli_errors`) | ✅ xong | 2026-10-19 18:31:30 | 0h00m20s | 168 lỗi |
| 4 | nollm | Retrieval + Rerank (Hit@K, MRR) (`retrieval`) | ✅ xong | 2026-10-19 18:41:21 | 0h03m41s | n=438 · Hit@5 0.970→0.991 · MRR 0.910→0.964 (dense→rerank) |
| 5 | nollm | Baseline RAG+NLI (verify_claim, ESS) (`baseline_rag_nli`) | ✅ xong | 2026-10-19 18:45:02 | 0h08m43s | n=438 · Acc 0.571 · F1 0.576 · 1.10s/claim |
| 6 | nollm | Ablation cache — oracle (`ablation_cache_oracle`) | ✅ xong | 2026-10-19 18:53:45 | 0h05m42s | 438 dòng cache |
| 7 | nollm | Ablation cache — e2e (`ablation_cache_e2e`) | ✅ xong | 2026-10-19 18:59:27 | 0h09m04s | 438 dòng cache |
| 8 | nollm | Ablation analyze — oracle (test 100 / dev 338) (`ablation_analyze_oracle`) | ✅ xong | 2026-10-19 19:08:31 | 0h02m41s | FULL Acc 0.670/F1 0.674 · B0 Acc 0.660/F1 0.654 (test) |
| 9 | nollm | Ablation analyze — e2e (test 100 / dev 338) (`ablation_analyze_e2e`) | ✅ xong | 2026-10-19 19:11:12 | 0h02m42s | FULL Acc 0.600/F1 0.606 · B0 Acc 0.560/F1 0.547 (test) |
| 10 | llm | Baseline LLM-only (qwen3:8b, không evidence) (`baseline_llm_only`) | ✅ xong | 2026-10-19 19:24:33 | 0h04m05s | n=438 · Acc 0.372 · F1 0.317 · 0.47s/claim |
| 11 | llm | Baseline RAG+LLM (LLM gán nhãn từ top-3 chunk) (`baseline_rag_llm`) | ✅ xong | 2026-10-19 19:28:38 | 0h09m49s | n=438 · Acc 0.434 · F1 0.348 · 1.26s/claim |
| 12 | llm | Baseline RAG+NLI+LLM (ESS + giải thích) (`baseline_rag_nli_llm`) | ✅ xong | 2026-10-19 19:38:27 | 0h35m51s | n=438 · Acc 0.571 · F1 0.576 · 4.85s/claim |
| 13 | llm | LLM explanation (ROUGE-L, BERTScore) (`llm_explain`) | ✅ xong | 2026-10-19 20:14:18 | 0h27m44s | n=438 · ROUGE-L 0.321 · BERTScore F1 0.723 |
| 14 | llm | Gộp 4 baseline → baseline_comparison_vifactcheck.json (`baseline_merge`) | ✅ xong | 2026-10-19 20:42:02 | 0h00m21s | llm_only F1 0.317 · rag_llm F1 0.348 · rag_nli F1 0.576 · rag_nli_llm F1 0.576 |
| 15 | llm | Ablation analyze — e2e, so với LLM baseline của run full (`ablation_analyze_e2e_llm`) | ✅ xong | 2026-10-19 22:46:15 | 0h02m42s | FULL Acc 0.600/F1 0.606 · B0 Acc 0.560/F1 0.547 (test) |

Log từng bước: `evaluation/results/vifactcheck_full/logs/<bước>.log` · lỗi gần nhất xem cuối log.
