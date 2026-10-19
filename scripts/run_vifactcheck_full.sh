#!/usr/bin/env bash
# Đánh giá ViFactCheck-Health full (438 mẫu), chia 2 phase:
#   nollm : NLI, retrieval, baseline RAG+NLI, ablation (không cần LLM)
#   llm   : baseline LLM-only / RAG+LLM / RAG+NLI+LLM, LLM explanation (cần Ollama qwen3:8b)
#
# Kết quả ghi riêng vào evaluation/results/vifactcheck_full/ (không ghi đè kết quả n=100 cũ).
# Tiến độ cả 2 phase: evaluation/results/vifactcheck_full/PROGRESS.md (cập nhật mỗi 20s, giờ GMT+7).
# Chạy lại sẽ bỏ qua các bước đã xong; ablation_cache tự resume theo idx.
#
#   bash scripts/run_vifactcheck_full.sh nollm            # mở tmux session "vifc-nollm"
#   bash scripts/run_vifactcheck_full.sh llm              # mở tmux session "vifc-llm"
#   tmux attach -t vifc-llm                                # xem trực tiếp (thoát: Ctrl+b rồi d)
#   cat evaluation/results/vifactcheck_full/PROGRESS.md    # xem tiến độ
#
# Yêu cầu: .venv đã cài, đã chạy `python -m evaluation.index_vifactcheck --recreate`,
# không có process nào khác (streamlit, eval khác) đang giữ lock qdrant_db/.

set -uo pipefail
export TZ="${TZ_DISPLAY:-Asia/Ho_Chi_Minh}"

PHASE="${1:-}"
if [[ "$PHASE" != nollm && "$PHASE" != llm ]]; then
  echo "Dùng: bash scripts/run_vifactcheck_full.sh <nollm|llm>"
  exit 1
fi

SESSION="vifc-$PHASE"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"

if [[ -z "${TMUX:-}" && "${2:-}" != "--inner" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' đã tồn tại: tmux attach -t $SESSION"
    exit 1
  fi
  tmux new-session -d -s "$SESSION" -c "$ROOT" \
    "bash '$SELF' $PHASE --inner; echo; echo '[Kết thúc] Enter để đóng session'; read"
  echo "Đã chạy phase '$PHASE' trong tmux session '$SESSION'"
  echo "  Xem trực tiếp : tmux attach -t $SESSION"
  echo "  Tiến độ       : $ROOT/evaluation/results/vifactcheck_full/PROGRESS.md"
  exit 0
fi

cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
OUT="evaluation/results/vifactcheck_full"
LOGS="$OUT/logs"
STATE="$OUT/.state"
PROGRESS="$OUT/PROGRESS.md"
N=438
mkdir -p "$LOGS" "$STATE" "$OUT/ablation"

# torch mặc định mở thread theo số core máy host → bị throttle nặng khi container có CPU quota.
# Giới hạn thread theo quota cgroup (v2: cpu.max, v1: cpu.cfs_quota_us).
cpu_quota() {
  local q p
  if [[ -f /sys/fs/cgroup/cpu.max ]]; then read -r q p < /sys/fs/cgroup/cpu.max
  else q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null); fi
  if [[ "$q" =~ ^[0-9]+$ && "$p" =~ ^[0-9]+$ && $p -gt 0 ]]; then echo $(( q / p > 0 ? q / p : 1 )); else nproc; fi
}
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(cpu_quota)}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

NOLLM_STEPS=(
  nli nli_plot nli_errors retrieval baseline_rag_nli
  ablation_cache_oracle ablation_cache_e2e
  ablation_analyze_oracle ablation_analyze_e2e
)
LLM_STEPS=(
  baseline_llm_only baseline_rag_llm baseline_rag_nli_llm llm_explain baseline_merge
  ablation_analyze_e2e_llm
)
ALL_STEPS=("${NOLLM_STEPS[@]}" "${LLM_STEPS[@]}")
if [[ "$PHASE" == nollm ]]; then STEPS=("${NOLLM_STEPS[@]}"); else STEPS=("${LLM_STEPS[@]}"); fi

declare -A DESC CMD
DESC[nli]="NLI oracle context (weighted_max, chunk 800)"
CMD[nli]="$PY -m evaluation.eval_nli --dataset-b vifactcheck --output $OUT/nli_results.json"
DESC[nli_plot]="Confusion matrix NLI (PNG)"
CMD[nli_plot]="$PY -m evaluation.plot_confusion_matrix --input $OUT/nli_results.json --output $OUT/nli_confusion_matrix.png"
DESC[nli_errors]="Phân tích lỗi NLI"
CMD[nli_errors]="$PY -m evaluation.error_analysis --e2e $OUT/_none.json --nli $OUT/nli_results.json --output $OUT/error_analysis_nli.json"
DESC[retrieval]="Retrieval + Rerank (Hit@K, MRR)"
CMD[retrieval]="$PY -m evaluation.eval_retrieval --dataset vifactcheck --output $OUT/retrieval_results.json"
DESC[baseline_rag_nli]="Baseline RAG+NLI (verify_claim, ESS)"
CMD[baseline_rag_nli]="$PY -m evaluation.eval_baselines --dataset vifactcheck --methods rag_nli --output $OUT/baseline_rag_nli.json"
DESC[ablation_cache_oracle]="Ablation cache — oracle"
CMD[ablation_cache_oracle]="$PY -m evaluation.ablation_cache --dataset vifactcheck --mode oracle --max-samples $N --output $OUT/ablation/cache_vifactcheck_oracle.jsonl"
DESC[ablation_cache_e2e]="Ablation cache — e2e"
CMD[ablation_cache_e2e]="$PY -m evaluation.ablation_cache --dataset vifactcheck --mode e2e --max-samples $N --output $OUT/ablation/cache_vifactcheck_e2e.jsonl"
DESC[ablation_analyze_oracle]="Ablation analyze — oracle (test 100 / dev 338)"
CMD[ablation_analyze_oracle]="$PY -m evaluation.ablation_analyze --dataset vifactcheck --mode oracle --cache $OUT/ablation/cache_vifactcheck_oracle.jsonl --output $OUT/ablation/summary_vifactcheck_oracle.json"
DESC[ablation_analyze_e2e]="Ablation analyze — e2e (test 100 / dev 338)"
CMD[ablation_analyze_e2e]="$PY -m evaluation.ablation_analyze --dataset vifactcheck --mode e2e --cache $OUT/ablation/cache_vifactcheck_e2e.jsonl --output $OUT/ablation/summary_vifactcheck_e2e.json"

DESC[baseline_llm_only]="Baseline LLM-only (qwen3:8b, không evidence)"
CMD[baseline_llm_only]="$PY -m evaluation.eval_baselines --dataset vifactcheck --methods llm_only --output $OUT/baseline_llm_only.json"
DESC[baseline_rag_llm]="Baseline RAG+LLM (LLM gán nhãn từ top-3 chunk)"
CMD[baseline_rag_llm]="$PY -m evaluation.eval_baselines --dataset vifactcheck --methods rag_llm --output $OUT/baseline_rag_llm.json"
DESC[baseline_rag_nli_llm]="Baseline RAG+NLI+LLM (ESS + giải thích)"
CMD[baseline_rag_nli_llm]="$PY -m evaluation.eval_baselines --dataset vifactcheck --methods rag_nli_llm --output $OUT/baseline_rag_nli_llm.json"
DESC[llm_explain]="LLM explanation (ROUGE-L, BERTScore)"
CMD[llm_explain]="$PY -m evaluation.eval_llm --dataset-b vifactcheck --max-samples $N --output $OUT/llm_results.json"
DESC[baseline_merge]="Gộp 4 baseline → baseline_comparison_vifactcheck.json"
CMD[baseline_merge]="$PY $ROOT/scripts/merge_vifactcheck_baselines.py $OUT"
DESC[ablation_analyze_e2e_llm]="Ablation analyze — e2e, so với LLM baseline của run full"
CMD[ablation_analyze_e2e_llm]="$PY -m evaluation.ablation_analyze --dataset vifactcheck --mode e2e --cache $OUT/ablation/cache_vifactcheck_e2e.jsonl --baselines $OUT/baseline_comparison_vifactcheck.json --output $OUT/ablation/summary_vifactcheck_e2e.json"

phase_of() { [[ " ${LLM_STEPS[*]} " == *" $1 "* ]] && echo llm || echo nollm; }

now() { date '+%Y-%m-%d %H:%M:%S'; }
fmt_dur() { printf '%dh%02dm%02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

# state/<step>: status|start|end|seconds|metrics
get_field() { [[ -f "$STATE/$1" ]] && cut -d'|' -f"$2" "$STATE/$1" || echo ""; }
set_state() { printf '%s|%s|%s|%s|%s\n' "$2" "$3" "$4" "$5" "$6" > "$STATE/$1"; }

# Dòng tqdm cuối cùng trong log của bước đang chạy
live_progress() {
  tail -c 4000 "$LOGS/$1.log" 2>/dev/null | tr '\r' '\n' | grep -E '[0-9]+/[0-9]+ \[' | tail -1 \
    | sed -E 's/\|[^|]*\|/|/; s/\[llm_utils\].*//' | cut -c1-110
}

metrics() {
  "$PY" - "$1" "$OUT" <<'EOF'
import json, os, sys
step, out = sys.argv[1], sys.argv[2]
def j(p): return json.load(open(os.path.join(out, p), encoding="utf-8"))
def f(x): return f"{x:.3f}"
try:
    if step == "nli":
        d = j("nli_results.json")
        print(f"n={len(d['details'])} · Acc {f(d['accuracy'])} · F1 {f(d['f1_macro'])}")
    elif step == "nli_errors":
        d = j("error_analysis_nli.json")["nli"]
        print(f"{d['total_errors']} lỗi")
    elif step == "retrieval":
        d = j("retrieval_results.json")["retrieval_metrics"]
        r, rr = d["retrieval_only"], d["with_reranker"]
        print(f"n={r['num_samples']} · Hit@5 {f(r['hit@5'])}→{f(rr['hit@5'])} · MRR {f(r['mrr'])}→{f(rr['mrr'])} (dense→rerank)")
    elif step.startswith("baseline_") and step != "baseline_merge":
        method = step.removeprefix("baseline_")
        s = j(f"{step}.json")["summary"][method]
        print(f"n={s['num_samples']} · Acc {f(s['accuracy'])} · F1 {f(s['f1_macro'])} · {s['latency_mean_sec']:.2f}s/claim")
    elif step == "baseline_merge":
        s = j("baseline_comparison_vifactcheck.json")["summary"]
        print(" · ".join(f"{k} F1 {f(v['f1_macro'])}" for k, v in s.items()))
    elif step == "llm_explain":
        d = j("llm_results.json")
        bs = d.get("bertscore") or {}
        print(f"n={d['num_samples']} · ROUGE-L {f(d['rouge_l'])} · BERTScore F1 {f(bs['f1']) if bs else 'n/a'}")
    elif step.startswith("ablation_cache_"):
        mode = step.rsplit("_", 1)[1]
        print(f"{sum(1 for _ in open(os.path.join(out, 'ablation', f'cache_vifactcheck_{mode}.jsonl')))} dòng cache")
    elif step.startswith("ablation_analyze_"):
        mode = step.split("_")[2]
        v = j(f"ablation/summary_vifactcheck_{mode}.json")["variants"]
        print(" · ".join(f"{k} Acc {f(v[k]['accuracy'])}/F1 {f(v[k]['f1_macro'])}" for k in ("FULL", "B0") if k in v) + " (test)")
except Exception as e:
    print(f"(không đọc được metric: {type(e).__name__})")
EOF
}

render() {
  local step status start secs met icon live
  {
    echo "# Tiến độ đánh giá ViFactCheck-Health full (n=$N)"
    echo
    echo "- Script: \`scripts/run_vifactcheck_full.sh <nollm|llm>\` · tmux session: \`vifc-nollm\` / \`vifc-llm\`"
    echo "- Dataset: \`evaluation/data/vifactcheck_health.jsonl\` · output: \`$OUT/\`"
    echo "- Giờ hiển thị: GMT+7 · cập nhật lúc: $(now)"
    echo
    echo "| # | Phase | Bước | Trạng thái | Bắt đầu | Thời gian | Kết quả / tiến độ |"
    echo "|---|-------|------|-----------|---------|-----------|-------------------|"
    local i=0
    for step in "${ALL_STEPS[@]}"; do
      i=$((i + 1))
      status=$(get_field "$step" 1); start=$(get_field "$step" 2)
      secs=$(get_field "$step" 4); met=$(get_field "$step" 5)
      case "$status" in
        done) icon="✅ xong" ;;
        failed) icon="❌ lỗi" ;;
        running) icon="⏳ đang chạy"; secs=$(( $(date +%s) - $(date -d "$start" +%s) ))
                 live=$(live_progress "$step"); met="${live:-…}" ;;
        *) icon="⬜ chờ"; start=""; secs="" ;;
      esac
      [[ -n "$secs" ]] && secs=$(fmt_dur "$secs")
      echo "| $i | $(phase_of "$step") | ${DESC[$step]} (\`$step\`) | $icon | ${start:-} | ${secs:-} | ${met//|/\\|} |"
    done
    echo
    echo "Log từng bước: \`$LOGS/<bước>.log\` · lỗi gần nhất xem cuối log."
  } > "$PROGRESS.tmp" && mv "$PROGRESS.tmp" "$PROGRESS"
}

run_step() {
  local step="$1" start t0 rc pid
  if [[ "$(get_field "$step" 1)" == "done" ]]; then
    echo "[skip] $step (đã xong)"
    return 0
  fi
  start=$(now); t0=$(date +%s)
  set_state "$step" running "$start" "" "" ""
  render
  echo
  echo "=== [$start] $step — ${DESC[$step]}"
  echo "    ${CMD[$step]}"

  # PYTHONUNBUFFERED để tqdm/print ra log ngay; tee để vẫn thấy trong tmux
  PYTHONUNBUFFERED=1 bash -c "${CMD[$step]}" > >(tee "$LOGS/$step.log") 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
    render
  done
  wait "$pid"; rc=$?

  local secs=$(( $(date +%s) - t0 ))
  if [[ $rc -eq 0 ]]; then
    set_state "$step" done "$start" "$(now)" "$secs" "$(metrics "$step")"
    echo "=== $step xong sau $(fmt_dur "$secs")"
  else
    set_state "$step" failed "$start" "$(now)" "$secs" "exit $rc — xem $LOGS/$step.log"
    echo "=== $step LỖI (exit $rc) — dừng. Sửa lỗi rồi chạy lại script để tiếp tục."
  fi
  render
  return $rc
}

ensure_ollama() {
  local host="http://localhost:11434" model
  model=$("$PY" -c "import config; print(config.OLLAMA_MODEL)")
  if ! curl -s -m 3 "$host/api/version" >/dev/null; then
    echo "Ollama chưa chạy → khởi động ollama serve (log: /workspace/ollama.log)"
    nohup ollama serve > /workspace/ollama.log 2>&1 &
    for _ in $(seq 1 30); do curl -s -m 2 "$host/api/version" >/dev/null && break; sleep 1; done
  fi
  if ! curl -s -m 3 "$host/api/tags" | grep -q "\"$model\""; then
    echo "Ollama không có model $model — chạy: ollama pull $model"
    return 1
  fi
  echo "Ollama OK ($model)"
}

if [[ ! -x "$PY" ]]; then echo "Không thấy $PY"; exit 1; fi
if [[ ! -d qdrant_db/collection/vifactcheck_health_bge_m3 ]]; then
  echo "Chưa có collection vifactcheck_health_bge_m3 — chạy: python -m evaluation.index_vifactcheck --recreate"
  exit 1
fi
if [[ "$PHASE" == llm ]]; then ensure_ollama || exit 1; fi

render
T_ALL=$(date +%s)
for step in "${STEPS[@]}"; do
  run_step "$step" || exit 1
done
echo
echo "Hoàn tất phase '$PHASE' sau $(fmt_dur $(( $(date +%s) - T_ALL ))). Tiến độ: $PROGRESS"
