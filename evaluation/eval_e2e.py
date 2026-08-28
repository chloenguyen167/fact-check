"""
Đánh giá toàn trình (End-to-End) trên tập Viet-MedFact.

Metrics: Accuracy, Latency
"""

import argparse
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import EVAL_MAX_SAMPLES, RESULTS_DIR, VIET_MEDFACT_PATH
from evaluation.io_utils import read_jsonl
from evaluation.labels import VERDICTS, normalize_verdict
from evaluation.metrics import classification_metrics, save_json, summarize_latency
from pipeline import verify_claim


def run_evaluation(records, with_explanation=False):
    y_true = []
    y_pred = []
    latencies = []
    details = []

    for row in tqdm(records, desc="Đánh giá E2E"):
        gold = normalize_verdict(row.get("label"))
        if gold is None:
            continue

        start = time.perf_counter()
        result = verify_claim(row["claim"], with_explanation=with_explanation, use_cache=False)
        elapsed = time.perf_counter() - start

        y_true.append(gold)
        y_pred.append(result["verdict"])
        latencies.append(elapsed)

        details.append({
            "claim": row["claim"],
            "label_gold": gold,
            "label_pred": result["verdict"],
            "verdict_score": result.get("verdict_score"),
            "article_id": row.get("article_id"),
            "latency_sec": elapsed,
            "evidence_preview": result.get("evidence", "")[:300],
        })

    metrics = classification_metrics(y_true, y_pred, VERDICTS)
    metrics["latency"] = summarize_latency(latencies)
    metrics["with_explanation"] = with_explanation
    metrics["details"] = details

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Đánh giá End-to-End")
    parser.add_argument("--input", default=VIET_MEDFACT_PATH)
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "e2e_results.json"),
    )
    parser.add_argument("--max-samples", type=int, default=EVAL_MAX_SAMPLES)
    parser.add_argument(
        "--with-explanation",
        action="store_true",
        help="Bao gồm bước sinh giải thích LLM (chậm hơn)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Chưa có {args.input}. Chạy: python -m evaluation.generate_viet_medfact"
        )

    records = read_jsonl(args.input)
    if args.max_samples:
        records = records[: args.max_samples]

    results = run_evaluation(records, with_explanation=args.with_explanation)
    save_json(results, args.output)

    print(f"End-to-End Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Macro: {results['f1_macro']:.4f}")
    print(f"Latency trung bình: {results['latency']['mean_sec']:.2f}s")
    print(f"P95 latency: {results['latency']['p95_sec']:.2f}s")
    print(f"Đã lưu -> {args.output}")


if __name__ == "__main__":
    main()
