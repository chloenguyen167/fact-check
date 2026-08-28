"""
Đánh giá module NLI trên Dataset B (ViFactCheck-Health hoặc PUBHEALTH-Vi).

Đưa trực tiếp claim + main_text vào mDeBERTa, bỏ qua Retrieval.
Sử dụng multi-evidence NLI: split main_text thành chunks, chạy NLI
trên từng chunk, rồi aggregate để có verdict chính xác hơn.

Metrics: Accuracy, Precision, Recall, F1-Macro, Confusion Matrix
"""

import argparse
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import EVAL_MAX_SAMPLES, RESULTS_DIR
from evaluation.io_utils import read_jsonl
from evaluation.labels import VERDICTS, dataset_b_path, pubhealth_label_to_verdict
from evaluation.metrics import classification_metrics, save_json, save_text
from nli import predict_nli_multi, set_nli_model, split_into_chunks


def run_evaluation(records, chunk_size=500, strategy="weighted_max"):
    y_true = []
    y_pred = []
    details = []

    for row in tqdm(records, desc="Đánh giá NLI"):
        gold = pubhealth_label_to_verdict(row["label"])
        if gold is None:
            continue

        chunks = split_into_chunks(row["main_text"], max_chars=chunk_size)
        pred, score, chunk_details = predict_nli_multi(
            row["claim"], chunks, strategy=strategy
        )

        y_true.append(gold)
        y_pred.append(pred)
        details.append({
            "claim": row["claim"],
            "label_gold": gold,
            "label_pred": pred,
            "score": float(score),
            "num_chunks": len(chunks),
            "claim_id": row.get("claim_id", ""),
        })

    metrics = classification_metrics(y_true, y_pred, VERDICTS)
    metrics["num_samples"] = len(y_true)
    metrics["strategy"] = strategy
    metrics["chunk_size"] = chunk_size
    metrics["details"] = details

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Đánh giá NLI")
    parser.add_argument(
        "--dataset-b",
        choices=["vifactcheck", "pubhealth"],
        default=None,
        help="Nguồn Dataset B (mặc định: config DATASET_B_DEFAULT)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Đường dẫn jsonl (ghi đè --dataset-b)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "nli_results.json"),
    )
    parser.add_argument("--max-samples", type=int, default=EVAL_MAX_SAMPLES)
    parser.add_argument(
        "--chunk-size", type=int, default=800,
        help="Kích thước chunk (ký tự) khi split main_text",
    )
    parser.add_argument(
        "--strategy", default="weighted_max",
        choices=["weighted_max", "mean"],
        help="Chiến lược aggregate NLI scores từ nhiều chunks",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to fine-tuned NLI model (default: use NLI_MODEL from config)",
    )
    args = parser.parse_args()

    if args.model:
        print(f"Using fine-tuned model: {args.model}")
        set_nli_model(args.model)

    input_path = args.input or dataset_b_path(args.dataset_b)
    if not os.path.exists(input_path):
        hint = (
            "python -m evaluation.prepare_vifactcheck"
            if "vifactcheck" in input_path
            else "python -m evaluation.prepare_pubhealth"
        )
        raise FileNotFoundError(f"Chưa có {input_path}. Chạy: {hint}")

    records = read_jsonl(input_path)
    if args.max_samples:
        records = records[: args.max_samples]

    print(f"Input: {input_path} ({len(records)} mẫu)")
    print(f"Strategy: {args.strategy}, chunk_size: {args.chunk_size}")

    results = run_evaluation(
        records, chunk_size=args.chunk_size, strategy=args.strategy
    )
    save_json(results, args.output)
    save_text(
        results["classification_report"],
        args.output.replace(".json", "_report.txt"),
    )

    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print(f"F1-Macro: {results['f1_macro']:.4f}")
    print(f"Precision (macro): {results['precision_macro']:.4f}")
    print(f"Recall (macro): {results['recall_macro']:.4f}")
    print(f"\nConfusion matrix ({results['labels']}):")
    print(results["confusion_matrix"])
    print(f"\nĐã lưu -> {args.output}")


if __name__ == "__main__":
    main()
