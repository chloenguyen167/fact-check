"""
Đánh giá module giải thích LLM trên Dataset B
(ViFactCheck-Health hoặc PUBHEALTH-Vi).

So sánh giải thích sinh ra với explanation/evidence tham chiếu.
Metrics: ROUGE-L, BERTScore (nếu cài bert-score)
"""

import argparse
import os
import sys

from rouge_score import rouge_scorer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import EVAL_MAX_SAMPLES, RESULTS_DIR
from evaluation.io_utils import read_jsonl
from evaluation.labels import dataset_b_path, pubhealth_label_to_verdict
from evaluation.metrics import save_json
from llm import explain
from nli import get_prediction, predict_nli


def compute_rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            scores.append(0.0)
            continue
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def compute_bertscore(predictions, references):
    try:
        from bert_score import score as bert_score
    except ImportError:
        return None

    # bert-score + transformers>=5 crash trên chuỗi rỗng
    # (BertTokenizer thiếu build_inputs_with_special_tokens).
    pairs = [
        (str(pred).strip(), str(ref).strip())
        for pred, ref in zip(predictions, references)
        if str(pred or "").strip() and str(ref or "").strip()
    ]
    if not pairs:
        return None

    preds, refs = zip(*pairs)
    try:
        precision, recall, f1 = bert_score(
            list(preds),
            list(refs),
            lang="vi",
            rescale_with_baseline=False,
        )
    except Exception as exc:
        print(f"BERTScore: bỏ qua do lỗi tương thích — {exc}")
        return None

    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "num_scored": len(pairs),
    }


def run_evaluation(records, max_samples=None):
    predictions = []
    references = []
    details = []

    subset = records[:max_samples] if max_samples else records

    for row in tqdm(subset, desc="Đánh giá LLM explanation"):
        verdict = pubhealth_label_to_verdict(row["label"])
        if verdict is None:
            continue

        nli_result = predict_nli(row["claim"], row["main_text"])
        verdict_pred, _ = get_prediction(nli_result)

        generated = explain(row["claim"], row["main_text"], verdict_pred)
        reference = row.get("explanation") or row.get("evidence") or ""

        predictions.append(generated)
        references.append(reference)
        details.append({
            "claim": row["claim"],
            "verdict": verdict_pred,
            "generated": generated,
            "reference": reference,
            "claim_id": row.get("claim_id", ""),
        })

    rouge_l = compute_rouge_l(predictions, references)
    bertscore = compute_bertscore(predictions, references)

    return {
        "rouge_l": rouge_l,
        "bertscore": bertscore,
        "num_samples": len(predictions),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Đánh giá LLM explanation")
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
        default=os.path.join(RESULTS_DIR, "llm_results.json"),
    )
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()

    input_path = args.input or dataset_b_path(args.dataset_b)
    if not os.path.exists(input_path):
        hint = (
            "python -m evaluation.prepare_vifactcheck"
            if "vifactcheck" in input_path
            else "python -m evaluation.prepare_pubhealth"
        )
        raise FileNotFoundError(f"Chưa có {input_path}. Chạy: {hint}")

    records = read_jsonl(input_path)
    print(f"Input: {input_path}")
    results = run_evaluation(records, max_samples=args.max_samples or EVAL_MAX_SAMPLES)
    save_json(results, args.output)

    print(f"ROUGE-L: {results['rouge_l']:.4f}")
    if results["bertscore"]:
        print(f"BERTScore F1: {results['bertscore']['f1']:.4f}")
    else:
        print("BERTScore: bỏ qua (chưa cài / lỗi tương thích transformers)")
    print(f"Đã lưu -> {args.output}")


if __name__ == "__main__":
    main()
