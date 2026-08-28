"""
Phân tích lỗi từ kết quả E2E và NLI.

Trích các case sai, phân loại nguyên nhân lỗi cơ bản.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import RESULTS_DIR
from evaluation.metrics import save_json


def classify_error(detail, eval_type="e2e"):
    gold = detail.get("label_gold")
    pred = detail.get("label_pred")

    if gold == pred:
        return None

    claim = detail.get("claim", "").lower()

    if "không" in claim and claim.count("không") >= 2:
        return "double_negation"

    if eval_type == "e2e" and detail.get("retrieval_rank") is None:
        return "retrieval_miss"

    if gold == "ĐÚNG" and pred == "KHÔNG ĐỦ BẰNG CHỨNG":
        return "db_missing_or_weak_evidence"

    if gold == "SAI" and pred == "KHÔNG ĐỦ BẰNG CHỨNG":
        return "nli_too_conservative"

    if gold == "KHÔNG ĐỦ BẰNG CHỨNG" and pred in {"ĐÚNG", "SAI"}:
        return "nli_overconfident"

    return "other"


def analyze_results(details, eval_type="e2e", max_examples=10):
    errors = []

    for detail in details:
        if detail.get("label_gold") == detail.get("label_pred"):
            continue

        errors.append({
            **detail,
            "error_type": classify_error(detail, eval_type=eval_type),
        })

    by_type = {}
    for item in errors:
        error_type = item["error_type"]
        by_type.setdefault(error_type, 0)
        by_type[error_type] += 1

    return {
        "total_errors": len(errors),
        "error_breakdown": by_type,
        "examples": errors[:max_examples],
    }


def merge_retrieval_into_e2e(e2e_details, retrieval_details):
    retrieval_map = {
        item["claim"]: item
        for item in retrieval_details
    }

    merged = []
    for item in e2e_details:
        extra = retrieval_map.get(item["claim"], {})
        merged.append({
            **item,
            "retrieval_rank": extra.get("retrieval_rank"),
            "rerank_rank": extra.get("rerank_rank"),
        })

    return merged


def main():
    parser = argparse.ArgumentParser(description="Phân tích lỗi")
    parser.add_argument(
        "--e2e",
        default=os.path.join(RESULTS_DIR, "e2e_results.json"),
    )
    parser.add_argument(
        "--nli",
        default=os.path.join(RESULTS_DIR, "nli_results.json"),
    )
    parser.add_argument(
        "--retrieval",
        default=os.path.join(RESULTS_DIR, "retrieval_results.json"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "error_analysis.json"),
    )
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    import json

    report = {}

    if os.path.exists(args.e2e):
        with open(args.e2e, encoding="utf-8") as file:
            e2e = json.load(file)

        details = e2e.get("details", [])
        if os.path.exists(args.retrieval):
            with open(args.retrieval, encoding="utf-8") as file:
                retrieval = json.load(file)
            details = merge_retrieval_into_e2e(details, retrieval.get("details", []))

        report["e2e"] = analyze_results(details, eval_type="e2e", max_examples=args.max_examples)

    if os.path.exists(args.nli):
        with open(args.nli, encoding="utf-8") as file:
            nli = json.load(file)
        report["nli"] = analyze_results(
            nli.get("details", []),
            eval_type="nli",
            max_examples=args.max_examples,
        )

    save_json(report, args.output)
    print(f"Đã lưu phân tích lỗi -> {args.output}")

    for section, data in report.items():
        print(f"\n[{section}] Tổng lỗi: {data['total_errors']}")
        for error_type, count in data["error_breakdown"].items():
            print(f"  - {error_type}: {count}")


if __name__ == "__main__":
    main()
