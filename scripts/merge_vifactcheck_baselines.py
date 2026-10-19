"""Gộp các file baseline_<method>.json (chạy riêng từng phương pháp) thành một file so sánh.

Usage: python scripts/merge_vifactcheck_baselines.py evaluation/results/vifactcheck_full
"""

import json
import os
import sys

METHODS = ["llm_only", "rag_llm", "rag_nli", "rag_nli_llm"]


def main():
    out_dir = sys.argv[1]
    merged = None
    for method in METHODS:
        path = os.path.join(out_dir, f"baseline_{method}.json")
        if not os.path.exists(path):
            print(f"[skip] thiếu {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if merged is None:
            merged = {k: data[k] for k in ("dataset", "collection")}
            merged["summary"], merged["methods"] = {}, {}
        merged["summary"][method] = data["summary"][method]
        merged["methods"][method] = data["methods"][method]

    if merged is None:
        raise SystemExit("Không có file baseline nào để gộp")

    output = os.path.join(out_dir, "baseline_comparison_vifactcheck.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"{'Phương pháp':<24} {'Accuracy':>10} {'F1-Macro':>10} {'Latency(s)':>12}")
    for row in merged["summary"].values():
        print(
            f"{row['method_label']:<24} {row['accuracy']:>10.4f} "
            f"{row['f1_macro']:>10.4f} {row['latency_mean_sec']:>12.2f}"
        )
    print(f"Đã lưu -> {output}")


if __name__ == "__main__":
    main()
