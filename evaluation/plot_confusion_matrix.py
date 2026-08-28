"""Vẽ confusion matrix từ kết quả NLI."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import RESULTS_DIR


def plot_confusion_matrix(matrix, labels, output_path):
    arr = np.array(matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title("Confusion Matrix - NLI")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Vẽ confusion matrix NLI")
    parser.add_argument(
        "--input",
        default=os.path.join(RESULTS_DIR, "nli_results.json"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "nli_confusion_matrix.png"),
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as file:
        results = json.load(file)

    plot_confusion_matrix(
        results["confusion_matrix"],
        results["labels"],
        args.output,
    )
    print(f"Đã lưu biểu đồ -> {args.output}")


if __name__ == "__main__":
    main()
