"""
Chạy toàn bộ pipeline đánh giá theo evaluation.md

Các bước:
1. Chuẩn bị dữ liệu (Viet-MedFact, Dataset B: ViFactCheck-Health / PUBHEALTH-Vi)
2. Đánh giá Retrieval / Reranking
3. Đánh giá NLI
4. Đánh giá LLM explanation
5. Đánh giá End-to-End
6. Phân tích lỗi
"""

import argparse
import os
import subprocess
import sys

from evaluation.config_eval import (
    DATASET_B_DEFAULT,
    PUBHEALTH_VI_PATH,
    VIFACTCHECK_HEALTH_PATH,
    VIET_MEDFACT_PATH,
)
from evaluation.labels import dataset_b_path


def run_step(description, command):
    print(f"\n{'=' * 60}")
    print(description)
    print("=" * 60)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Bước thất bại: {description}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Chạy toàn bộ đánh giá")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Dùng dataset Viet-MedFact offline (không cần LLM sinh claim)",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Bỏ qua bước chuẩn bị dữ liệu (dùng file đã có)",
    )
    parser.add_argument(
        "--force-data-prep",
        action="store_true",
        help="Sinh lại dataset dù đã có file jsonl",
    )
    parser.add_argument(
        "--skip-llm-eval",
        action="store_true",
        help="Bỏ qua đánh giá LLM explanation (tiết kiệm API)",
    )
    parser.add_argument(
        "--dataset-b",
        choices=["vifactcheck", "pubhealth"],
        default=DATASET_B_DEFAULT,
        help="Dataset B cho NLI/LLM (mặc định: vifactcheck — tiếng Việt, không dịch)",
    )
    parser.add_argument(
        "--skip-pubhealth",
        action="store_true",
        help="Alias: bỏ qua Dataset B (NLI/LLM). Giữ để tương thích lệnh cũ.",
    )
    parser.add_argument(
        "--skip-dataset-b",
        action="store_true",
        help="Bỏ qua chuẩn bị và đánh giá Dataset B (NLI/LLM)",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--skip-quick-check",
        action="store_true",
        help="Bỏ qua quick_check trước khi đánh giá",
    )
    args = parser.parse_args()

    skip_dataset_b = args.skip_dataset_b or args.skip_pubhealth
    dataset_b = args.dataset_b
    dataset_b_file = dataset_b_path(dataset_b)

    python = sys.executable
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    max_arg = ["--max-samples", str(args.max_samples)] if args.max_samples else []

    need_viet_medfact = not os.path.exists(VIET_MEDFACT_PATH)
    need_dataset_b = not skip_dataset_b and not os.path.exists(dataset_b_file)
    skip_data_prep = args.skip_data_prep or (
        not args.force_data_prep and not need_viet_medfact and not need_dataset_b
    )

    if skip_data_prep:
        if args.skip_data_prep or args.force_data_prep:
            reason = "theo --skip-data-prep"
        else:
            reason = "đã có dataset"
        print(f"\nBỏ qua chuẩn bị dữ liệu ({reason}).")
        if os.path.exists(VIET_MEDFACT_PATH):
            print(f"  - {VIET_MEDFACT_PATH}")
        if os.path.exists(dataset_b_file):
            print(f"  - {dataset_b_file}")
    else:
        if need_viet_medfact:
            if args.offline:
                run_step(
                    "Tạo Viet-MedFact offline",
                    [python, "-m", "evaluation.create_offline_dataset"],
                )
            else:
                run_step(
                    "Sinh Viet-MedFact bằng LLM",
                    [python, "-m", "evaluation.generate_viet_medfact"],
                )
        else:
            print(f"\nĐã có {VIET_MEDFACT_PATH}, bỏ qua sinh Viet-MedFact.")

        if not skip_dataset_b:
            if need_dataset_b or args.force_data_prep:
                if dataset_b == "vifactcheck":
                    cmd = [python, "-m", "evaluation.prepare_vifactcheck"]
                    if args.force_data_prep:
                        cmd.append("--force")
                    run_step("Chuẩn bị ViFactCheck-Health", cmd)
                else:
                    cmd = [python, "-m", "evaluation.prepare_pubhealth"]
                    if args.force_data_prep:
                        cmd.append("--force")
                    run_step("Chuẩn bị PUBHEALTH-Vi", cmd)
            else:
                print(f"\nĐã có {dataset_b_file}, bỏ qua chuẩn bị Dataset B.")

    if not args.skip_quick_check:
        run_step(
            "Quick check (kiểm tra lỗi trước khi đánh giá)",
            [python, "-m", "evaluation.quick_check", "--samples", "10"],
        )

    run_step(
        "Đánh giá Retrieval & Reranking",
        [python, "-m", "evaluation.eval_retrieval", *max_arg],
    )

    if not skip_dataset_b:
        run_step(
            "Đánh giá NLI",
            [
                python,
                "-m",
                "evaluation.eval_nli",
                "--dataset-b",
                dataset_b,
                *max_arg,
            ],
        )

        if not args.skip_llm_eval:
            llm_max = ["--max-samples", str(args.max_samples or 20)]
            run_step(
                "Đánh giá LLM explanation",
                [
                    python,
                    "-m",
                    "evaluation.eval_llm",
                    "--dataset-b",
                    dataset_b,
                    *llm_max,
                ],
            )

    run_step(
        "Đánh giá End-to-End",
        [python, "-m", "evaluation.eval_e2e", *max_arg],
    )

    baseline_max = ["--max-samples", str(args.max_samples or 20)]
    run_step(
        "So sánh baseline (LLM-only / RAG+LLM / RAG+NLI)",
        [python, "-m", "evaluation.eval_baselines", *baseline_max],
    )

    run_step(
        "Phân tích lỗi",
        [python, "-m", "evaluation.error_analysis"],
    )

    if not skip_dataset_b and os.path.exists(
        os.path.join(root, "evaluation", "results", "nli_results.json")
    ):
        run_step(
            "Vẽ confusion matrix NLI",
            [python, "-m", "evaluation.plot_confusion_matrix"],
        )

    print("\nHoàn tất! Kết quả nằm trong evaluation/results/")
    print(f"Dataset B đã dùng: {dataset_b} -> {dataset_b_file}")
    if dataset_b == "pubhealth" and not os.path.exists(PUBHEALTH_VI_PATH):
        print(f"(chưa có {PUBHEALTH_VI_PATH})")
    if dataset_b == "vifactcheck" and not os.path.exists(VIFACTCHECK_HEALTH_PATH):
        print(f"(chưa có {VIFACTCHECK_HEALTH_PATH})")


if __name__ == "__main__":
    main()
