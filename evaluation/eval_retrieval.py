"""
Đánh giá Retrieval & Reranking trên Viet-MedFact hoặc ViFactCheck-Health.

Metrics: Hit@K, MRR
Ablation: embedding-only vs embedding + reranker
"""

import argparse
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBEDDING_MODEL
from evaluation.config_eval import (
    EVAL_MAX_SAMPLES,
    RESULTS_DIR,
    RETRIEVAL_K_VALUES,
    RERANK_K_VALUES,
    VIFACTCHECK_HEALTH_PATH,
    VIET_MEDFACT_PATH,
)
from evaluation.io_utils import read_jsonl
from evaluation.metrics import (
    aggregate_retrieval_results,
    find_article_rank,
    save_json,
)
from evaluation.vifactcheck_utils import vifactcheck_collection_name
from retrieval import retrieve_with_metadata, set_retrieval_collection
from reranker import rerank_records

DATASETS = {
    "viet_medfact": {
        "path": VIET_MEDFACT_PATH,
        "output": os.path.join(RESULTS_DIR, "retrieval_results.json"),
        "collection": None,
    },
    "vifactcheck": {
        "path": VIFACTCHECK_HEALTH_PATH,
        "output": os.path.join(RESULTS_DIR, "retrieval_results_vifactcheck.json"),
        "collection": vifactcheck_collection_name(EMBEDDING_MODEL),
    },
}


def evaluate_sample(claim, article_id):
    retrieval_hits = retrieve_with_metadata(claim, limit=max(RETRIEVAL_K_VALUES))
    retrieval_rank = find_article_rank(retrieval_hits, article_id)

    reranked = rerank_records(claim, retrieval_hits, top_n=len(retrieval_hits))
    rerank_rank = find_article_rank(reranked, article_id)

    return {
        "retrieval_only": retrieval_rank,
        "with_reranker": rerank_rank,
        "retrieval_hits": retrieval_hits[:5],
        "rerank_hits": reranked[:3],
    }


def run_evaluation(records):
    per_sample = {
        "retrieval_only": [],
        "with_reranker": [],
    }
    details = []

    for row in tqdm(records, desc="Đánh giá retrieval"):
        result = evaluate_sample(row["claim"], row["article_id"])

        per_sample["retrieval_only"].append({"rank": result["retrieval_only"]})
        per_sample["with_reranker"].append({"rank": result["with_reranker"]})

        details.append({
            "claim": row["claim"],
            "article_id": row["article_id"],
            "label": row.get("label"),
            "retrieval_rank": result["retrieval_only"],
            "rerank_rank": result["with_reranker"],
        })

    summary = aggregate_retrieval_results(per_sample, RETRIEVAL_K_VALUES)

    rerank_summary = {}
    for mode, key in [("retrieval_only", "retrieval_rank"), ("with_reranker", "rerank_rank")]:
        ranks = [item[key] for item in details]
        rerank_summary[mode] = {
            f"hit@{k}": aggregate_retrieval_results(
                {mode: [{"rank": r} for r in ranks]},
                [k],
            )[mode][f"hit@{k}"]
            for k in RERANK_K_VALUES
        }

    return {
        "retrieval_metrics": summary,
        "rerank_hit_at_k": rerank_summary,
        "retrieval_k_values": RETRIEVAL_K_VALUES,
        "rerank_k_values": RERANK_K_VALUES,
        "num_samples": len(records),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Đánh giá Retrieval & Reranking")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="viet_medfact",
    )
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=EVAL_MAX_SAMPLES)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    input_path = args.input or cfg["path"]
    output_path = args.output or cfg["output"]

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Chưa có {input_path}")

    records = read_jsonl(input_path)
    if args.max_samples:
        records = records[: args.max_samples]

    if args.dataset == "vifactcheck" and records and not records[0].get("article_id"):
        raise ValueError(
            "Thiếu article_id — chạy: python -m evaluation.prepare_vifactcheck --force"
        )

    set_retrieval_collection(cfg["collection"])
    try:
        results = run_evaluation(records)
        results["dataset"] = args.dataset
        results["collection"] = cfg["collection"]
        save_json(results, output_path)
    finally:
        set_retrieval_collection(None)

    print("\n=== Retrieval (embedding only) ===")
    for key, value in results["retrieval_metrics"]["retrieval_only"].items():
        print(f"  {key}: {value:.4f}")

    print("\n=== Retrieval + Reranker ===")
    for key, value in results["retrieval_metrics"]["with_reranker"].items():
        print(f"  {key}: {value:.4f}")

    print(f"\nĐã lưu chi tiết -> {output_path}")


if __name__ == "__main__":
    main()
