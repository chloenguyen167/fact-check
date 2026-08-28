"""
So sánh phương pháp fact-check:
1. LLM-only
2. RAG + LLM
3. RAG + NLI (pipeline đề xuất — verdict)
4. RAG + NLI + LLM (pipeline đầy đủ — verdict + giải thích; Acc giống RAG+NLI)

Dataset:
  - viet_medfact (A): claim từ SKDS/UMC, corpus medical_articles_bge_m3
  - vifactcheck (B): ViFactCheck-Health, corpus vifactcheck_health_bge_m3 (cần index trước)
"""

import argparse
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import (
    EVAL_MAX_SAMPLES,
    RESULTS_DIR,
    VIFACTCHECK_BASELINE_RESULTS,
    VIFACTCHECK_HEALTH_PATH,
    VIET_MEDFACT_PATH,
)
from evaluation.io_utils import read_jsonl
from evaluation.labels import VERDICTS, normalize_verdict
from evaluation.metrics import classification_metrics, save_json, summarize_latency
from evaluation.vifactcheck_utils import vifactcheck_collection_name
from config import EMBEDDING_MODEL
from pipeline import verify_claim, verify_claim_llm_only, verify_claim_rag_llm

METHODS = {
    "llm_only": {
        "label": "LLM-only",
        "runner": verify_claim_llm_only,
        "needs_qdrant": False,
    },
    "rag_llm": {
        "label": "RAG + LLM",
        "runner": verify_claim_rag_llm,
        "needs_qdrant": True,
    },
    "rag_nli": {
        "label": "RAG + NLI",
        "runner": lambda claim: verify_claim(claim, with_explanation=False, use_cache=False),
        "needs_qdrant": True,
    },
    "rag_nli_llm": {
        "label": "RAG + NLI + LLM",
        "runner": lambda claim: verify_claim(claim, with_explanation=True, use_cache=False),
        "needs_qdrant": True,
    },
}

DATASET_CONFIG = {
    "viet_medfact": {
        "path": VIET_MEDFACT_PATH,
        "default_output": os.path.join(RESULTS_DIR, "baseline_comparison.json"),
        "collection": None,
        "prepare_hint": "python -m evaluation.generate_viet_medfact",
        "index_hint": None,
    },
    "vifactcheck": {
        "path": VIFACTCHECK_HEALTH_PATH,
        "default_output": VIFACTCHECK_BASELINE_RESULTS,
        "collection": vifactcheck_collection_name(EMBEDDING_MODEL),
        "prepare_hint": "python -m evaluation.prepare_vifactcheck --force",
        "index_hint": "python -m evaluation.index_vifactcheck --recreate",
    },
}


def _claim_field(row):
    return row.get("claim") or row.get("Statement") or ""


def _configure_retrieval(dataset_key):
    from retrieval import set_retrieval_collection

    cfg = DATASET_CONFIG[dataset_key]
    name = cfg["collection"]
    set_retrieval_collection(name)
    return name


def _ensure_vifactcheck_index(collection_name):
    from qdrant_store import close_qdrant_client, get_qdrant_client

    client = get_qdrant_client()
    try:
        info = client.get_collection(collection_name)
        if info.points_count == 0:
            raise RuntimeError(
                f"Collection {collection_name} rỗng. "
                f"Chạy: python -m evaluation.index_vifactcheck --recreate"
            )
        print(f"ViFactCheck RAG index: {collection_name} ({info.points_count} chunks)")
    finally:
        close_qdrant_client()


def evaluate_method(records, method_key):
    runner = METHODS[method_key]["runner"]
    y_true = []
    y_pred = []
    latencies = []
    details = []

    for row in tqdm(records, desc=f"Đánh giá {METHODS[method_key]['label']}"):
        gold = normalize_verdict(row.get("label"))
        if gold is None:
            continue

        claim = _claim_field(row)
        start = time.perf_counter()
        result = runner(claim)
        elapsed = time.perf_counter() - start

        y_true.append(gold)
        y_pred.append(result["verdict"])
        latencies.append(elapsed)

        details.append({
            "claim": claim,
            "label_gold": gold,
            "label_pred": result["verdict"],
            "article_id": row.get("article_id"),
            "claim_id": row.get("claim_id", ""),
            "url": row.get("url", ""),
            "latency_sec": elapsed,
            "llm_raw": result.get("llm_raw"),
            "verdict_score": result.get("verdict_score"),
            "explanation_preview": (result.get("explanation") or "")[:300],
            "evidence_preview": result.get("evidence", "")[:300],
        })

    metrics = classification_metrics(y_true, y_pred, VERDICTS)
    metrics["latency"] = summarize_latency(latencies)
    metrics["method"] = method_key
    metrics["method_label"] = METHODS[method_key]["label"]
    metrics["num_samples"] = len(y_true)
    metrics["details"] = details
    return metrics


def _build_comparison(results, dataset_key, collection_name):
    summary = {
        method_key: {
            "method_label": results[method_key]["method_label"],
            "accuracy": results[method_key]["accuracy"],
            "f1_macro": results[method_key]["f1_macro"],
            "precision_macro": results[method_key]["precision_macro"],
            "recall_macro": results[method_key]["recall_macro"],
            "latency_mean_sec": results[method_key]["latency"]["mean_sec"],
            "num_samples": results[method_key]["num_samples"],
        }
        for method_key in results
    }
    return {
        "dataset": dataset_key,
        "collection": collection_name,
        "summary": summary,
        "methods": results,
    }


def run_comparison(records, methods, dataset_key, output_path=None):
    from retrieval import set_retrieval_collection

    collection_name = _configure_retrieval(dataset_key)

    try:
        if dataset_key == "vifactcheck" and collection_name:
            _ensure_vifactcheck_index(collection_name)

        results = {}
        for method_key in methods:
            meta = METHODS[method_key]
            if meta.get("needs_qdrant"):
                from qdrant_store import get_qdrant_client

                print(
                    f"\n[{method_key}] Qdrant collection: "
                    f"{collection_name or 'medical_articles (mặc định)'}"
                )
                get_qdrant_client()

            results[method_key] = evaluate_method(records, method_key)

            if output_path:
                payload = _build_comparison(results, dataset_key, collection_name)
                save_json(payload, output_path)
                print(f"  Đã lưu tạm -> {output_path} ({', '.join(results)})")

        return _build_comparison(results, dataset_key, collection_name)
    finally:
        set_retrieval_collection(None)


def print_summary(summary):
    print("\n=== So sánh phương pháp ===")
    print(f"{'Phương pháp':<24} {'Accuracy':>10} {'F1-Macro':>10} {'Latency(s)':>12}")
    print("-" * 60)
    for method_key, row in summary.items():
        print(
            f"{row['method_label']:<24} "
            f"{row['accuracy']:>10.4f} "
            f"{row['f1_macro']:>10.4f} "
            f"{row['latency_mean_sec']:>12.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="So sánh LLM-only / RAG+LLM / RAG+NLI / RAG+NLI+LLM"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIG.keys()),
        default="viet_medfact",
        help="viet_medfact (A) hoặc vifactcheck (B, cần index_vifactcheck trước)",
    )
    parser.add_argument("--input", default=None, help="Ghi đè đường dẫn jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=EVAL_MAX_SAMPLES)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS.keys()),
        default=["llm_only", "rag_llm", "rag_nli", "rag_nli_llm"],
        help="Chọn phương pháp (rag_nli_llm chậm do sinh giải thích)",
    )
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    input_path = args.input or cfg["path"]
    output_path = args.output or cfg["default_output"]

    if not os.path.exists(input_path):
        hints = [cfg["prepare_hint"]]
        if cfg.get("index_hint"):
            hints.append(cfg["index_hint"])
        raise FileNotFoundError(
            f"Chưa có {input_path}. Chạy:\n  " + "\n  ".join(h for h in hints if h)
        )

    records = read_jsonl(input_path)
    if args.dataset == "vifactcheck" and records and not records[0].get("article_id"):
        raise ValueError(
            "vifactcheck_health.jsonl thiếu article_id. "
            "Chạy: python -m evaluation.prepare_vifactcheck --force"
        )

    if args.max_samples:
        records = records[: args.max_samples]

    print(f"Dataset: {args.dataset} ({len(records)} mẫu)")
    comparison = run_comparison(
        records, args.methods, args.dataset, output_path=output_path
    )
    print_summary(comparison["summary"])
    print(f"\nĐã lưu -> {output_path}")


if __name__ == "__main__":
    main()
