"""
Kiểm tra nhanh toàn bộ pipeline trước khi chạy đánh giá đầy đủ.

Chạy:
    python -m evaluation.quick_check
    python -m evaluation.quick_check --run-mini-eval
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import (
    PUBHEALTH_VI_PATH,
    RESULTS_DIR,
    VIFACTCHECK_HEALTH_PATH,
    VIET_MEDFACT_PATH,
)
from evaluation.io_utils import read_jsonl
from evaluation.metrics import find_article_rank
from ingest import get_collection_name

PASS = "OK"
FAIL = "FAIL"
WARN = "WARN"


class CheckResult:
    def __init__(self):
        self.items = []

    def add(self, status, name, detail=""):
        self.items.append((status, name, detail))
        icon = {"OK": "✓", "FAIL": "✗", "WARN": "!"}.get(status, "?")
        line = f"  [{icon}] {name}"
        if detail:
            line += f": {detail}"
        print(line)

    @property
    def failed(self):
        return any(s == FAIL for s, _, _ in self.items)

    @property
    def warnings(self):
        return sum(1 for s, _, _ in self.items if s == WARN)


def check_datasets(result):
    print("\n--- Dataset ---")
    for path, label in [
        (VIET_MEDFACT_PATH, "Viet-MedFact"),
        (VIFACTCHECK_HEALTH_PATH, "ViFactCheck-Health"),
        (PUBHEALTH_VI_PATH, "PUBHEALTH-Vi"),
    ]:
        if os.path.exists(path):
            count = sum(1 for _ in open(path, encoding="utf-8") if _.strip())
            result.add(PASS, label, f"{count} dòng — {path}")
        else:
            # Dataset B: chỉ FAIL nếu cả hai nguồn đều thiếu
            if label == "Viet-MedFact":
                result.add(FAIL, label, f"không tìm thấy {path}")
            else:
                result.add(WARN, label, f"không tìm thấy {path}")

    if not os.path.exists(VIFACTCHECK_HEALTH_PATH) and not os.path.exists(
        PUBHEALTH_VI_PATH
    ):
        result.add(
            FAIL,
            "Dataset B",
            "thiếu cả ViFactCheck-Health và PUBHEALTH-Vi — "
            "chạy: python -m evaluation.prepare_vifactcheck",
        )


def check_qdrant(result):
    print("\n--- Qdrant ---")
    from config import QDRANT_PATH
    from qdrant_client import QdrantClient

    if not os.path.exists(QDRANT_PATH):
        result.add(FAIL, "Qdrant DB", f"chưa có {QDRANT_PATH} — chạy: python ingest.py")
        return set()

    collection_name = get_collection_name()
    client = QdrantClient(path=QDRANT_PATH)

    try:
        info = client.get_collection(collection_name)
    except Exception as exc:
        result.add(
            FAIL,
            "Collection",
            f"{collection_name} không tồn tại ({exc}) — chạy: python ingest.py",
        )
        return set()

    result.add(PASS, "Collection", f"{collection_name}, {info.points_count} chunks")

    indexed_ids = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=500,
            offset=offset,
            with_payload=["article_id"],
        )
        for point in points:
            aid = point.payload.get("article_id")
            if aid:
                indexed_ids.add(aid)
        if offset is None:
            break

    bad_prefix = [aid for aid in indexed_ids if aid.startswith("article_")]
    if bad_prefix:
        result.add(
            FAIL,
            "article_id format",
            f"{len(bad_prefix)} chunk dùng 'article_*' (cũ) — cần chạy lại: python ingest.py",
        )
    else:
        skds_n = sum(1 for aid in indexed_ids if aid.startswith("skds_"))
        umc_n = sum(1 for aid in indexed_ids if aid.startswith("umc_"))
        result.add(PASS, "article_id format", f"skds={skds_n}, umc={umc_n} bài unique")

    return indexed_ids


def check_id_coverage(result, indexed_ids):
    print("\n--- Coverage (Viet-MedFact ↔ Qdrant) ---")

    if not os.path.exists(VIET_MEDFACT_PATH):
        result.add(WARN, "Coverage", "bỏ qua — chưa có viet_medfact.jsonl")
        return

    records = read_jsonl(VIET_MEDFACT_PATH)
    eval_ids = {r["article_id"] for r in records}

    if not indexed_ids:
        result.add(FAIL, "Coverage", "0% — Qdrant trống hoặc chưa index")
        return

    covered = eval_ids & indexed_ids
    missing = eval_ids - indexed_ids
    pct = len(covered) / len(eval_ids) * 100 if eval_ids else 0

    if pct >= 95:
        result.add(PASS, "Coverage", f"{len(covered)}/{len(eval_ids)} bài ({pct:.0f}%)")
    elif pct >= 50:
        result.add(WARN, "Coverage", f"{len(covered)}/{len(eval_ids)} bài ({pct:.0f}%)")
    else:
        result.add(
            FAIL,
            "Coverage",
            f"{len(covered)}/{len(eval_ids)} bài ({pct:.0f}%) — thiếu {len(missing)} bài",
        )

    if missing and len(missing) <= 5:
        result.add(WARN, "Thiếu", ", ".join(sorted(missing)))
    elif missing:
        sample = ", ".join(sorted(missing)[:5])
        result.add(WARN, "Thiếu (mẫu)", f"{sample}, ... (+{len(missing) - 5})")


def check_retrieval_smoke(result, n_samples=10):
    print(f"\n--- Retrieval smoke ({n_samples} mẫu) ---")

    if not os.path.exists(VIET_MEDFACT_PATH):
        result.add(WARN, "Retrieval smoke", "bỏ qua")
        return

    try:
        from retrieval import retrieve_with_metadata
    except Exception as exc:
        result.add(FAIL, "Retrieval import", str(exc))
        return

    records = read_jsonl(VIET_MEDFACT_PATH)[:n_samples]
    hits_count = 0
    ranks = []

    for row in records:
        hits = retrieve_with_metadata(row["claim"], limit=20)
        rank = find_article_rank(hits, row["article_id"])
        ranks.append(rank)
        if rank is not None:
            hits_count += 1

    hit_rate = hits_count / len(records) if records else 0
    detail = f"hit@20={hits_count}/{len(records)} ({hit_rate:.0%})"

    if hit_rate >= 0.5:
        result.add(PASS, "Retrieval smoke", detail)
    elif hit_rate > 0:
        result.add(WARN, "Retrieval smoke", detail)
    else:
        result.add(
            FAIL,
            "Retrieval smoke",
            f"{detail} — không tìm thấy bài nguồn nào (kiểm tra ingest + article_id)",
        )


def run_mini_eval(python, root, n_samples):
    print(f"\n--- Mini eval ({n_samples} mẫu / stage) ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    max_arg = ["--max-samples", str(n_samples)]

    steps = [
        ("eval_retrieval", [python, "-m", "evaluation.eval_retrieval", *max_arg]),
        ("eval_nli", [python, "-m", "evaluation.eval_nli", *max_arg]),
        ("eval_e2e", [python, "-m", "evaluation.eval_e2e", *max_arg]),
    ]

    failed = []
    for name, cmd in steps:
        print(f"\n>> {name}")
        proc = subprocess.run(cmd, cwd=root)
        if proc.returncode != 0:
            failed.append(name)

    return failed


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra nhanh pipeline đánh giá")
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Số mẫu cho retrieval smoke test (mặc định: 10)",
    )
    parser.add_argument(
        "--run-mini-eval",
        action="store_true",
        help="Chạy thêm eval_retrieval/nli/e2e với --max-samples",
    )
    parser.add_argument(
        "--mini-eval-samples",
        type=int,
        default=10,
        help="Số mẫu cho mini eval (mặc định: 10)",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    print("=" * 60)
    print("QUICK CHECK — kiểm tra pipeline trước khi đánh giá đầy đủ")
    print("=" * 60)

    result = CheckResult()
    check_datasets(result)
    indexed_ids = check_qdrant(result)
    check_id_coverage(result, indexed_ids)
    check_retrieval_smoke(result, n_samples=args.samples)

    mini_failed = []
    if args.run_mini_eval:
        mini_failed = run_mini_eval(
            sys.executable, root, args.mini_eval_samples
        )

    print("\n" + "=" * 60)
    if result.failed or mini_failed:
        print("KẾT QUẢ: CÓ LỖI — sửa trước khi chạy run_all")
        if mini_failed:
            print(f"  Mini eval thất bại: {', '.join(mini_failed)}")
        print("\nGợi ý:")
        print("  python ingest.py              # index lại toàn bộ SKDS + UMC")
        print("  python -m evaluation.quick_check --run-mini-eval")
        sys.exit(1)

    if result.warnings:
        print(f"KẾT QUẢ: OK với {result.warnings} cảnh báo — có thể chạy run_all")
    else:
        print("KẾT QUẢ: OK — sẵn sàng chạy python -m evaluation.run_all")
    print("=" * 60)


if __name__ == "__main__":
    main()
