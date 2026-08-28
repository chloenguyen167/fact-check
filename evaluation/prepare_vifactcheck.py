"""
Chuẩn bị tập ViFactCheck-Health (Dataset B, tiếng Việt sẵn) từ
https://huggingface.co/datasets/tranthaihoa/vifactcheck

Lọc topic sức khỏe / y tế, ánh xạ nhãn về true/false/unproven
(cùng schema với PUBHEALTH-Vi — không cần dịch).
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import (
    DATA_DIR,
    VIFACTCHECK_DATASET,
    VIFACTCHECK_HEALTH_PATH,
    VIFACTCHECK_HEALTH_TOPICS,
)
from evaluation.io_utils import write_jsonl
from evaluation.labels import VIFACTCHECK_ID_TO_LABEL
from evaluation.vifactcheck_utils import attach_article_id


def normalize_topic(topic: str) -> str:
    text = str(topic or "").strip().lower()
    # khoẻ (U+1ECFE) vs khỏe (U+1EC7E)
    return text.replace("khoẻ", "khỏe")


def is_health_topic(topic: str, allowed=None) -> bool:
    allowed = allowed or VIFACTCHECK_HEALTH_TOPICS
    return normalize_topic(topic) in {normalize_topic(t) for t in allowed}


def load_health_rows(splits=None):
    from datasets import load_dataset

    print(f"Đang tải {VIFACTCHECK_DATASET}...")
    ds = load_dataset(VIFACTCHECK_DATASET)
    splits = splits or ["train", "dev", "test"]

    rows = []
    topic_counter = Counter()
    for split_name in splits:
        if split_name not in ds:
            continue
        for row in ds[split_name]:
            topic = row.get("Topic", "")
            if not is_health_topic(topic):
                continue
            topic_counter[topic] += 1
            label_id = int(row["labels"])
            if label_id not in VIFACTCHECK_ID_TO_LABEL:
                continue
            rows.append({
                "claim": str(row.get("Statement") or "").strip(),
                "main_text": str(row.get("Context") or "").strip(),
                "explanation": str(row.get("Evidence") or "").strip(),
                "evidence": str(row.get("Evidence") or "").strip(),
                "label": VIFACTCHECK_ID_TO_LABEL[label_id],
                "labels_raw": label_id,
                "claim_id": f"vifactcheck_{row.get('annotation_id', row.get('index'))}",
                "topic": topic,
                "author": row.get("Author", ""),
                "url": row.get("Url", ""),
                "split": split_name,
            })

    print(f"Topic sức khỏe: {dict(topic_counter)}")
    print(f"Tổng mẫu health: {len(rows)}")
    return rows


def filter_by_split(rows, split):
    if not split or split == "all":
        return rows
    return [r for r in rows if r.get("split") == split]


def sample_balanced(rows, sample_size, seed=42):
    if not sample_size or sample_size >= len(rows):
        return rows

    import random

    rng = random.Random(seed)
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    labels = sorted(by_label)
    per_label = max(1, sample_size // len(labels))
    picked = []
    for label in labels:
        pool = list(by_label[label])
        rng.shuffle(pool)
        picked.extend(pool[: min(len(pool), per_label)])

    rng.shuffle(picked)
    return picked[: min(sample_size, len(picked))]


def main():
    parser = argparse.ArgumentParser(
        description="Chuẩn bị ViFactCheck-Health (lọc topic sức khỏe)"
    )
    parser.add_argument(
        "--output",
        default=VIFACTCHECK_HEALTH_PATH,
        help="Đường dẫn jsonl đầu ra",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "dev", "test"],
        default="all",
        help="Chỉ lấy split này (mặc định: all)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Giới hạn số mẫu (cân bằng nhãn). Mặc định: giữ toàn bộ health",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ghi đè file nếu đã tồn tại",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(args.output) and not args.force:
        print(
            f"Đã tồn tại {args.output}. "
            "Dùng --force để tạo lại."
        )
        return

    rows = load_health_rows()
    rows = filter_by_split(rows, args.split)
    if args.sample_size:
        rows = sample_balanced(rows, args.sample_size, seed=args.seed)

    # Bỏ mẫu thiếu claim/context
    rows = [r for r in rows if r["claim"] and r["main_text"]]
    rows = [attach_article_id(r) for r in rows]

    label_dist = Counter(r["label"] for r in rows)
    split_dist = Counter(r["split"] for r in rows)
    write_jsonl(args.output, rows)

    print(f"Nhãn: {dict(label_dist)}  (0=Supported→true, 1=Refuted→false, 2=NEI→unproven)")
    print(f"Split: {dict(split_dist)}")
    print(f"Đã lưu {len(rows)} mẫu -> {args.output}")
    print(
        "Dùng cho NLI/LLM:\n"
        f"  python -m evaluation.eval_nli --input {args.output}\n"
        f"  python -m evaluation.eval_llm --input {args.output}\n"
        "RAG baseline (sau khi index):\n"
        "  python -m evaluation.index_vifactcheck --recreate\n"
        "  python -m evaluation.eval_baselines --dataset vifactcheck --max-samples 100"
    )


if __name__ == "__main__":
    main()
