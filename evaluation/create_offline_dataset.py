"""
Tạo tập Viet-MedFact mẫu nhanh (không cần LLM) để thử pipeline đánh giá.

Dùng heuristic đơn giản từ tiêu đề bài viết.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import VIET_MEDFACT_PATH
from evaluation.io_utils import write_jsonl
from evaluation.labels import VERDICT_DUNG, VERDICT_KHONG_DU, VERDICT_SAI


def load_articles():
    df = pd.read_csv("data/articles_skds.csv")[["title", "content", "source"]]
    df["article_id"] = [f"skds_{i}" for i in df.index]
    return df


def build_offline_claims(row):
    title = str(row["title"]).strip()
    return [
        {
            "claim": title,
            "label": VERDICT_DUNG,
            "article_id": row["article_id"],
            "source": row.get("source", ""),
            "title": title,
        },
        {
            "claim": f"{title} hoàn toàn sai và nguy hiểm",
            "label": VERDICT_SAI,
            "article_id": row["article_id"],
            "source": row.get("source", ""),
            "title": title,
        },
        {
            "claim": f"Uống thuốc X có thể chữa được bệnh liên quan đến {title[:40]}",
            "label": VERDICT_KHONG_DU,
            "article_id": row["article_id"],
            "source": row.get("source", ""),
            "title": title,
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Tạo Viet-MedFact offline")
    parser.add_argument("--num-articles", type=int, default=30)
    parser.add_argument("--output", default=VIET_MEDFACT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_articles()
    sample = df.sample(n=min(args.num_articles, len(df)), random_state=args.seed)

    records = []
    for _, row in sample.iterrows():
        records.extend(build_offline_claims(row))

    write_jsonl(args.output, records)
    print(f"Đã tạo {len(records)} claim offline -> {args.output}")


if __name__ == "__main__":
    main()
