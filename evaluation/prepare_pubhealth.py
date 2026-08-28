"""
Chuẩn bị tập PUBHEALTH-Vi (Dataset B) từ
https://huggingface.co/datasets/ImperialCollegeLondon/health_fact

Tải PUBHEALTH (Google Drive), lấy mẫu cân bằng nhãn, dịch sang tiếng Việt.
"""

import argparse
import os
import sys
import zipfile

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import (
    DATA_DIR,
    LLM_REQUEST_DELAY,
    PUBHEALTH_LABELS,
    PUBHEALTH_RAW_PATH,
    PUBHEALTH_SAMPLE_SIZE,
    PUBHEALTH_VI_PATH,
)
from evaluation.io_utils import read_jsonl, write_jsonl
from evaluation.llm_utils import translate_to_vietnamese

PUBHEALTH_DRIVE_ID = "1eTtRs5cUlBP5dXsx-FTAlmXuB6JQi2qj"
PUBHEALTH_CACHE_DIR = os.path.join(DATA_DIR, "pubhealth_cache")
VALID_LABELS = {"true", "false", "unproven", "mixture"}


def download_pubhealth_archive():
    os.makedirs(PUBHEALTH_CACHE_DIR, exist_ok=True)
    train_path = os.path.join(PUBHEALTH_CACHE_DIR, "PUBHEALTH", "train.tsv")

    if os.path.exists(train_path):
        return

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Cần cài gdown để tải PUBHEALTH: pip install gdown"
        ) from exc

    zip_path = os.path.join(PUBHEALTH_CACHE_DIR, "pubhealth.zip")
    if not os.path.exists(zip_path):
        print("Đang tải PUBHEALTH từ Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={PUBHEALTH_DRIVE_ID}",
            zip_path,
            quiet=False,
        )

    print("Đang giải nén PUBHEALTH...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(PUBHEALTH_CACHE_DIR)


def read_pubhealth_tsv(path):
    df = pd.read_csv(path, sep="\t", dtype=str)

    # test.tsv có thêm cột rỗng ở đầu
    if "claim_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "drop_col"})
        if "claim_id" not in df.columns and len(df.columns) > 1:
            df = df.iloc[:, 1:]
            df.columns = [
                "claim_id", "claim", "date_published", "explanation",
                "fact_checkers", "main_text", "sources", "label", "subjects",
            ]

    df = df.fillna("")
    df = df[df["label"].isin(VALID_LABELS)].copy()
    return df


def load_pubhealth_df():
    download_pubhealth_archive()
    base = os.path.join(PUBHEALTH_CACHE_DIR, "PUBHEALTH")

    frames = []
    for name in ("train.tsv", "dev.tsv", "test.tsv"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            frames.append(read_pubhealth_tsv(path))

    if not frames:
        raise FileNotFoundError(
            f"Không tìm thấy TSV trong {base}. Chạy lại để tải dữ liệu."
        )

    df = pd.concat(frames, ignore_index=True)
    print(f"Đã tải {len(df)} mẫu PUBHEALTH (true/false/unproven/mixture)")
    return df


def sample_balanced(df, sample_size, labels, seed=42):
    per_label = max(1, sample_size // len(labels))
    parts = []

    for label in labels:
        subset = df[df["label"] == label]
        count = min(len(subset), per_label)
        if count > 0:
            parts.append(subset.sample(n=count, random_state=seed))

    if not parts:
        raise ValueError("Không lấy được mẫu nào từ dataset PUBHEALTH")

    merged = pd.concat(parts, ignore_index=True)
    return merged.sample(frac=1, random_state=seed).head(min(sample_size, len(merged)))


def export_raw(df, output_path):
    records = []
    for index, row in df.iterrows():
        records.append({
            "claim": row["claim"],
            "main_text": row["main_text"],
            "explanation": row.get("explanation", ""),
            "label": row["label"],
            "claim_id": row.get("claim_id") or f"pubhealth_{index}",
        })

    write_jsonl(output_path, records)
    return records


def translate_records(
    records,
    output_path,
    provider="auto",
    request_delay=None,
    existing=None,
):
    translated = list(existing or [])
    done_ids = {r.get("claim_id") for r in translated}

    pending = []
    for index, row in enumerate(records):
        claim_id = row.get("claim_id", f"pubhealth_{index}")
        if claim_id not in done_ids:
            pending.append((index, row))

    if pending:
        print(
            f"Tiếp tục dịch: đã có {len(translated)}/{len(records)}, "
            f"còn {len(pending)} mẫu"
        )
    else:
        print(f"Đã dịch đủ {len(translated)}/{len(records)} mẫu.")
        return translated

    for index, row in tqdm(pending, desc="Dịch PUBHEALTH"):
        try:
            claim_vi = translate_to_vietnamese(
                row["claim"], "claim", provider=provider, request_delay=request_delay
            )
            main_text_vi = translate_to_vietnamese(
                row["main_text"][:4000], "main_text",
                provider=provider, request_delay=request_delay,
            )
            explanation_vi = ""
            if row.get("explanation"):
                explanation_vi = translate_to_vietnamese(
                    row["explanation"][:3000], "explanation",
                    provider=provider, request_delay=request_delay,
                )

            translated.append({
                "claim": claim_vi,
                "main_text": main_text_vi,
                "explanation": explanation_vi,
                "label": row["label"],
                "claim_en": row["claim"],
                "main_text_en": row["main_text"],
                "explanation_en": row.get("explanation", ""),
                "claim_id": row.get("claim_id", f"pubhealth_{index}"),
            })
            write_jsonl(output_path, translated)
        except Exception as exc:
            print(f"Lỗi dịch mẫu {index} ({row.get('claim_id')}): {exc}")

    write_jsonl(output_path, translated)
    return translated


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị PUBHEALTH-Vi")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=PUBHEALTH_SAMPLE_SIZE,
    )
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Chỉ tải và lưu bản gốc tiếng Anh",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ghi đè pubhealth_vi.jsonl nếu đã tồn tại",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục dịch từ pubhealth_raw.jsonl, bỏ qua mẫu đã có trong pubhealth_vi.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--provider",
        choices=["auto", "ollama", "openrouter", "gemini"],
        default="auto",
    )
    parser.add_argument("--delay", type=float, default=LLM_REQUEST_DELAY)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.resume:
        if not os.path.exists(PUBHEALTH_RAW_PATH):
            print(
                f"Không tìm thấy {PUBHEALTH_RAW_PATH}. "
                "Chạy lệnh không có --resume trước để tạo file gốc."
            )
            return

        raw_records = read_jsonl(PUBHEALTH_RAW_PATH)
        existing = (
            read_jsonl(PUBHEALTH_VI_PATH)
            if os.path.exists(PUBHEALTH_VI_PATH)
            else []
        )
        print(f"Đọc {len(raw_records)} mẫu gốc, {len(existing)} mẫu đã dịch")

        translated = translate_records(
            raw_records,
            PUBHEALTH_VI_PATH,
            provider=args.provider,
            request_delay=args.delay,
            existing=existing,
        )
        print(f"Đã lưu {len(translated)} mẫu tiếng Việt -> {PUBHEALTH_VI_PATH}")
        return

    if os.path.exists(PUBHEALTH_VI_PATH) and not args.skip_translate and not args.force:
        print(
            f"Đã tồn tại {PUBHEALTH_VI_PATH}. "
            "Dùng --resume để dịch tiếp hoặc --force để tạo lại từ đầu."
        )
        return

    df = load_pubhealth_df()
    sampled = sample_balanced(
        df,
        sample_size=args.sample_size,
        labels=PUBHEALTH_LABELS,
        seed=args.seed,
    )

    raw_records = export_raw(sampled, PUBHEALTH_RAW_PATH)
    print(f"Đã lưu {len(raw_records)} mẫu gốc -> {PUBHEALTH_RAW_PATH}")

    if args.skip_translate:
        return

    translated = translate_records(
        raw_records,
        PUBHEALTH_VI_PATH,
        provider=args.provider,
        request_delay=args.delay,
    )
    print(f"Đã lưu {len(translated)} mẫu tiếng Việt -> {PUBHEALTH_VI_PATH}")


if __name__ == "__main__":
    main()
