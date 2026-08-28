"""
Sinh tập dữ liệu Viet-MedFact (Dataset A) bằng LLM.

Mỗi bài viết sinh 3 claim: ĐÚNG / SAI / KHÔNG ĐỦ BẰNG CHỨNG,
kèm article_id để đánh giá Retrieval.
"""

import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import (
    DATA_DIR,
    LLM_REQUEST_DELAY,
    VIET_MEDFACT_CLAIMS_PER_ARTICLE,
    VIET_MEDFACT_NUM_ARTICLES,
    VIET_MEDFACT_NUM_SKDS,
    VIET_MEDFACT_NUM_UMC,
    VIET_MEDFACT_PATH,
)
from evaluation.io_utils import read_jsonl, write_jsonl
from evaluation.labels import VERDICT_DUNG, VERDICT_KHONG_DU, VERDICT_SAI
from evaluation.llm_utils import chat_completion, extract_json_block

REQUIRED_LABELS = {VERDICT_DUNG, VERDICT_SAI, VERDICT_KHONG_DU}
MIN_CLAIM_LENGTH = 10
MAX_GENERATION_ATTEMPTS = 3

LABEL_ALIASES = {
    "DUNG": VERDICT_DUNG,
    "ĐÚNG": VERDICT_DUNG,
    "TRUE": VERDICT_DUNG,
    "SAI": VERDICT_SAI,
    "FALSE": VERDICT_SAI,
    "KHONG DU": VERDICT_KHONG_DU,
    "KHÔNG ĐỦ BẰNG CHỨNG": VERDICT_KHONG_DU,
    "KHONG DU BANG CHUNG": VERDICT_KHONG_DU,
    "NEUTRAL": VERDICT_KHONG_DU,
}


def load_articles():
    frames = []

    skds_path = "data/articles_skds.csv"
    if os.path.exists(skds_path):
        df_skds = pd.read_csv(skds_path)[["title", "content", "source"]]
        df_skds["article_id"] = [f"skds_{i}" for i in df_skds.index]
        df_skds["source_group"] = "skds"
        frames.append(df_skds)

    umc_path = "data/articles.csv"
    if os.path.exists(umc_path):
        df_umc = pd.read_csv(umc_path, header=None)
        df_umc.columns = [
            "id", "title", "publish_time", "content", "source", "link",
        ]
        df_umc = df_umc[["title", "content", "source"]]
        df_umc["article_id"] = [f"umc_{i}" for i in df_umc.index]
        df_umc["source_group"] = "umc"
        frames.append(df_umc)

    if not frames:
        raise FileNotFoundError("Không tìm thấy data/articles_skds.csv hoặc data/articles.csv")

    return pd.concat(frames, ignore_index=True)


def sample_articles(df, num_skds, num_umc, seed=42):
    """Lấy mẫu cân bằng theo nguồn SKDS / UMC (umcclinic)."""
    parts = []

    skds_df = df[df["source_group"] == "skds"]
    umc_df = df[df["source_group"] == "umc"]

    if num_skds > 0 and len(skds_df) > 0:
        parts.append(
            skds_df.sample(n=min(num_skds, len(skds_df)), random_state=seed)
        )

    if num_umc > 0 and len(umc_df) > 0:
        parts.append(
            umc_df.sample(n=min(num_umc, len(umc_df)), random_state=seed)
        )

    if not parts:
        return df.sample(n=min(num_skds + num_umc, len(df)), random_state=seed)

    return pd.concat(parts, ignore_index=True)


def normalize_label(raw_label):
    label = str(raw_label).strip().upper()
    if label in REQUIRED_LABELS:
        return label
    return LABEL_ALIASES.get(label)


def build_prompt(title, content):
    article_text = f"{title}\n\n{content}"[:6000]

    return f"""
Bạn đóng vai người dùng mạng xã hội đọc bài viết y khoa và đưa ra phát biểu (claim).

Đọc bài viết sau và sinh đúng {VIET_MEDFACT_CLAIMS_PER_ARTICLE} claim tiếng Việt.

QUAN TRỌNG: Cả 3 claim đều BẮT BUỘC phải có nội dung, không được để trống.

1. label "ĐÚNG": Một phát biểu CHÍNH XÁC, bám sát nội dung bài viết.
2. label "SAI": Một phát biểu SAI LỆCH, cố tình bóp méo hoặc đảo ngược thông tin trong bài (ví dụ bài nói A có lợi -> claim nói A gây ung thư). Đây là dữ liệu nghiên cứu fact-checking, phải viết claim SAI cụ thể.
3. label "KHÔNG ĐỦ BẰNG CHỨNG": Chủ đề liên quan nhưng bài viết không đủ căn cứ xác nhận hay bác bỏ.

Yêu cầu:
- Mỗi claim 1-2 câu, tự nhiên như lời đồn trên mạng.
- Không copy nguyên văn tiêu đề.
- Trường "claim" không được rỗng.
- Trả về JSON thuần, không markdown:

{{
  "claims": [
    {{"label": "ĐÚNG", "claim": "nội dung claim đúng"}},
    {{"label": "SAI", "claim": "nội dung claim sai"}},
    {{"label": "KHÔNG ĐỦ BẰNG CHỨNG", "claim": "nội dung claim không đủ bằng chứng"}}
  ]
}}

Bài viết:
{article_text}
"""


def build_single_label_prompt(title, content, label):
    article_text = f"{title}\n\n{content}"[:6000]

    if label == VERDICT_DUNG:
        instruction = (
            "Viết MỘT phát biểu tiếng Việt CHÍNH XÁC, bám sát nội dung bài viết."
        )
    elif label == VERDICT_SAI:
        instruction = (
            "Viết MỘT phát biểu tiếng Việt SAI LỆCH, cố tình bóp méo thông tin quan trọng "
            "trong bài (tin giả trên mạng). Phải có nội dung cụ thể, không được để trống."
        )
    else:
        instruction = (
            "Viết MỘT phát biểu tiếng Việt về chủ đề liên quan nhưng bài viết "
            "KHÔNG đủ căn cứ để xác nhận hay bác bỏ."
        )

    return f"""
{instruction}

Chỉ trả về JSON:
{{"claim": "..."}}

Bài viết:
{article_text}
"""


def parse_claim_items(payload):
    items = payload.get("claims", payload if isinstance(payload, list) else [])
    if not isinstance(items, list):
        raise ValueError(f"Định dạng claims không hợp lệ: {payload!r}")

    parsed = {}
    for item in items:
        if not isinstance(item, dict):
            continue

        label = normalize_label(item.get("label", ""))
        claim = str(item.get("claim", "")).strip()

        if label and claim and len(claim) >= MIN_CLAIM_LENGTH:
            parsed[label] = claim

    return parsed


def records_from_claims(claims_by_label, row):
    return [
        {
            "claim": claims_by_label[label],
            "label": label,
            "article_id": row["article_id"],
            "source": row.get("source", ""),
            "title": row.get("title", ""),
        }
        for label in sorted(claims_by_label)
    ]


def generate_single_claim(row, label, provider, request_delay):
    prompt = build_single_label_prompt(row["title"], row["content"], label)
    response = chat_completion(
        prompt,
        temperature=0.5,
        max_tokens=300,
        provider=provider,
        request_delay=request_delay,
    )
    payload = extract_json_block(response)
    claim = str(payload.get("claim", "")).strip()

    if len(claim) < MIN_CLAIM_LENGTH:
        raise ValueError(f"Claim {label} quá ngắn hoặc trống")

    return claim


def generate_claims_for_article(row, provider="auto", request_delay=None):
    claims_by_label = {}

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt = build_prompt(row["title"], row["content"])
        response = chat_completion(
            prompt,
            temperature=0.4,
            max_tokens=1200,
            provider=provider,
            request_delay=request_delay,
        )
        payload = extract_json_block(response)
        claims_by_label.update(parse_claim_items(payload))

        if REQUIRED_LABELS <= set(claims_by_label):
            return records_from_claims(claims_by_label, row)

        missing = REQUIRED_LABELS - set(claims_by_label)
        print(
            f"  Lần {attempt}: thiếu {missing} — thử lại..."
        )

    missing = REQUIRED_LABELS - set(claims_by_label)
    for label in missing:
        try:
            claims_by_label[label] = generate_single_claim(
                row, label, provider, request_delay
            )
        except Exception as exc:
            print(f"  Không sinh được claim {label}: {exc}")

    if not REQUIRED_LABELS <= set(claims_by_label):
        raise ValueError(
            f"Thiếu nhãn sau khi sinh: {REQUIRED_LABELS - set(claims_by_label)}"
        )

    return records_from_claims(claims_by_label, row)


def article_has_complete_claims(records, article_id):
    by_label = {}
    for record in records:
        if record.get("article_id") != article_id:
            continue
        claim = str(record.get("claim", "")).strip()
        label = record.get("label")
        if claim and len(claim) >= MIN_CLAIM_LENGTH and label in REQUIRED_LABELS:
            by_label[label] = record

    return REQUIRED_LABELS <= set(by_label)


def upsert_article_records(all_records, article_id, new_records):
    kept = [r for r in all_records if r.get("article_id") != article_id]
    return kept + new_records


def get_incomplete_article_ids(records):
    article_ids = {r["article_id"] for r in records}
    return {
        aid for aid in article_ids
        if not article_has_complete_claims(records, aid)
    }


def main():
    parser = argparse.ArgumentParser(description="Sinh tập dữ liệu Viet-MedFact")
    parser.add_argument(
        "--num-articles",
        type=int,
        default=None,
        help="Tổng số bài (legacy). Dùng --num-skds / --num-umc để chia theo nguồn.",
    )
    parser.add_argument(
        "--num-skds",
        type=int,
        default=VIET_MEDFACT_NUM_SKDS,
        help="Số bài lấy từ SKDS (suckhoedoisong)",
    )
    parser.add_argument(
        "--num-umc",
        type=int,
        default=VIET_MEDFACT_NUM_UMC,
        help="Số bài lấy từ UMC (umcclinic)",
    )
    parser.add_argument(
        "--source",
        choices=["all", "skds", "umc"],
        default="all",
        help="Chỉ sinh claim từ một nguồn (vd: umc để bổ sung UMC)",
    )
    parser.add_argument(
        "--output",
        default=VIET_MEDFACT_PATH,
        help="Đường dẫn file JSONL đầu ra",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục từ file output hiện có, bỏ qua article đã xử lý",
    )
    parser.add_argument(
        "--fix-empty",
        action="store_true",
        help="Sửa lại các bài có claim trống hoặc thiếu nhãn",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=LLM_REQUEST_DELAY,
        help="Giây chờ giữa các request LLM (tránh 429)",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "ollama", "openrouter", "gemini"],
        default="auto",
        help="LLM provider (auto = Ollama local nếu có, rồi OpenRouter/Gemini)",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_articles()
    article_lookup = {row["article_id"]: row for _, row in df.iterrows()}

    num_skds = args.num_skds
    num_umc = args.num_umc
    if args.num_articles is not None:
        half = args.num_articles // 2
        num_skds = half
        num_umc = args.num_articles - half

    if args.source == "skds":
        num_umc = 0
    elif args.source == "umc":
        num_skds = 0

    sample = sample_articles(df, num_skds=num_skds, num_umc=num_umc, seed=args.seed)
    print(
        f"Mẫu: {len(sample)} bài "
        f"(SKDS={(sample['source_group'] == 'skds').sum()}, "
        f"UMC={(sample['source_group'] == 'umc').sum()})"
    )

    all_records = []
    skip_ids = set()

    if os.path.exists(args.output):
        all_records = read_jsonl(args.output)

    if args.fix_empty:
        fix_ids = get_incomplete_article_ids(all_records)
        all_records = [
            r for r in all_records
            if r.get("article_id") not in fix_ids
        ]
        print(f"Fix-empty: cần sinh lại {len(fix_ids)} bài")
        targets = [
            article_lookup[aid]
            for aid in fix_ids
            if aid in article_lookup
        ]
    else:
        if args.resume:
            skip_ids = {
                aid for aid in {r["article_id"] for r in all_records}
                if article_has_complete_claims(all_records, aid)
            }
            print(f"Resume: bỏ qua {len(skip_ids)} bài đã đủ 3 claim")
        targets = [
            row for _, row in sample.iterrows()
            if row["article_id"] not in skip_ids
        ]

    for row in tqdm(targets, desc="Sinh claim"):
        article_id = row["article_id"]
        try:
            records = generate_claims_for_article(
                row,
                provider=args.provider,
                request_delay=args.delay,
            )
            all_records = upsert_article_records(all_records, article_id, records)
            write_jsonl(args.output, all_records)
        except Exception as exc:
            print(f"Lỗi article {article_id}: {exc}")
            write_jsonl(args.output, all_records)

    # Lọc lần cuối: bỏ claim trống nếu còn sót
    all_records = [
        r for r in all_records
        if str(r.get("claim", "")).strip()
        and len(str(r.get("claim", "")).strip()) >= MIN_CLAIM_LENGTH
    ]
    write_jsonl(args.output, all_records)

    complete_articles = sum(
        1 for aid in {r["article_id"] for r in all_records}
        if article_has_complete_claims(all_records, aid)
    )
    print(
        f"Đã lưu {len(all_records)} claim, "
        f"{complete_articles} bài đủ 3 nhãn -> {args.output}"
    )


if __name__ == "__main__":
    main()
