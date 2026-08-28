"""
Index Context ViFactCheck-Health vào Qdrant (collection riêng cho đánh giá RAG).

Mỗi URL = 1 bài (~62 bài unique / 438 claim). Không trộn SKDS/UMC để retrieval
không bị nhiễu bởi corpus y tế khác khi chấm baseline trên ViFactCheck.

Usage:
  python -m evaluation.prepare_vifactcheck --force
  python -m evaluation.index_vifactcheck
  python -m evaluation.index_vifactcheck --recreate
"""

import argparse
import os
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBEDDING_MODEL
from evaluation.config_eval import VIFACTCHECK_HEALTH_PATH
from evaluation.io_utils import read_jsonl
from evaluation.vifactcheck_utils import url_to_article_id
from qdrant_store import close_qdrant_client, get_qdrant_client

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
EMBED_BATCH_SIZE = 64
DB_WRITE_BATCH_SIZE = 200


def get_collection_name():
    from evaluation.vifactcheck_utils import vifactcheck_collection_name

    return vifactcheck_collection_name(EMBEDDING_MODEL)


def load_articles(jsonl_path):
    rows = read_jsonl(jsonl_path)
    by_url = {}
    for row in rows:
        url = str(row.get("url") or "").strip()
        main_text = str(row.get("main_text") or "").strip()
        if not url or not main_text:
            continue
        article_id = url_to_article_id(url)
        if url not in by_url:
            by_url[url] = {
                "article_id": article_id,
                "title": str(row.get("topic") or row.get("author") or "ViFactCheck"),
                "content": main_text,
                "source": "vifactcheck",
                "link": url,
            }
    return list(by_url.values())


def chunk_articles(articles):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    for article in articles:
        text = article["content"].strip()
        if not text:
            continue
        for chunk in splitter.split_text(text):
            yield {
                "text": chunk,
                "article_id": article["article_id"],
                "title": article["title"],
                "source": article["source"],
                "link": article["link"],
            }


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_index(recreate=False):
    if not os.path.exists(VIFACTCHECK_HEALTH_PATH):
        raise FileNotFoundError(
            f"Chưa có {VIFACTCHECK_HEALTH_PATH}. "
            "Chạy: python -m evaluation.prepare_vifactcheck --force"
        )

    articles = load_articles(VIFACTCHECK_HEALTH_PATH)
    if not articles:
        raise ValueError("Không có bài nào để index từ ViFactCheck-Health")

    print(f"Articles unique (theo URL): {len(articles)}")

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    dim = model.get_sentence_embedding_dimension()
    collection_name = get_collection_name()

    client = get_qdrant_client()
    try:
        exists = False
        try:
            info = client.get_collection(collection_name)
            exists = info.points_count > 0
            print(f"Collection {collection_name}: {info.points_count} points")
        except Exception:
            pass

        if recreate or not exists:
            if exists and recreate:
                print(f"Recreate collection {collection_name}...")
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        elif exists:
            print(f"Dùng collection hiện có ({collection_name}). Dùng --recreate để index lại.")

        total_written = 0
        if recreate or not exists:
            for batch_number, chunk_batch in enumerate(
                batched(chunk_articles(articles), DB_WRITE_BATCH_SIZE),
                start=1,
            ):
                texts = [item["text"] for item in chunk_batch]
                embeddings = model.encode(
                    texts,
                    batch_size=EMBED_BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=batch_number == 1,
                )
                points = [
                    PointStruct(
                        id=total_written + i,
                        vector=vector.tolist(),
                        payload={
                            "text": item["text"],
                            "article_id": item["article_id"],
                            "title": item["title"],
                            "source": item["source"],
                            "link": item.get("link") or "",
                        },
                    )
                    for i, (item, vector) in enumerate(zip(chunk_batch, embeddings))
                ]
                client.upsert(collection_name=collection_name, points=points)
                total_written += len(chunk_batch)
                print(f"Saved batch {batch_number}: {total_written} chunks")

            print(f"Done — {total_written} chunks in {collection_name}")
        else:
            print("Bỏ qua ghi chunk (collection đã có dữ liệu).")
    finally:
        close_qdrant_client()


def main():
    parser = argparse.ArgumentParser(description="Index ViFactCheck-Health vào Qdrant")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xóa và tạo lại collection vifactcheck_health_*",
    )
    args = parser.parse_args()
    build_index(recreate=args.recreate)
    print(f"\nCollection: {get_collection_name()}")
    print(
        "Tiếp theo:\n"
        "  python -m evaluation.eval_baselines --dataset vifactcheck --max-samples 100"
    )


if __name__ == "__main__":
    main()
