# /xây dựng Vector Database.

import argparse
import re

import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import *

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
EMBED_BATCH_SIZE = 64
DB_WRITE_BATCH_SIZE = 200


def load_skds():
    df = pd.read_csv("data/articles_skds.csv")
    cols = ["title", "content", "source"]
    if "link" in df.columns:
        cols.append("link")
    df = df[cols].copy()
    if "link" not in df.columns:
        df["link"] = ""
    else:
        df["link"] = df["link"].fillna("").astype(str)
    df["article_id"] = [f"skds_{i}" for i in df.index]
    return df


def load_umc():
    df = pd.read_csv("data/articles.csv", header=None)
    # CSV UMC thường có 6 cột; nếu nhiều hơn vẫn lấy theo vị trí chuẩn
    base_cols = ["id", "title", "publish_time", "content", "source", "link"]
    if df.shape[1] < len(base_cols):
        raise ValueError(
            f"articles.csv cần ít nhất {len(base_cols)} cột, nhận {df.shape[1]}"
        )
    df = df.iloc[:, : len(base_cols)].copy()
    df.columns = base_cols
    df = df[["title", "content", "source", "link"]].copy()
    df["link"] = df["link"].fillna("").astype(str)
    df["article_id"] = [f"umc_{i}" for i in df.index]
    return df


def load_all(max_rows=None):
    df = pd.concat([load_skds(), load_umc()], ignore_index=True)

    if max_rows is not None:
        return df.head(max_rows)

    return df


def chunk_rows(df):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for _, row in df.iterrows():
        title = "" if pd.isna(row["title"]) else str(row["title"])
        content = "" if pd.isna(row["content"]) else str(row["content"])
        source = "" if pd.isna(row.get("source")) else str(row["source"])
        link = "" if pd.isna(row.get("link")) else str(row.get("link") or "").strip()

        text = f"{title}\n\n{content}".strip()

        if not text:
            continue

        article_id = row["article_id"]

        for chunk in splitter.split_text(text):
            yield {
                "text": chunk,
                "article_id": article_id,
                "title": title,
                "source": source,
                "link": link,
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


def get_collection_name():
    model_suffix = EMBEDDING_MODEL.split("/")[-1].lower()
    model_suffix = re.sub(r"[^a-z0-9]+", "_", model_suffix).strip("_")
    return f"{COLLECTION_NAME}_{model_suffix}"


def build_db(max_rows=None):
    df = load_all(max_rows=max_rows)
    print(f"Articles: {len(df)}")
    print(
        f"  SKDS: {(df['article_id'].str.startswith('skds_')).sum()}, "
        f"UMC: {(df['article_id'].str.startswith('umc_')).sum()}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    dim = model.get_sentence_embedding_dimension()

    client = QdrantClient(path=QDRANT_PATH)
    try:
        collection_name = get_collection_name()

        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
        )

        total_written = 0

        for batch_number, chunk_batch in enumerate(
            batched(chunk_rows(df), DB_WRITE_BATCH_SIZE),
            start=1,
        ):
            texts = [item["text"] for item in chunk_batch]

            embeddings = model.encode(
                texts,
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=True,
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

            client.upsert(
                collection_name=collection_name,
                points=points,
            )

            total_written += len(chunk_batch)
            print(f"Saved batch {batch_number}: {total_written} chunks")

        print(f"Done — {total_written} chunks, collection={collection_name}")
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="Build Qdrant vector DB")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Giới hạn số bài (mặc định: toàn bộ SKDS + UMC)",
    )
    args = parser.parse_args()
    build_db(max_rows=args.max_rows)


if __name__ == "__main__":
    main()
