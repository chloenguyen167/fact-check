import re

import torch
from sentence_transformers import SentenceTransformer

from config import *
from qdrant_store import get_qdrant_client

_active_collection = None


def get_collection_name():

    model_suffix = EMBEDDING_MODEL.split("/")[-1].lower()
    model_suffix = re.sub(r"[^a-z0-9]+", "_", model_suffix).strip("_")
    return f"{COLLECTION_NAME}_{model_suffix}"


def set_retrieval_collection(name):
    """Ghi đè collection cho eval (vd. vifactcheck_health_bge_m3). None = mặc định."""
    global _active_collection
    _active_collection = name


def get_active_collection_name():
    return _active_collection or collection_name


collection_name = get_collection_name()

# CUDA nếu có (nhanh hơn ~50 lần); MPS giữ CPU như cũ
embedder = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def retrieve(claim, limit=None):

    q = embedder.encode(
        claim,
        normalize_embeddings=True
    )

    hits = get_qdrant_client().query_points(
        collection_name=get_active_collection_name(),
        query=q.tolist(),
        limit=limit or TOP_K
    ).points

    return [
        hit.payload["text"]
        for hit in hits
    ]


def retrieve_with_metadata(claim, limit=None):

    q = embedder.encode(
        claim,
        normalize_embeddings=True
    )

    hits = get_qdrant_client().query_points(
        collection_name=get_active_collection_name(),
        query=q.tolist(),
        limit=limit or TOP_K
    ).points

    return [
        {
            "text": hit.payload["text"],
            "article_id": hit.payload.get("article_id"),
            "title": hit.payload.get("title", ""),
            "source": hit.payload.get("source", ""),
            "link": hit.payload.get("link") or "",
            "publish_time": hit.payload.get("publish_time") or "",
        }
        for hit in hits
    ]


def format_preview(text, max_chars=500):

    cleaned = " ".join(text.split())

    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[:max_chars].rstrip() + "..."

#main
if __name__ == "__main__":

    docs = retrieve(
        "uống nước chanh chữa ung thư"
    )

    for i, doc in enumerate(docs[:5], start=1):
        print(f"\n=== Result {i} ===")
        print(format_preview(doc))
