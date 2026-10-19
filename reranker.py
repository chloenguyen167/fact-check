import numpy as np
from sentence_transformers import CrossEncoder
from config import *

reranker = CrossEncoder(
    RERANKER_MODEL,
    device="cpu"
)

def rerank(claim, documents, top_n=None):

    pairs = [
        [claim, doc]
        for doc in documents
    ]

    # predict() trả về logit thô -> đưa qua sigmoid để được thang 0-1
    logits = reranker.predict(pairs)
    scores = 1 / (1 + np.exp(-np.array(logits)))

    ranked = list(
        zip(documents, scores)
    )

    ranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[: (top_n or TOP_RERANK)]


def score_texts(claim, texts):
    """Điểm liên quan (0-1) cho từng text, giữ nguyên thứ tự đầu vào."""
    if not texts:
        return []
    logits = reranker.predict([[claim, t] for t in texts])
    return (1 / (1 + np.exp(-np.array(logits)))).tolist()


def rerank_records(claim, records, top_n=None):
    """Rerank documents kèm metadata (article_id, ...)."""

    texts = [record["text"] for record in records]
    ranked = rerank(claim, texts, top_n=top_n or len(texts))

    text_to_record = {record["text"]: record for record in records}

    return [
        {
            **text_to_record[text],
            "score": float(score),
        }
        for text, score in ranked
    ]

if __name__ == "__main__":

    from retrieval import retrieve

    claim = "uống nước chanh chữa ung thư"

    docs = retrieve(claim)

    results = rerank(
        claim,
        docs
    )

    for doc, score in results:
        print(score)
        print(doc[:500])
        print("="*50)