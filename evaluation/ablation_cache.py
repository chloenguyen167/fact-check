"""
Bước 1 của thí nghiệm ablation: chạy các model nặng MỘT lần và cache điểm số.

Với mỗi claim lưu lại:
  - top-20 dense retrieval (bge-m3) + điểm cross-encoder (bge-reranker-v2-m3)
  - điểm NLI mức chunk (giống pipeline gốc) cho các chunk ứng viên
  - các cửa sổ câu (1–2 câu) trong chunk ứng viên + điểm NLI + điểm reranker
  - thời gian từng giai đoạn

Sau đó `evaluation.ablation_analyze` tính mọi biến thể (có/không reranker,
có/không chọn câu bằng chứng, có/không cổng liên quan...) mà không phải chạy lại model.

Chế độ:
  --mode e2e     : claim → retrieve (Qdrant) → rerank → NLI   (Viet-MedFact / ViFactCheck)
  --mode oracle  : claim + main_text vàng (không retrieval)   (ViFactCheck)

Ví dụ:
  python -m evaluation.ablation_cache --dataset viet_medfact --mode e2e
  python -m evaluation.ablation_cache --dataset vifactcheck --mode e2e
  python -m evaluation.ablation_cache --dataset vifactcheck --mode oracle
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import RESULTS_DIR, VIET_MEDFACT_PATH, VIFACTCHECK_HEALTH_PATH
from evaluation.io_utils import read_jsonl
from evaluation.labels import normalize_verdict, pubhealth_label_to_verdict
from config import EMBEDDING_MODEL, NLI_MODEL, RERANKER_MODEL, TOP_K
from nli import split_windows  # dùng chung với pipeline thật

NLI_MAX_LENGTH = 512
NLI_MAX_EVIDENCE_CHARS = 1500  # giống nli.trim_evidence
CANDIDATES_PER_ORDER = 3  # top-3 theo rerank ∪ top-3 theo dense
ORACLE_CHUNK_CHARS = 800  # giống eval_nli (chunk_size=800)

DATASETS = {
    "viet_medfact": {"path": VIET_MEDFACT_PATH, "collection": None},
    "vifactcheck": {"path": VIFACTCHECK_HEALTH_PATH, "collection": "vifactcheck"},
}


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def gold_label(row):
    return normalize_verdict(row.get("label")) or pubhealth_label_to_verdict(row.get("label"))


class NLIScorer:
    def __init__(self, model_name=NLI_MODEL):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.dev = device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.dev).eval()
        id2label = self.model.config.id2label
        self.labels = [id2label[i].lower() for i in range(len(id2label))]

    @torch.no_grad()
    def score(self, claim, premises, batch_size=16):
        """premise = evidence, hypothesis = claim (giống nli.predict_nli_scores)."""
        out = []
        for i in range(0, len(premises), batch_size):
            batch = premises[i:i + batch_size]
            enc = self.tok(
                batch, [claim] * len(batch),
                truncation=True, max_length=NLI_MAX_LENGTH,
                padding=True, return_tensors="pt",
            ).to(self.dev)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
            for row in probs:
                d = dict(zip(self.labels, row.tolist()))
                out.append([d["entailment"], d["contradiction"], d["neutral"]])
        return out


class Reranker:
    def __init__(self, model_name=RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device())

    def score(self, claim, docs, batch_size=8):
        if not docs:
            return []
        logits = self.model.predict([[claim, d] for d in docs], batch_size=batch_size)
        return (1 / (1 + np.exp(-np.asarray(logits, dtype=np.float64)))).tolist()


def trim(text, max_chars=NLI_MAX_EVIDENCE_CHARS):
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."


def score_chunks(claim, chunk_texts, positions, nli, rr):
    """NLI mức chunk + cửa sổ câu (NLI + reranker) cho các chunk ở `positions`."""
    out = {}
    for pos in positions:
        text = chunk_texts[pos]
        t0 = time.perf_counter()
        nli_chunk = nli.score(claim, [trim(text)])[0]
        t1 = time.perf_counter()
        windows = split_windows(text)
        w_nli = nli.score(claim, windows)
        t2 = time.perf_counter()
        w_rr = rr.score(claim, windows)
        t3 = time.perf_counter()
        out[str(pos)] = {
            "nli": nli_chunk,
            "w_nli": w_nli,
            "w_rr": w_rr,
            "w_len": [len(w) for w in windows],
            "t_nli_chunk": t1 - t0,
            "t_nli_win": t2 - t1,
            "t_rr_win": t3 - t2,
        }
    return out


def process_e2e(row, claim, embedder, client, collection, nli, rr):
    t0 = time.perf_counter()
    q = embedder.encode(claim, normalize_embeddings=True)
    hits = client.query_points(collection_name=collection, query=q.tolist(), limit=TOP_K).points
    t1 = time.perf_counter()
    texts = [h.payload["text"] for h in hits]
    rr_scores = rr.score(claim, texts)
    t2 = time.perf_counter()

    order_rr = sorted(range(len(texts)), key=lambda i: -rr_scores[i])
    order_dense = list(range(len(texts)))
    positions = sorted(set(order_rr[:CANDIDATES_PER_ORDER]) | set(order_dense[:CANDIDATES_PER_ORDER]))
    chunks = score_chunks(claim, texts, positions, nli, rr)
    return {
        "hits": [
            {"article_id": h.payload.get("article_id"), "dense": float(h.score), "rr": float(s)}
            for h, s in zip(hits, rr_scores)
        ],
        "chunks": chunks,
        "t_retrieve": t1 - t0,
        "t_rerank": t2 - t1,
    }


def process_oracle(row, claim, nli, rr):
    from nli import split_into_chunks

    texts = split_into_chunks(row["main_text"], max_chars=ORACLE_CHUNK_CHARS)
    t0 = time.perf_counter()
    rr_scores = rr.score(claim, texts)
    t1 = time.perf_counter()
    chunks = score_chunks(claim, texts, list(range(len(texts))), nli, rr)
    return {
        "hits": [{"article_id": None, "dense": None, "rr": float(s)} for s in rr_scores],
        "chunks": chunks,
        "t_retrieve": 0.0,
        "t_rerank": t1 - t0,
    }


def main():
    parser = argparse.ArgumentParser(description="Cache điểm số cho ablation")
    parser.add_argument("--dataset", choices=list(DATASETS), default="viet_medfact")
    parser.add_argument("--mode", choices=["e2e", "oracle"], default="e2e")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.mode == "oracle" and args.dataset != "vifactcheck":
        raise SystemExit("--mode oracle chỉ dùng cho vifactcheck (cần main_text vàng)")

    output = args.output or os.path.join(
        RESULTS_DIR, "ablation", f"cache_{args.dataset}_{args.mode}.jsonl"
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)

    records = read_jsonl(DATASETS[args.dataset]["path"])
    if args.max_samples:
        records = records[: args.max_samples]

    done = set()
    if os.path.exists(output):
        for line in open(output, encoding="utf-8"):
            done.add(json.loads(line)["idx"])
    print(f"{args.dataset}/{args.mode}: {len(records)} mẫu, đã có {len(done)} → {output}")

    nli = NLIScorer()
    rr = Reranker()
    embedder = client = collection = None
    if args.mode == "e2e":
        from qdrant_store import get_qdrant_client
        from retrieval import embedder as _embedder, get_collection_name
        from evaluation.vifactcheck_utils import vifactcheck_collection_name

        embedder = _embedder
        client = get_qdrant_client()
        collection = (
            vifactcheck_collection_name(EMBEDDING_MODEL)
            if DATASETS[args.dataset]["collection"] == "vifactcheck"
            else get_collection_name()
        )
        print(f"Collection: {collection}")

    with open(output, "a", encoding="utf-8") as f:
        for idx, row in enumerate(tqdm(records, desc=f"cache {args.dataset}/{args.mode}")):
            if idx in done:
                continue
            gold = gold_label(row)
            if gold is None:
                continue
            claim = str(row.get("claim") or "").strip()
            if args.mode == "e2e":
                rec = process_e2e(row, claim, embedder, client, collection, nli, rr)
            else:
                rec = process_oracle(row, claim, nli, rr)
            rec.update({
                "idx": idx,
                "claim": claim,
                "label": gold,
                "article_id": row.get("article_id"),
                "split": row.get("split"),
            })
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
