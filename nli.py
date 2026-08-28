import os
import re

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline as hf_pipeline,
)

from config import *

NLI_MAX_LENGTH = 512
NLI_MAX_EVIDENCE_CHARS = 1500

_LABEL_MAP = {
    "entailment": "ĐÚNG",
    "contradiction": "SAI",
    "neutral": "KHÔNG ĐỦ BẰNG CHỨNG",
}

_FINETUNED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

_active_classifier = None
_active_model_path = None


def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_classifier(model_path=None):
    """Load classifier from fine-tuned path or default NLI_MODEL."""
    global _active_classifier, _active_model_path

    target = model_path or NLI_MODEL
    if _active_model_path == target and _active_classifier is not None:
        return _active_classifier

    device_str = _get_device()
    device_idx = 0 if device_str in ("cuda", "mps") else -1

    print(f"[nli] Loading model: {target} (device={device_str})")
    _active_classifier = hf_pipeline(
        "text-classification",
        model=target,
        device=device_idx if device_str == "cuda" else device_str,
        truncation=True,
        max_length=NLI_MAX_LENGTH,
    )
    _active_model_path = target
    return _active_classifier


def set_nli_model(model_path):
    """Switch NLI model at runtime (e.g. to a fine-tuned checkpoint)."""
    _load_classifier(model_path)


def get_classifier():
    if _active_classifier is None:
        _load_classifier()
    return _active_classifier


classifier = None  # lazy-loaded


def trim_evidence(evidence, max_chars=NLI_MAX_EVIDENCE_CHARS):
    text = " ".join(str(evidence).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def predict_nli(claim, evidence):
    """Single-evidence NLI (top-1 label only)."""
    clf = get_classifier()
    claim = str(claim).strip()
    evidence = trim_evidence(evidence)
    result = clf(
        {"text": evidence, "text_pair": claim},
        truncation=True,
        max_length=NLI_MAX_LENGTH,
    )
    return result


def predict_nli_scores(claim, evidence):
    """Return dict of {entailment, contradiction, neutral} scores."""
    clf = get_classifier()
    claim = str(claim).strip()
    evidence = trim_evidence(evidence)
    raw = clf(
        {"text": evidence, "text_pair": claim},
        truncation=True,
        max_length=NLI_MAX_LENGTH,
        top_k=None,
    )
    return {item["label"]: item["score"] for item in raw}


def split_into_chunks(text, max_chars=500, min_chars=50):
    """Split a long text into meaningful chunks for per-chunk NLI."""
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else [text]

    paragraphs = re.split(r"\n\s*\n|\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if len(paragraphs) <= 1:
        paragraphs = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""
    for para in paragraphs:
        if not para.strip():
            continue
        if len(current) + len(para) + 1 <= max_chars:
            current = (current + " " + para).strip() if current else para
        else:
            if current and len(current) >= min_chars:
                chunks.append(current)
            current = para
    if current and len(current) >= min_chars:
        chunks.append(current)

    return chunks if chunks else [text[:max_chars]]


def predict_nli_multi(claim, evidence_chunks, strategy="weighted_max"):
    """
    Run NLI on multiple evidence chunks and aggregate.

    Strategies:
    - "weighted_max": for each label, take the max across chunks.
      Uses margin-based logic: a decisive label (entailment/contradiction)
      is accepted only when its confidence clearly exceeds the average
      neutral score, preventing topically-related-but-neutral evidence
      from being misclassified as entailment.
    - "mean": average each label's scores, pick the highest.
    """
    if not evidence_chunks:
        return "KHÔNG ĐỦ BẰNG CHỨNG", 0.0, []

    all_scores = {"entailment": [], "contradiction": [], "neutral": []}
    chunk_details = []

    for chunk in evidence_chunks:
        scores = predict_nli_scores(claim, chunk)
        for label in all_scores:
            all_scores[label].append(scores.get(label, 0.0))
        chunk_details.append(scores)

    if strategy == "mean":
        agg = {
            label: sum(vals) / len(vals) for label, vals in all_scores.items()
        }
        best_nli_label = max(agg, key=agg.get)
        best_score = agg[best_nli_label]
    else:
        max_ent = max(all_scores["entailment"])
        max_con = max(all_scores["contradiction"])
        avg_neu = sum(all_scores["neutral"]) / len(all_scores["neutral"])

        ent_margin = max_ent - avg_neu
        con_margin = max_con - avg_neu

        best_margin = max(ent_margin, con_margin)

        if best_margin > 0.3:
            if max_ent >= max_con:
                best_nli_label, best_score = "entailment", max_ent
            else:
                best_nli_label, best_score = "contradiction", max_con
        elif max_con > 0.5:
            best_nli_label, best_score = "contradiction", max_con
        else:
            best_nli_label = "neutral"
            best_score = avg_neu

    return _LABEL_MAP[best_nli_label], best_score, chunk_details


def get_label(result):
    label = None
    if isinstance(result, dict):
        label = result.get("label")
    elif isinstance(result, (list, tuple)) and result:
        first_item = result[0]
        if isinstance(first_item, dict):
            label = first_item.get("label")

    if not label:
        raise ValueError(f"Unexpected NLI result format: {result!r}")

    return _LABEL_MAP.get(label.lower(), "KHÔNG ĐỦ BẰNG CHỨNG")


def get_prediction(result):
    item = None
    if isinstance(result, dict):
        item = result
    elif isinstance(result, (list, tuple)) and result:
        item = result[0]

    if not isinstance(item, dict) or "label" not in item:
        raise ValueError(f"Unexpected NLI result format: {result!r}")

    label = item["label"]
    score = item.get("score", 0.0)
    verdict = _LABEL_MAP.get(label.lower(), "KHÔNG ĐỦ BẰNG CHỨNG")

    return verdict, score


if __name__ == "__main__":
    claim = "nước chanh chữa khỏi ung thư"
    evidence = (
        "Hiện chưa có bằng chứng khoa học "
        "cho thấy nước chanh có thể chữa khỏi ung thư."
    )

    result = predict_nli(claim, evidence)
    print("Single:", result)

    verdict, score, details = predict_nli_multi(claim, [evidence])
    print(f"Multi: {verdict} ({score:.3f})")

    long_text = (
        "Nước chanh rất giàu vitamin C, tốt cho sức khỏe. "
        "Tuy nhiên, không có nghiên cứu nào chứng minh nước chanh có thể chữa ung thư. "
        "WHO khuyến cáo không nên tin vào các phương pháp chữa ung thư chưa được kiểm chứng. "
        "Ung thư cần được điều trị bằng các phương pháp y học hiện đại."
    )
    chunks = split_into_chunks(long_text, max_chars=200)
    print(f"\nChunks ({len(chunks)}):")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c[:80]}...")
    verdict, score, details = predict_nli_multi(claim, chunks)
    print(f"Multi (chunks): {verdict} ({score:.3f})")
