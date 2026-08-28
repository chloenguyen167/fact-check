"""Tiện ích ViFactCheck-Health: article_id, collection RAG eval."""

import hashlib
import re


def url_to_article_id(url: str) -> str:
    """Một URL = một bài = một article_id ổn định."""
    normalized = str(url or "").strip().rstrip("/").lower()
    if not normalized:
        return ""
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]
    return f"vifactcheck_{digest}"


def attach_article_id(row: dict) -> dict:
    """Thêm article_id vào record jsonl (theo URL)."""
    out = dict(row)
    out["article_id"] = url_to_article_id(row.get("url", ""))
    return out


def vifactcheck_collection_name(embedding_model: str, prefix: str = "vifactcheck_health") -> str:
    model_suffix = embedding_model.split("/")[-1].lower()
    model_suffix = re.sub(r"[^a-z0-9]+", "_", model_suffix).strip("_")
    return f"{prefix}_{model_suffix}"
