import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def hit_at_k(ranks, k):
  """Tỷ lệ mẫu có rank <= k (rank bắt đầu từ 1, None nếu không tìm thấy)."""

  if not ranks:
      return 0.0

  hits = sum(1 for rank in ranks if rank is not None and rank <= k)
  return hits / len(ranks)


def mean_reciprocal_rank(ranks):
  """MRR: trung bình 1/rank của kết quả đúng đầu tiên."""

  if not ranks:
      return 0.0

  scores = []
  for rank in ranks:
      if rank is None or rank <= 0:
          scores.append(0.0)
      else:
          scores.append(1.0 / rank)

  return float(np.mean(scores))


def find_article_rank(records, target_article_id):
  """Tìm vị trí (1-based) của article_id trong danh sách kết quả."""

  if not target_article_id:
      return None

  for index, record in enumerate(records, start=1):
      if record.get("article_id") == target_article_id:
          return index

  return None


def classification_metrics(y_true, y_pred, labels):
  return {
      "accuracy": float(accuracy_score(y_true, y_pred)),
      "precision_macro": float(
          precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
      ),
      "recall_macro": float(
          recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
      ),
      "f1_macro": float(
          f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
      ),
      "classification_report": classification_report(
          y_true, y_pred, labels=labels, zero_division=0
      ),
      "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
      "labels": labels,
  }


def save_json(data, path):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as file:
      json.dump(data, file, ensure_ascii=False, indent=2)


def save_text(text, path):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as file:
      file.write(text)


def summarize_latency(latencies):
  if not latencies:
      return {"count": 0, "mean_sec": 0.0, "p50_sec": 0.0, "p95_sec": 0.0}

  arr = np.array(latencies, dtype=float)
  return {
      "count": int(len(arr)),
      "mean_sec": float(np.mean(arr)),
      "p50_sec": float(np.percentile(arr, 50)),
      "p95_sec": float(np.percentile(arr, 95)),
  }


def aggregate_retrieval_results(per_sample_results, k_values):
  summary = defaultdict(dict)

  for mode in per_sample_results:
      ranks = [item["rank"] for item in per_sample_results[mode]]
      for k in k_values:
          summary[mode][f"hit@{k}"] = hit_at_k(ranks, k)
      summary[mode]["mrr"] = mean_reciprocal_rank(ranks)
      summary[mode]["num_samples"] = len(ranks)

  return dict(summary)
