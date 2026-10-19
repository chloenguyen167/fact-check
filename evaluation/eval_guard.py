"""
Đánh giá Guard 3 tầng (lọc input trước pipeline) — ablation theo tầng.

  - Hợp lệ (phải cho qua): toàn bộ claim của Viet-MedFact + ViFactCheck-Health
  - Không hợp lệ (phải chặn): evaluation/data/guard_test_invalid.csv (viết tay,
    không trùng data/guard_train.csv)

Cấu hình:
  none           : không có Guard (mọi input vào pipeline)
  rules          : chỉ tầng 1 (rule)
  rules+ner      : tầng 1 + NER-fallback (khi chưa có trọng số PhoBERT)
  rules+phobert  : tầng 1 + 2 + 3 đầy đủ (nếu đã train models/guard_phobert)

Ví dụ:
  python -m evaluation.eval_guard
"""

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import DATA_DIR, RESULTS_DIR, VIET_MEDFACT_PATH, VIFACTCHECK_HEALTH_PATH
from evaluation.io_utils import read_jsonl
from guard.classifier import GuardClassifier
from guard.guard import Guard
from guard.rules import rule_filter

INVALID_PATH = os.path.join(DATA_DIR, "guard_test_invalid.csv")


def load_inputs():
    valid = [r["claim"] for r in read_jsonl(VIET_MEDFACT_PATH)]
    valid += [r["claim"] for r in read_jsonl(VIFACTCHECK_HEALTH_PATH)]
    with open(INVALID_PATH, encoding="utf-8") as f:
        invalid = list(csv.DictReader(f))
    return valid, invalid


def build_checkers():
    import config as cfg

    checkers = {
        "none": lambda t: True,
        "rules": lambda t: rule_filter(t).verdict == "pass",
    }
    ner_guard = Guard(classifier_path=None, gazetteer_dir=cfg.GUARD_GAZETTEER_DIR)
    checkers["rules+ner"] = lambda t: ner_guard.check(t).valid
    if GuardClassifier(cfg.GUARD_CLASSIFIER_PATH).available:
        full = Guard(
            classifier_path=cfg.GUARD_CLASSIFIER_PATH,
            threshold=cfg.GUARD_THRESHOLD,
            gazetteer_dir=cfg.GUARD_GAZETTEER_DIR,
            ner_assist_min_conf=cfg.GUARD_NER_ASSIST_MIN_CONF,
        )
        checkers["rules+phobert"] = lambda t: full.check(t).valid
    return checkers


def main():
    valid, invalid = load_inputs()
    results = {}
    for name, check in build_checkers().items():
        t0 = time.perf_counter()
        valid_pass = [check(t) for t in valid]
        invalid_pass = [check(r["text"]) for r in invalid]
        elapsed = time.perf_counter() - t0
        by_type = {}
        for r, passed in zip(invalid, invalid_pass):
            d = by_type.setdefault(r["label"], [0, 0])
            d[0] += 0 if passed else 1
            d[1] += 1
        tp = sum(not p for p in invalid_pass)  # chặn đúng
        fp = sum(not p for p in valid_pass)  # chặn nhầm claim hợp lệ
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / len(invalid)
        results[name] = {
            "valid_total": len(valid),
            "valid_false_reject": fp,
            "false_reject_rate": fp / len(valid),
            "invalid_total": len(invalid),
            "invalid_blocked": tp,
            "block_recall": recall,
            "block_precision": precision,
            "block_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "blocked_by_type": {k: {"blocked": v[0], "total": v[1]} for k, v in by_type.items()},
            "ms_per_input": 1000 * elapsed / (len(valid) + len(invalid)),
            "false_reject_examples": [t for t, p in zip(valid, valid_pass) if not p][:15],
            "missed_invalid_examples": [r["text"] for r, p in zip(invalid, invalid_pass) if p],
        }
        r = results[name]
        print(f"{name:<14} block recall={r['block_recall']:.3f} precision={r['block_precision']:.3f} "
              f"F1={r['block_f1']:.3f} false-reject={r['false_reject_rate']:.3%} "
              f"({r['ms_per_input']:.1f} ms/input)")

    out = os.path.join(RESULTS_DIR, "guard_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu -> {out}")


if __name__ == "__main__":
    main()
