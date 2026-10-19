"""
Chọn thiết kế luật quyết định CHỈ trên tập dev (idx >= 100) bằng 5-fold CV nội bộ.
Tập test (100 mẫu đầu) không được đọc ở đây.

  python -m evaluation.ablation_design_dev --dataset viet_medfact --mode e2e
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.ablation_analyze import (
    NEI, DUNG, SAI, TAU_GRID, TEST_SIZE, evidence_positions, gather_scores, macro_f1, relevance,
)
from evaluation.config_eval import RESULTS_DIR

DIFF_GRID = np.round(np.arange(0.0, 0.951, 0.05), 2)


def feats(recs, unit, m, mode):
    out = []
    for r in recs:
        pos = evidence_positions(r, "rr", mode)
        s = np.asarray(gather_scores(r, pos, unit, m) or [[0, 0, 1]])
        out.append((s[:, 0].max(), s[:, 1].max(), relevance(r, pos, "rr")))
    return np.array(out)


def rule_threshold(F, te, tc, tr):
    ent, con, rel = F[:, 0], F[:, 1], F[:, 2]
    pred = np.full(len(F), NEI, dtype=object)
    ok_e, ok_c = ent >= te, con >= tc
    pred[ok_e & (~ok_c | (ent >= con))] = DUNG
    pred[ok_c & (~ok_e | (con > ent))] = SAI
    pred[rel < tr] = NEI
    return pred


def rule_diff(F, dp, dn, tr):
    d = F[:, 0] - F[:, 1]
    pred = np.full(len(F), NEI, dtype=object)
    pred[d >= dp] = DUNG
    pred[d <= -dn] = SAI
    pred[F[:, 2] < tr] = NEI
    return pred


def grid(rule, F_tr, y_tr):
    rel_q = sorted(set(np.round(np.quantile(F_tr[:, 2], [0, .1, .2, .3, .4]), 4)) | {-np.inf})
    g = TAU_GRID if rule is rule_threshold else DIFF_GRID
    best = None
    for a, b, tr in itertools.product(g, g, rel_q):
        f1 = macro_f1(y_tr, rule(F_tr, a, b, tr))
        if best is None or f1 > best[0]:
            best = (f1, (a, b, tr))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="viet_medfact")
    ap.add_argument("--mode", default="e2e")
    args = ap.parse_args()
    path = os.path.join(RESULTS_DIR, "ablation", f"cache_{args.dataset}_{args.mode}.jsonl")
    dev = [json.loads(line) for line in open(path, encoding="utf-8")]
    dev = sorted([r for r in dev if r["idx"] >= TEST_SIZE], key=lambda r: r["idx"])
    y = np.array([r["label"] for r in dev], dtype=object)
    rng = np.random.default_rng(7)
    folds = np.array_split(rng.permutation(len(dev)), 5)
    print(f"dev n={len(dev)}")
    rows = []
    for unit, m in [("chunk", None), ("window", 1), ("window", 3), ("window", 5), ("window", None)]:
        F = feats(dev, unit, m, args.mode)
        for rule in (rule_threshold, rule_diff):
            pred = np.empty(len(dev), dtype=object)
            for f in folds:
                tr = np.setdiff1d(np.arange(len(dev)), f)
                params = grid(rule, F[tr], y[tr])
                pred[f] = rule(F[f], *params)
            rows.append((unit, m, rule.__name__, float(np.mean(pred == y)), macro_f1(y, pred)))
            print(f"  {unit:<7} m={str(m):<5} {rule.__name__:<15} acc={rows[-1][3]:.3f} F1={rows[-1][4]:.3f}")
    best = max(rows, key=lambda r: r[4])
    print("best:", best)


if __name__ == "__main__":
    main()
