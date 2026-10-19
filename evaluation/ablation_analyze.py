"""
Bước 2 của thí nghiệm ablation: tính các biến thể từ cache (không chạy model).

Chia tập: test = 100 mẫu đầu (trùng tập đã dùng cho baseline LLM-only / RAG+LLM),
dev = phần còn lại → chỉ dùng để chọn siêu tham số (ngưỡng, số cửa sổ m).

Biến thể (xem VARIANTS):
  FULL (đề xuất) : rerank top-3 chunk → NLI mức chunk + weighted_max quyết định "có kết luận hay NEI";
                   top-m câu (ESS, chọn bằng cross-encoder) quyết định hướng ĐÚNG/SAI. Chỉ m được chọn trên dev.
  Ablation       : − ESS (= RAG+NLI gốc), − rerank chunk, − cả hai; − NLI (= RAG+LLM), − RAG (= LLM-only)
  Mở rộng đã thử : + hiệu chỉnh ngưỡng τe/τc, + cổng liên quan τr, tổng hợp trung bình

Ví dụ:
  python -m evaluation.ablation_analyze --dataset viet_medfact --mode e2e
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config_eval import RESULTS_DIR
from evaluation.labels import VERDICTS
from evaluation.metrics import classification_metrics

DUNG, SAI, NEI = VERDICTS
TEST_SIZE = 100
TOP_K_EVIDENCE = 3
TAU_GRID = np.round(np.arange(0.30, 0.991, 0.05), 2)
M_GRID = [1, 2, 3, 5, None]  # None = mọi cửa sổ
N_BOOT = 2000
SEED = 13

BASELINE_FILES = {
    ("viet_medfact", "e2e"): "baseline_comparison.json",
    ("vifactcheck", "e2e"): "baseline_comparison_vifactcheck.json",
}


# --------------------------------------------------------------------------- #
# Trích đặc trưng từ cache
# --------------------------------------------------------------------------- #

def evidence_positions(rec, order, mode):
    hits = rec["hits"]
    if mode == "oracle":
        positions = list(range(len(hits)))
        if order == "rr":  # oracle: giữ mọi chunk, chỉ sắp theo rr
            positions.sort(key=lambda i: -hits[i]["rr"])
        return positions
    if order == "rr":
        ranked = sorted(range(len(hits)), key=lambda i: -hits[i]["rr"])
    else:
        ranked = list(range(len(hits)))
    return ranked[:TOP_K_EVIDENCE]


def gather_scores(rec, positions, unit, m):
    """Trả về list [e, c, n] theo đơn vị chunk hoặc cửa sổ câu (top-m theo reranker)."""
    chunks = rec["chunks"]
    if unit == "chunk":
        return [chunks[str(p)]["nli"] for p in positions]
    pairs = []
    for p in positions:
        ch = chunks[str(p)]
        pairs.extend(zip(ch["w_rr"], ch["w_nli"]))
    if m is not None:
        pairs = sorted(pairs, key=lambda x: -x[0])[:m]
    return [s for _, s in pairs]


def relevance(rec, positions, source):
    vals = [rec["hits"][p][source] for p in positions]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else 1.0


# --------------------------------------------------------------------------- #
# Luật quyết định
# --------------------------------------------------------------------------- #

def decide_weighted_max(scores, margin=0.3, con_fallback=0.5):
    """Sao chép nli.predict_nli_multi(strategy="weighted_max")."""
    if not scores:
        return NEI
    arr = np.asarray(scores)
    max_ent, max_con, avg_neu = arr[:, 0].max(), arr[:, 1].max(), arr[:, 2].mean()
    if max(max_ent - avg_neu, max_con - avg_neu) > margin:
        return DUNG if max_ent >= max_con else SAI
    if max_con > con_fallback:
        return SAI
    return NEI


def decide_mean(scores):
    if not scores:
        return NEI
    return [DUNG, SAI, NEI][int(np.asarray(scores).mean(axis=0).argmax())]


def decide_calibrated(ent, con, rel, tau_e, tau_c, tau_r):
    """Vector hoá: ent/con/rel là mảng numpy."""
    pred = np.full(ent.shape, NEI, dtype=object)
    ok_e = ent >= tau_e
    ok_c = con >= tau_c
    both = ok_e & ok_c
    pred[ok_e & ~ok_c] = DUNG
    pred[ok_c & ~ok_e] = SAI
    pred[both & (ent >= con)] = DUNG
    pred[both & (ent < con)] = SAI
    pred[rel < tau_r] = NEI
    return pred


def per_class(y_true, y_pred):
    out = {}
    for lab in VERDICTS:
        tp = int(np.sum((y_pred == lab) & (y_true == lab)))
        fp = int(np.sum((y_pred == lab) & (y_true != lab)))
        fn = int(np.sum((y_pred != lab) & (y_true == lab)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                    "support": tp + fn}
    return out


def macro_f1(y_true, y_pred):
    f1s = []
    for lab in VERDICTS:
        tp = np.sum((y_pred == lab) & (y_true == lab))
        fp = np.sum((y_pred == lab) & (y_true != lab))
        fn = np.sum((y_pred != lab) & (y_true == lab))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


# --------------------------------------------------------------------------- #
# Biến thể
# --------------------------------------------------------------------------- #

def features(recs, spec, m, mode):
    """Đặc trưng mức chunk (quyết định có kết luận hay không) và mức câu (hướng ĐÚNG/SAI)."""
    out = {k: [] for k in ("ent_c", "con_c", "ent_w", "con_w", "rel")}
    chunk_scores = []
    for rec in recs:
        pos = evidence_positions(rec, spec["order"], mode)
        sc = np.asarray(gather_scores(rec, pos, "chunk", None))
        sw = np.asarray(gather_scores(rec, pos, "window", m if spec.get("select", True) else None) or [[0, 0, 1]])
        out["ent_c"].append(sc[:, 0].max())
        out["con_c"].append(sc[:, 1].max())
        out["ent_w"].append(sw[:, 0].max())
        out["con_w"].append(sw[:, 1].max())
        out["rel"].append(relevance(rec, pos, spec["rel_source"]))
        chunk_scores.append(sc.tolist())
    feats = {k: np.array(v) for k, v in out.items()}
    feats["chunk_scores"] = chunk_scores
    return feats


def direction(f, dir_unit, ok_e=None, ok_c=None):
    if dir_unit == "window":
        return np.where(f["ent_w"] >= f["con_w"], DUNG, SAI)
    # mức chunk: giữ luật ngưỡng gốc (chỉ một ngưỡng qua → nhãn đó; cả hai → xác suất lớn hơn)
    ent, con = f["ent_c"], f["con_c"]
    d = np.where(ent >= con, DUNG, SAI)
    if ok_e is not None:
        d = np.where(ok_e & ~ok_c, DUNG, np.where(ok_c & ~ok_e, SAI, d))
    return d


def decide_calibrated(f, tau_e, tau_c, tau_r, dir_unit):
    ok_e, ok_c = f["ent_c"] >= tau_e, f["con_c"] >= tau_c
    pred = np.full(len(ok_e), NEI, dtype=object)
    decisive = ok_e | ok_c
    pred[decisive] = direction(f, dir_unit, ok_e, ok_c)[decisive]
    pred[f["rel"] < tau_r] = NEI
    return pred


def decide_wmax_hybrid(f, tau_r, dir_unit, gate):
    base = np.array([decide_weighted_max(s) for s in f["chunk_scores"]], dtype=object)
    if dir_unit == "window":
        decisive = base != NEI
        base[decisive] = direction(f, "window")[decisive]
    if gate:
        base[f["rel"] < tau_r] = NEI
    return base


def tau_r_grid(rel_dev, use_gate):
    if not use_gate:
        return [-np.inf]
    qs = np.quantile(rel_dev, [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50])
    return sorted(set(np.round(qs, 4).tolist()) | {-np.inf})


def m_grid(spec):
    return M_GRID if spec["dir"] == "window" and spec.get("select", True) else [None]


def tune_calibrated(dev, spec, mode):
    y = np.array([r["label"] for r in dev], dtype=object)
    best = None
    for m in m_grid(spec):
        f = features(dev, spec, m, mode)
        for tau_r in tau_r_grid(f["rel"], spec["gate"]):
            for tau_e, tau_c in itertools.product(TAU_GRID, TAU_GRID):
                pred = decide_calibrated(f, tau_e, tau_c, tau_r, spec["dir"])
                key = (macro_f1(y, pred), float(np.mean(pred == y)))
                if best is None or key > best[0]:
                    best = (key, {"m": m, "tau_e": float(tau_e), "tau_c": float(tau_c), "tau_r": float(tau_r)})
    return best[1], best[0]


def tune_gate_only(dev, spec, mode):
    """−Calib: luật weighted_max cố định quyết định có kết luận hay không; chỉ chọn m và τ_r."""
    y = np.array([r["label"] for r in dev], dtype=object)
    best = None
    for m in m_grid(spec):
        f = features(dev, spec, m, mode)
        for tau_r in tau_r_grid(f["rel"], spec["gate"]):
            pred = decide_wmax_hybrid(f, tau_r, spec["dir"], spec["gate"])
            key = (macro_f1(y, pred), float(np.mean(pred == y)))
            if best is None or key > best[0]:
                best = (key, {"m": m, "tau_r": float(tau_r)})
    return best[1], best[0]


def predict(recs, spec, params, mode):
    f = features(recs, spec, params.get("m"), mode)
    if spec["decision"] == "wmax":
        return decide_wmax_hybrid(f, params.get("tau_r", -np.inf), spec["dir"], spec["gate"])
    if spec["decision"] == "mean":
        return np.array([decide_mean(s) for s in f["chunk_scores"]], dtype=object)
    return decide_calibrated(f, params["tau_e"], params["tau_c"], params["tau_r"], spec["dir"])


# dir = đơn vị bằng chứng quyết định hướng ĐÚNG/SAI; mức chunk luôn quyết định "có kết luận hay NEI".
# Thiết kế FULL được chọn trên dev (ablation_design_dev / repeated CV): tốt nhất trên dev của cả
# Viet-MedFact và ViFactCheck; các mở rộng có nhiều tham số (calib, gate) bị overfit trên dev.
VARIANTS = [
    ("FULL", "Đề xuất (Rerank + NLI + ESS)", dict(order="rr", dir="window", decision="wmax", gate=False, rel_source="rr")),
    # Ablation — loại bỏ thành phần
    ("B0", "− ESS (RAG + NLI gốc)", dict(order="rr", dir="chunk", decision="wmax", gate=False, rel_source="rr")),
    ("-RERANK", "− Rerank chunk (dense top-3)", dict(order="dense", dir="window", decision="wmax", gate=False, rel_source="dense")),
    ("B0-RR", "− ESS − Rerank", dict(order="dense", dir="chunk", decision="wmax", gate=False, rel_source="dense")),
    # Mở rộng đã thử — bổ sung thành phần
    ("+CALIB", "+ Hiệu chỉnh ngưỡng", dict(order="rr", dir="window", decision="calib", gate=False, rel_source="rr")),
    ("+GATE", "+ Cổng liên quan", dict(order="rr", dir="window", decision="wmax", gate=True, rel_source="rr")),
    ("+CALIB+GATE", "+ Hiệu chỉnh + cổng", dict(order="rr", dir="window", decision="calib", gate=True, rel_source="rr")),
    ("MEAN", "Tổng hợp trung bình (chunk)", dict(order="rr", dir="chunk", decision="mean", gate=False, rel_source="rr")),
]


# --------------------------------------------------------------------------- #
# Thống kê
# --------------------------------------------------------------------------- #

def bootstrap_ci(y_true, y_pred, rng):
    n = len(y_true)
    accs, f1s = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        accs.append(np.mean(y_true[idx] == y_pred[idx]))
        f1s.append(macro_f1(y_true[idx], y_pred[idx]))
    return {
        "acc_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
    }


def paired_bootstrap_delta(y_true, pred_a, pred_b, rng):
    """P(F1(a) <= F1(b)) qua bootstrap ghép cặp — p-value một phía cho 'a tốt hơn b'."""
    n = len(y_true)
    d_f1, d_acc = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        d_f1.append(macro_f1(y_true[idx], pred_a[idx]) - macro_f1(y_true[idx], pred_b[idx]))
        d_acc.append(np.mean(y_true[idx] == pred_a[idx]) - np.mean(y_true[idx] == pred_b[idx]))
    d_f1, d_acc = np.array(d_f1), np.array(d_acc)
    return {
        "delta_f1_ci95": [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
        "p_f1": float(np.mean(d_f1 <= 0)),
        "delta_acc_ci95": [float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
        "p_acc": float(np.mean(d_acc <= 0)),
    }


def mcnemar_exact(y_true, pred_a, pred_b):
    from scipy.stats import binomtest

    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b01 = int(np.sum(a_ok & ~b_ok))
    b10 = int(np.sum(~a_ok & b_ok))
    if b01 + b10 == 0:
        return {"a_only": b01, "b_only": b10, "p": 1.0}
    return {"a_only": b01, "b_only": b10, "p": float(binomtest(b01, b01 + b10, 0.5).pvalue)}


# --------------------------------------------------------------------------- #
# Độ trễ ước tính (đo trên máy chạy cache)
# --------------------------------------------------------------------------- #

def latency(recs, key, params, mode):
    """Ước tính thời gian/claim theo các giai đoạn thực sự cần cho biến thể."""
    spec = variant_spec(key, mode)
    out = []
    for rec in recs:
        t = rec["t_retrieve"]
        if spec["order"] == "rr" or mode == "oracle":
            t += rec["t_rerank"]
        for p in evidence_positions(rec, spec["order"], mode):
            ch = rec["chunks"][str(p)]
            t += ch["t_nli_chunk"]
            if spec["dir"] == "window":
                n_win = max(1, len(ch["w_nli"]))
                if spec.get("select", True) and params.get("m"):
                    t += ch["t_rr_win"] + ch["t_nli_win"] * min(1.0, params["m"] / n_win)
                else:
                    t += ch["t_nli_win"]
        out.append(t)
    arr = np.array(out)
    return {"mean_sec": float(arr.mean()), "p50_sec": float(np.median(arr)), "p95_sec": float(np.percentile(arr, 95))}


VARIANTS_BY_KEY = {k: (name, spec) for k, name, spec in VARIANTS}


def variant_spec(key, mode):
    spec = dict(VARIANTS_BY_KEY[key][1])
    if mode == "oracle" and spec["order"] == "dense":
        # context vàng không có bước truy xuất: "bỏ rerank" = mọi chunk, không chọn câu bằng cross-encoder
        spec.update(order="all", rel_source="rr", select=False)
    return spec


def fit_predict(train, test, spec, mode):
    if spec["decision"] == "calib":
        params, _ = tune_calibrated(train, spec, mode)
    elif spec["decision"] == "wmax" and (spec["gate"] or len(m_grid(spec)) > 1):
        params, _ = tune_gate_only(train, spec, mode)
    else:
        params = {}
    return predict(test, spec, params, mode), params


def run_cv(recs, mode, k, repeats=5):
    """Lặp `repeats` lần k-fold CV (seed cố định): chọn tham số trên k-1 fold, dự đoán fold còn lại."""
    y = np.array([r["label"] for r in recs], dtype=object)
    scores = {key: {"f1": [], "acc": []} for key, _, _ in VARIANTS}
    deltas = {key: [] for key, _, _ in VARIANTS}
    for rep in range(repeats):
        rng = np.random.default_rng(1000 + rep)
        order = rng.permutation(len(recs))
        folds = np.array_split(order, k)
        oof = {}
        for key, _, _ in VARIANTS:
            spec = variant_spec(key, mode)
            pred = np.empty(len(recs), dtype=object)
            for f in folds:
                train_idx = np.setdiff1d(order, f)
                p, _ = fit_predict([recs[i] for i in train_idx], [recs[i] for i in f], spec, mode)
                pred[f] = p
            oof[key] = pred
            scores[key]["f1"].append(macro_f1(y, pred))
            scores[key]["acc"].append(float(np.mean(pred == y)))
        for key in oof:
            deltas[key].append(scores[key]["f1"][-1] - scores["B0"]["f1"][-1])
    out = {"k": k, "repeats": repeats, "n": len(recs), "variants": {}}
    for key in scores:
        out["variants"][key] = {
            "name": VARIANTS_BY_KEY[key][0],
            "f1_macro": float(np.mean(scores[key]["f1"])),
            "f1_sd": float(np.std(scores[key]["f1"])),
            "accuracy": float(np.mean(scores[key]["acc"])),
            "delta_f1_vs_B0": float(np.mean(deltas[key])),
            "wins_vs_B0": int(sum(d > 0 for d in deltas[key])),
        }
    return out


def load_baselines(dataset, mode, test_claims):
    fname = BASELINE_FILES.get((dataset, mode))
    if not fname:
        return {}
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return {}
    data = json.load(open(path, encoding="utf-8"))
    out = {}
    for key in ("llm_only", "rag_llm"):
        if key not in data["methods"]:
            continue
        details = data["methods"][key]["details"]
        if len(details) < len(test_claims):
            continue
        if any(details[i]["claim"].strip() != test_claims[i] for i in range(len(test_claims))):
            print(f"[warn] {key}: claim không khớp thứ tự test → bỏ qua")
            continue
        out[key] = {
            "pred": np.array([d["label_pred"] for d in details[: len(test_claims)]], dtype=object),
            "latency": data["methods"][key]["latency"],
            "label": data["methods"][key]["method_label"],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Phân tích ablation từ cache")
    parser.add_argument("--dataset", default="viet_medfact")
    parser.add_argument("--mode", default="e2e", choices=["e2e", "oracle"])
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--cv", type=int, default=5, help="Số fold CV trên toàn bộ cache (0 = bỏ qua)")
    args = parser.parse_args()

    base = os.path.join(RESULTS_DIR, "ablation")
    cache = args.cache or os.path.join(base, f"cache_{args.dataset}_{args.mode}.jsonl")
    output = args.output or os.path.join(base, f"summary_{args.dataset}_{args.mode}.json")

    recs = [json.loads(line) for line in open(cache, encoding="utf-8")]
    recs.sort(key=lambda r: r["idx"])
    test = [r for r in recs if r["idx"] < args.test_size]
    dev = [r for r in recs if r["idx"] >= args.test_size]
    print(f"{args.dataset}/{args.mode}: test={len(test)} dev={len(dev)}")
    if not dev:
        raise SystemExit("Chưa có mẫu dev (idx >= 100) trong cache")

    y_test = np.array([r["label"] for r in test], dtype=object)
    rng = np.random.default_rng(SEED)
    results, preds = {}, {}

    for key, name, _ in VARIANTS:
        spec = variant_spec(key, args.mode)
        if spec["decision"] == "calib":
            params, dev_score = tune_calibrated(dev, spec, args.mode)
        elif spec["decision"] == "wmax" and (spec["gate"] or len(m_grid(spec)) > 1):
            params, dev_score = tune_gate_only(dev, spec, args.mode)
        else:
            params, dev_score = {}, None
        pred = predict(test, spec, params, args.mode)
        preds[key] = pred
        m = classification_metrics(list(y_test), list(pred), VERDICTS)
        m.pop("classification_report", None)
        results[key] = {
            "name": name,
            "spec": {k: v for k, v in spec.items()},
            "params": params,
            "dev_f1_acc": dev_score,
            **m,
            **bootstrap_ci(y_test, pred, rng),
            "latency": latency(test, key, params, args.mode),
            "pred_dist": {lab: int(np.sum(pred == lab)) for lab in VERDICTS},
            "per_class": per_class(y_test, pred),
        }

    test_claims = [r["claim"] for r in test]
    for key, b in load_baselines(args.dataset, args.mode, test_claims).items():
        preds[key] = b["pred"]
        m = classification_metrics(list(y_test), list(b["pred"]), VERDICTS)
        m.pop("classification_report", None)
        results[key] = {
            "name": b["label"],
            **m,
            **bootstrap_ci(y_test, b["pred"], rng),
            "latency": b["latency"],
            "pred_dist": {lab: int(np.sum(b["pred"] == lab)) for lab in VERDICTS},
            "per_class": per_class(y_test, b["pred"]),
        }

    pairs = [("FULL", k) for k in preds if k != "FULL"]
    pairs += [("B0", k) for k in ("rag_llm", "llm_only", "B0-RR") if k in preds]
    comparisons = {}
    for a, b in pairs:
        comparisons[f"{a}_vs_{b}"] = {
            **paired_bootstrap_delta(y_test, preds[a], preds[b], rng),
            "mcnemar": mcnemar_exact(y_test, preds[a], preds[b]),
        }

    # Độ nhạy theo m (số câu bằng chứng) — giữ các ngưỡng khác đã chọn
    sensitivity = {}
    full_spec = variant_spec("FULL", args.mode)
    for m in M_GRID:
        params = dict(results["FULL"]["params"], m=m)
        pred = predict(test, full_spec, params, args.mode)
        sensitivity[str(m)] = {"acc": float(np.mean(pred == y_test)), "f1_macro": macro_f1(y_test, pred)}

    cv = run_cv(recs, args.mode, args.cv) if args.cv else None

    payload = {
        "cv": cv,
        "dataset": args.dataset,
        "mode": args.mode,
        "test_size": len(test),
        "dev_size": len(dev),
        "label_dist_test": {lab: int(np.sum(y_test == lab)) for lab in VERDICTS},
        "variants": results,
        "comparisons": comparisons,
        "sensitivity_m": sensitivity,
        "test_details": [
            {"idx": r["idx"], "claim": r["claim"], "gold": r["label"],
             **{k: str(preds[k][i]) for k in preds}}
            for i, r in enumerate(test)
        ],
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n{'Biến thể':<34}{'Acc':>7}{'F1':>7}  {'CI95 F1':<15}{'Lat(s)':>7}  params")
    for key, r in results.items():
        ci = r["f1_ci95"]
        print(f"{r['name']:<34}{r['accuracy']:>7.3f}{r['f1_macro']:>7.3f}  "
              f"[{ci[0]:.2f},{ci[1]:.2f}]   {r['latency']['mean_sec']:>6.2f}  {r.get('params', '')}")
    print("\nSo sánh cặp:")
    for k, c in comparisons.items():
        print(f"  {k:<22} ΔF1 CI95={np.round(c['delta_f1_ci95'], 3)} p_F1={c['p_f1']:.3f} "
              f"ΔAcc CI95={np.round(c['delta_acc_ci95'], 3)} McNemar p={c['mcnemar']['p']:.3f}")
    if cv:
        print(f"\n{cv['repeats']}×{args.cv}-fold CV trên n={cv['n']}:  Acc   F1±sd   ΔF1 vs B0 (thắng/{cv['repeats']})")
        for key, r in cv["variants"].items():
            print(f"  {r['name']:<34}{r['accuracy']:>6.3f} {r['f1_macro']:.3f}±{r['f1_sd']:.3f}  "
                  f"{r['delta_f1_vs_B0']:+.3f} ({r['wins_vs_B0']})")
    print(f"\nĐã lưu -> {output}")


if __name__ == "__main__":
    main()
