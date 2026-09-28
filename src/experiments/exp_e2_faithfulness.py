"""E2 -- Faithfulness on real ADBench data (label-free).

For each dataset we explain the points WAND flags (top-N by score) and
measure how faithful each explanation is to the scorer, with no labels:

  deletion  : mask the most-attributed features (replace by the per-feature
              median) one fraction at a time; a faithful explanation makes
              the score collapse quickly.  Reported as normalised AUC
              (LOWER = better).
  insertion : start from the all-median reference and re-insert features in
              attribution order; the score should rise quickly.  Normalised
              AUC (HIGHER = better).
  faithful  : insertion_auc - deletion_auc  (HIGHER = better).

All explainers (witness, gradient, SHAP, LIME, random) operate on the same
WAND scorer, so the comparison is apples-to-apples.  We also log the
witness/gradient rank agreement and per-explanation cost.

Usage:  python src/exp_e2_faithfulness.py [--datasets a,b,c] [--nexpl 20]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from core.explain import WANDExplainer, rank_correlation       # noqa: E402
from experiments.exp_e1_synthetic import shap_attr, lime_attr            # noqa: E402


# ----------------------------------------------------------------------
def load_npz(path: Path, max_n: int = 20000, seed: int = 0):
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float64)
    y = np.asarray(d["y"]).ravel().astype(int)
    keep = X.std(axis=0) > 1e-12
    X = X[:, keep]
    X = (X - X.mean(0)) / X.std(0)
    if X.shape[0] > max_n:                       # subsample for tractable fit
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], max_n, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def _curve(score_fn, x, ref, order, steps, mode):
    d = len(x)
    fracs = np.linspace(0.0, 1.0, steps)
    batch = np.empty((steps, d))
    for t, f in enumerate(fracs):
        kk = int(round(f * d))
        if mode == "deletion":
            xx = x.copy(); xx[order[:kk]] = ref[order[:kk]]
        else:                                    # insertion
            xx = ref.copy(); xx[order[:kk]] = x[order[:kk]]
        batch[t] = xx
    s = score_fn(batch)
    # RISE-style normalisation: divide by the original point's score s(x)
    # (a single stable per-point scale) and clip to [0,1].  Bounded, so no
    # single dataset can dominate the mean.  s(x) is curve endpoint with
    # all features present: index 0 for deletion, -1 for insertion.
    s0 = s[0] if mode == "deletion" else s[-1]
    denom = s0 if abs(s0) > 1e-12 else 1.0
    curve = np.clip(s / denom, 0.0, 1.0)
    # deletion: 1 -> ~0, lower area = faithful; insertion: ~0 -> 1, higher = faithful
    return float(np.sum((curve[1:] + curve[:-1]) / 2.0 * np.diff(fracs)))


def faithfulness(score_fn, Xa, ref, attr, steps=11):
    dele, ins = [], []
    for x, a in zip(Xa, attr):
        order = np.argsort(a)[::-1]
        dele.append(_curve(score_fn, x, ref, order, steps, "deletion"))
        ins.append(_curve(score_fn, x, ref, order, steps, "insertion"))
    return float(np.mean(dele)), float(np.mean(ins))


# ----------------------------------------------------------------------
def run(datasets=None, nexpl=20, nbg=40, shap_ns=300, lime_ns=600, K=1024):
    ddir = ROOT / "datasets" / "odds"
    paths = sorted(ddir.glob("*.npz"))
    if datasets:
        want = set(datasets.split(","))
        paths = [p for p in paths if p.stem.split("_", 1)[1] in want]

    rows = []
    for p in paths:
        short = p.stem.split("_", 1)[1]
        try:
            X, y = load_npz(p)
        except Exception as e:
            print(f"[skip] {short}: {e}", flush=True); continue
        n, d = X.shape
        if d < 3 or n < 50:
            print(f"[skip] {short}: too small ({n}x{d})", flush=True); continue

        expl = WANDExplainer(K=K, seed=0).fit(X)
        s = expl.score(X)
        auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")
        ref = np.median(X, axis=0)
        idx = np.argsort(s)[::-1][:nexpl]        # top-scored = flagged points
        Xa = X[idx]
        Xtrain = X[s <= np.median(s)]            # "inlier" reference pool

        t0 = time.perf_counter(); Aw = expl.witness_attribution(Xa); tw = time.perf_counter() - t0
        t0 = time.perf_counter(); Ag = expl.gradient_attribution(Xa); tg = time.perf_counter() - t0
        try:
            As, qs, ts = shap_attr(expl, Xtrain, Xa, min(nbg, len(Xtrain)),
                                   min(shap_ns, 2 * d + 256))
        except Exception as e:
            print(f"   shap failed on {short}: {e}", flush=True)
            As, qs, ts = np.random.rand(*Xa.shape), float("nan"), float("nan")
        Al, ql, tl = lime_attr(expl, Xtrain, Xa, lime_ns)
        Ar = np.random.default_rng(0).random(Xa.shape)

        agree = rank_correlation(Aw, Ag)
        for name, A, q, dt in [("witness", Aw, 0, tw), ("gradient", Ag, 0, tg),
                               ("shap", As, qs, ts), ("lime", Al, ql, tl),
                               ("random", Ar, 0, 0.0)]:
            dele, ins = faithfulness(expl.score, Xa, ref, A)
            rows.append(dict(dataset=short, n=n, d=d, auc=round(auc, 3),
                             method=name, deletion=round(dele, 4),
                             insertion=round(ins, 4),
                             faithful=round(ins - dele, 4),
                             queries=q / max(len(idx), 1),   # per-point (q is batch total)
                             sec_per_expl=dt / max(len(idx), 1),
                             wit_grad_agree=round(agree, 3)))

        # AD-native baseline: ECOD explains its OWN detector (per-feature O),
        # so it is evaluated self-consistently against the ECOD score and its
        # own flagged points -- the fair "native explainer" comparator.
        try:
            from pyod.models.ecod import ECOD
            ec = ECOD().fit(X)
            e_idx = np.argsort(ec.decision_scores_)[::-1][:nexpl]
            A_ec = np.abs(np.asarray(ec.O)[e_idx])
            dele, ins = faithfulness(lambda Z: ec.decision_function(np.atleast_2d(Z)),
                                     X[e_idx], ref, A_ec)
            rows.append(dict(dataset=short, n=n, d=d, auc=round(auc, 3),
                             method="ecod", deletion=round(dele, 4),
                             insertion=round(ins, 4), faithful=round(ins - dele, 4),
                             queries=0, sec_per_expl=0.0, wit_grad_agree=round(agree, 3)))
        except Exception as e:
            print(f"   ecod failed on {short}: {e}", flush=True)

        line = "  ".join(f"{r['method']}:{r['faithful']:+.3f}" for r in rows[-6:])
        print(f"[{short:14s} {n:6d}x{d:<4d} AUC={auc:.3f}] faithful(ins-del)  {line}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "e2_faithfulness.csv"
    df.to_csv(out, index=False)
    print("\n=== E2 summary (mean over datasets) ===")
    g = df.groupby("method").agg(deletion=("deletion", "mean"),
                                 insertion=("insertion", "mean"),
                                 faithful=("faithful", "mean"),
                                 queries=("queries", "mean"),
                                 sec=("sec_per_expl", "mean"))
    print(g.round(4).to_string())
    print(f"\nmean witness/gradient agreement: {df['wit_grad_agree'].mean():.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None, help="comma-separated short names")
    ap.add_argument("--nexpl", type=int, default=20)
    run(**vars(ap.parse_args()))
