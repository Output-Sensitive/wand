"""E1 -- Synthetic ground-truth feature attribution.

Inliers are isotropic Gaussian in R^d; each anomaly deviates on a known
random subset S of features (location shift).  An explanation is correct if
it attributes the anomaly to the features in S.  We compare WAND's two
native explanation modes (witness, gradient) against model-agnostic SHAP
and LIME applied to the *same* WAND scorer, on:

  attribution-AUC : per anomaly, rank features by attribution, score AUC
                    against ground-truth membership in S.  (1.0 = perfect)
  precision@|S|   : fraction of the top-|S| attributed features in S.
  cost            : scorer queries and wall-clock seconds per explanation.

Usage:  python src/exp_e1_synthetic.py [--quick]
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
from core.explain import WANDExplainer  # noqa: E402


# ----------------------------------------------------------------------
def make_data(n_in, k, d, s, shift, seed, correlated=True, kind="axis"):
    """Inliers ~ N(0, Sigma); each anomaly deviates on a random feature
    subset S of size s, with responsible features S as ground truth.

    kind="axis"    : independent shifts on each feature in S (marginally
                     visible -- the regime marginal methods like ECOD own).
    kind="oblique" : a shift along a single random direction supported on S
                     (a feature *combination*).  The per-feature marginal
                     shift is small, so marginal methods go blind, while a
                     directional method can still find the witness.
    Correlated Sigma (default) is the realistic case."""
    rng = np.random.default_rng(seed)
    if correlated:
        # Strong low-rank correlation: a few latent factors + small floor, so
        # the inlier cloud has genuine low-variance directions.  This is what
        # makes correlation ("oblique") anomalies invisible to marginal methods.
        r = max(2, d // 4)
        B = rng.standard_normal((d, r))
        cov = B @ B.T / r + 0.05 * np.eye(d)
        Lc = np.linalg.cholesky(cov)
        std = np.sqrt(np.diag(cov))
    else:
        Lc = np.eye(d)
        std = np.ones(d)
    Xin = rng.standard_normal((n_in, d)) @ Lc.T
    Xout = rng.standard_normal((k, d)) @ Lc.T
    feats = np.zeros((k, d), dtype=bool)
    for i in range(k):
        S = rng.choice(d, size=s, replace=False)
        if kind == "axis":
            sign = rng.choice([-1.0, 1.0], size=s)
            Xout[i, S] += sign * shift * std[S]
        else:                                    # oblique correlation anomaly
            # Displace along the minimum-variance direction of Sigma[S,S]:
            # jointly extreme (large Mahalanobis) but marginally near-normal,
            # so per-feature/marginal methods are blind.  Target Mahalanobis
            # distance = `shift`.
            SigSS = cov[np.ix_(S, S)]
            lam, Vv = np.linalg.eigh(SigSS)      # ascending eigenvalues
            v_dir = Vv[:, 0]                      # smallest-variance direction
            a = shift * np.sqrt(max(lam[0], 1e-9)) * rng.choice([-1.0, 1.0])
            Xout[i, S] += a * v_dir
        feats[i, S] = True
    X = np.vstack([Xin, Xout])
    y = np.r_[np.zeros(n_in), np.ones(k)]
    return X, y, feats


def attribution_metrics(A, feats):
    """A: (k,d) attribution; feats: (k,d) bool ground truth.
    Returns mean attribution-AUC and mean precision@|S| over rows."""
    aucs, precs = [], []
    for a, f in zip(A, feats):
        if f.any() and (~f).any():
            aucs.append(roc_auc_score(f.astype(int), a))
        s = int(f.sum())
        top = np.argsort(a)[::-1][:s]
        precs.append(f[top].mean())
    return float(np.mean(aucs)), float(np.mean(precs))


# ----------------------------------------------------------------------
class QueryCounter:
    """Wraps a scorer to count the number of points scored."""
    def __init__(self, fn):
        self.fn = fn
        self.n = 0

    def __call__(self, X):
        X = np.atleast_2d(X)
        self.n += X.shape[0]
        return self.fn(X)


def shap_attr(expl, Xtrain, Xa, nbg, nsamples):
    import shap
    bg = shap.sample(Xtrain, nbg, random_state=0)
    counter = QueryCounter(expl.score)
    ex = shap.KernelExplainer(counter, bg, silent=True)
    t0 = time.perf_counter()
    sv = ex.shap_values(Xa, nsamples=nsamples, silent=True)
    dt = time.perf_counter() - t0
    sv = np.asarray(sv)
    if sv.ndim == 3:           # (m,d,1) on some versions
        sv = sv[..., 0]
    return np.abs(sv), counter.n, dt


def ecod_fit_attr(X, anom_idx):
    """AD-native baseline: ECOD's built-in per-feature tail scores.

    ECOD decomposes its score over features (decision = sum_j O[:,j]), so the
    rows of O are a native, free attribution.  We fit ECOD unsupervised on X
    (anomalies included) and read O for the flagged rows -- no surrogate, no
    sampling.  Returns (attribution, fitted_model)."""
    from pyod.models.ecod import ECOD
    clf = ECOD().fit(X)
    return np.abs(np.asarray(clf.O)[anom_idx]), clf


def lime_attr(expl, Xtrain, Xa, nsamples):
    from lime.lime_tabular import LimeTabularExplainer
    counter = QueryCounter(expl.score)
    predict = lambda Z: counter(Z)
    lex = LimeTabularExplainer(
        Xtrain, mode="regression", discretize_continuous=False,
        random_state=0, verbose=False)
    d = Xtrain.shape[1]
    out = np.zeros((Xa.shape[0], d))
    t0 = time.perf_counter()
    for i in range(Xa.shape[0]):
        e = lex.explain_instance(Xa[i], predict, num_features=d,
                                 num_samples=nsamples)
        for j, v in e.as_map()[1]:
            out[i, j] = abs(v)
    dt = time.perf_counter() - t0
    return out, counter.n, dt


# ----------------------------------------------------------------------
def run(quick=False, sweep=False):
    if quick:
        configs = [(800, 50, 30, 3, 3.0)]
        seeds = [0]
        n_explain, nbg, shap_ns, lime_ns = 12, 30, 256, 500
    elif sweep:
        # shift sweep at fixed d, s for a faithfulness-vs-strength figure
        configs = [(1000, 100, 30, 5, sh) for sh in (2.0, 2.5, 3.0, 3.5, 4.0)]
        seeds = [0, 1, 2]
        n_explain, nbg, shap_ns, lime_ns = 15, 40, 300, 800
    else:
        configs = []
        for kind in ("axis", "oblique"):
            for d in (50, 100, 200):
                for s in (3, 6):
                    configs.append((1000, d, 30, s, 3.0, kind))
        seeds = [0, 1, 2]
        n_explain, nbg, shap_ns, lime_ns = 15, 40, 300, 800

    # normalise config tuples to include a kind field
    configs = [c if len(c) == 6 else (*c, "axis") for c in configs]

    rows = []
    for (n_in, d, k, s, shift, kind) in configs:
        for seed in seeds:
            X, y, feats = make_data(n_in, k, d, s, shift, seed, kind=kind)
            Xtrain = X[y == 0]
            expl = WANDExplainer(K=1024, seed=seed).fit(X)
            auc = roc_auc_score(y, expl.score(X))
            anom = np.where(y == 1)[0]
            idx = anom[:n_explain]
            Xa, fa = X[idx], feats[:len(idx)]

            # WAND witness (free: computed during scoring)
            t0 = time.perf_counter(); Aw = expl.witness_attribution(Xa)
            tw = time.perf_counter() - t0
            # WAND gradient (one backward per point)
            t0 = time.perf_counter(); Ag = expl.gradient_attribution(Xa)
            tg = time.perf_counter() - t0
            # SHAP / LIME on the same scorer
            As, qs, ts = shap_attr(expl, Xtrain, Xa, nbg, shap_ns)
            Al, ql, tl = lime_attr(expl, Xtrain, Xa, lime_ns)
            # AD-native baseline: ECOD's own per-feature attribution
            t0 = time.perf_counter(); Ae, ecod_clf = ecod_fit_attr(X, idx)
            te = time.perf_counter() - t0
            ecod_auc = roc_auc_score(y, ecod_clf.decision_scores_)

            for name, A, q, dt, det in [
                ("witness",  Aw, 0,  tw, auc),
                ("gradient", Ag, 0,  tg, auc),
                ("ecod",     Ae, 0,  te, ecod_auc),
                ("shap",     As, qs, ts, auc),
                ("lime",     Al, ql, tl, auc),
            ]:
                fauc, prec = attribution_metrics(A, fa)
                rows.append(dict(kind=kind, d=d, s=s, seed=seed,
                                 auc=auc, det_auc=det, method=name,
                                 attr_auc=fauc, prec=prec,
                                 queries=q / len(idx),
                                 sec_per_expl=dt / len(idx)))
            print(f"[{kind:7s} d={d:3d} s={s} seed={seed}] "
                  f"acAUC={auc:.3f} ecodAUC={ecod_auc:.3f}  "
                  + "  ".join(f"{r['method']}:{r['attr_auc']:.3f}"
                              for r in rows[-5:]), flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    tag = "e1_synthetic_quick" if quick else ("e1_sweep" if sweep else "e1_synthetic")
    out = ROOT / "results" / f"{tag}.csv"
    df.to_csv(out, index=False)
    print("\n=== E1 summary (mean over configs/seeds) ===")
    keys = ["kind", "method"] if "kind" in df.columns else ["method"]
    g = df.groupby(keys).agg(
        det_auc=("det_auc", "mean"),
        attr_auc=("attr_auc", "mean"), prec=("prec", "mean"),
        queries=("queries", "mean"), sec=("sec_per_expl", "mean"))
    print(g.round(4).to_string())
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    run(**vars(ap.parse_args()))
