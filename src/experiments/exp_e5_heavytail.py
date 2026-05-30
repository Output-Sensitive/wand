"""E5 -- Heavy-tailed / non-sub-Gaussian robustness.

Assumption~1 asks inliers to be sub-Gaussian; real tabular data is often
heavy-tailed.  We claim the median/MAD calibration rescales moderate heavy
tails back toward the sub-Gaussian regime.  Here we test it directly:
inliers are multivariate Student-t with degrees of freedom swept from
Gaussian (df=inf) down to Cauchy (df=1), anomalies are axis shifts in
robust (MAD) units, and we report

  detection ROC-AUC : WAND (median/MAD), Isolation Forest, ECOD.
  attribution-AUC   : WAND witness, under the same tails.

The point: WAND degrades gracefully as the tails get heavier and stays
clearly ahead of Isolation Forest and ECOD, and its witness explanations
remain accurate -- so the sub-Gaussian assumption is not fragile in
practice for the median/MAD-calibrated detector.

Usage:  python src/exp_e5_heavytail.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from core.explain import WANDExplainer                # noqa: E402
from experiments.exp_e1_synthetic import attribution_metrics    # noqa: E402


def make_t(n_in, k, d, s, shift, df, seed):
    """Multivariate Student-t inliers (correlated), axis-shift anomalies in
    robust (MAD) units so the shift is comparable across df."""
    rng = np.random.default_rng(seed)
    Wm = rng.standard_normal((d, d)) / np.sqrt(d)
    Lc = np.linalg.cholesky(Wm @ Wm.T + 0.3 * np.eye(d))
    Z = rng.standard_normal((n_in + k, d)) @ Lc.T
    if np.isfinite(df):
        g = rng.chisquare(df, size=(n_in + k, 1)) / df
        Z = Z / np.sqrt(g)
    Xin = Z[:n_in]
    mad = np.median(np.abs(Xin - np.median(Xin, 0)), 0) * 1.4826 + 1e-9
    Xout = Z[n_in:]
    feats = np.zeros((k, d), bool)
    for i in range(k):
        S = rng.choice(d, s, replace=False)
        Xout[i, S] += rng.choice([-1.0, 1.0], s) * shift * mad[S]
        feats[i, S] = True
    X = np.vstack([Xin, Xout])
    y = np.r_[np.zeros(n_in), np.ones(k)]
    return X, y, feats


def run(seeds=(0, 1, 2)):
    from pyod.models.iforest import IForest
    from pyod.models.ecod import ECOD
    dfs = [np.inf, 8, 4, 3, 2, 1]
    rows = []
    for df in dfs:
        acc = {m: [] for m in ["wand", "iforest", "ecod"]}
        wit = []
        for seed in seeds:
            X, y, feats = make_t(1000, 30, 20, 3, 6.0, df, seed)
            ex = WANDExplainer(K=1024, seed=seed, robust=True).fit(X)
            acc["wand"].append(roc_auc_score(y, ex.score(X)))
            acc["iforest"].append(roc_auc_score(y, IForest(random_state=seed).fit(X).decision_scores_))
            acc["ecod"].append(roc_auc_score(y, ECOD().fit(X).decision_scores_))
            idx = np.where(y == 1)[0]
            wit.append(attribution_metrics(ex.witness_attribution(X[idx]), feats)[0])
        row = dict(df=("inf" if not np.isfinite(df) else int(df)))
        for m in acc:
            row[m] = float(np.mean(acc[m]))
        row["witness_attr_auc"] = float(np.mean(wit))
        rows.append(row)
        print(f"df={row['df']:>3} | AUC wand={row['wand']:.3f} "
              f"iforest={row['iforest']:.3f} ecod={row['ecod']:.3f} "
              f"| witness attr-AUC={row['witness_attr_auc']:.3f}", flush=True)

    import pandas as pd
    df_ = pd.DataFrame(rows)
    out = ROOT / "results" / "e5_heavytail.csv"
    df_.to_csv(out, index=False)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11,
                         "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10})
    x = np.arange(len(rows))
    labels = [str(r["df"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for m, c, lab in [("wand", "#2c7fb8", "WAND"),
                      ("iforest", "#41b6c4", "IForest"),
                      ("ecod", "#d95f0e", "ECOD")]:
        ax[0].plot(x, [r[m] for r in rows], marker="o", color=c, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_xlabel("Student-$t$ d.o.f. (left = heavier tails)")
    ax[0].set_ylabel("detection ROC-AUC")
    ax[0].set_title("(a) detection under heavy tails", fontsize=11)
    ax[0].legend(fontsize=10, loc="lower right"); ax[0].grid(alpha=0.3)
    ax[0].invert_xaxis()

    ax[1].plot(x, [r["witness_attr_auc"] for r in rows], marker="s", color="#2c7fb8")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_xlabel("Student-$t$ d.o.f. (left = heavier tails)")
    ax[1].set_ylabel("witness attribution-AUC")
    ax[1].set_title("(b) explanation quality under heavy tails", fontsize=11)
    ax[1].grid(alpha=0.3); ax[1].invert_xaxis()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "heavytail.pdf", bbox_inches="tight")
    print(f"\nsaved -> {out} and figures/heavytail.pdf")


if __name__ == "__main__":
    run()
