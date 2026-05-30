"""E4 -- Qualitative case study with named features.

ADBench's .npz files carry no feature names, so for an interpretable case
study we use the Breast Cancer Wisconsin (Diagnostic) data, which has 30
clinically named features.  We build an anomaly-detection task -- benign
cases as inliers, a small set of malignant cases as anomalies -- and show
that WAND's directional-witness attribution localises a flagged
malignant case to the clinically recognised malignancy markers (worst /
large radius, perimeter, area, concave points), with no labels used.

Outputs:
  - figures/e4_casestudy.pdf : top witness features for one case +
    aggregate attribution over all detected cases (witness vs SHAP).
  - prints the ranked features.

Usage:  python src/exp_e4_casestudy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from core.explain import WANDExplainer                          # noqa: E402
from experiments.exp_e1_synthetic import shap_attr                        # noqa: E402


def build_task(n_anom=20, seed=0):
    data = load_breast_cancer()
    X, t, names = data.data, data.target, list(data.feature_names)
    # sklearn: target 0 = malignant, 1 = benign.  Inliers = benign.
    rng = np.random.default_rng(seed)
    benign = X[t == 1]
    malig = X[t == 0]
    sel = rng.choice(len(malig), n_anom, replace=False)
    Xa = malig[sel]
    Xall = np.vstack([benign, Xa])
    y = np.r_[np.zeros(len(benign)), np.ones(n_anom)]
    # standardise (fit on the mixed sample, as in deployment)
    mu, sd = Xall.mean(0), Xall.std(0)
    Xall = (Xall - mu) / sd
    return Xall, y, names


def main():
    X, y, names = build_task()
    expl = WANDExplainer(K=1024, seed=0).fit(X)
    s = expl.score(X)
    auc = roc_auc_score(y, s)
    print(f"Breast-cancer AD task: n={X.shape[0]} d={X.shape[1]} AUC={auc:.3f}")

    anom_idx = np.where(y == 1)[0]
    # the most confidently flagged malignant case
    case = anom_idx[np.argmax(s[anom_idx])]
    Aw_case = expl.witness_attribution(X[case:case + 1])[0]
    order = np.argsort(Aw_case)[::-1]
    print(f"\nMost-flagged malignant case (score rank "
          f"{int((s > s[case]).sum()) + 1}/{len(s)}):")
    for j in order[:6]:
        print(f"   {names[j]:30s}  {Aw_case[j]:.3f}")

    # aggregate witness attribution over all detected malignant cases
    Aw = expl.witness_attribution(X[anom_idx])
    agg_w = Aw.mean(0)
    try:
        As, _, _ = shap_attr(expl, X[y == 0], X[anom_idx],
                             nbg=50, nsamples=600)
        agg_s = (As / np.abs(As).sum(1, keepdims=True).clip(1e-12)).mean(0)
    except Exception as e:
        print("shap failed:", e); agg_s = np.zeros_like(agg_w)

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})

    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.4))
    # (a) single case top-8 witness
    k = 8
    o = order[:k][::-1]
    ax[0].barh(range(k), Aw_case[o], color="#2c7fb8")
    ax[0].set_yticks(range(k))
    ax[0].set_yticklabels([names[j] for j in o], fontsize=12)
    ax[0].set_xlabel("witness attribution")
    ax[0].set_title("(a) one flagged malignant case", fontsize=12)

    # (b) aggregate witness vs shap, top features by witness
    ko = np.argsort(agg_w)[::-1][:k][::-1]
    yy = np.arange(k)
    ax[1].barh(yy + 0.2, agg_w[ko], height=0.4, color="#2c7fb8", label="WAND witness")
    ax[1].barh(yy - 0.2, agg_s[ko], height=0.4, color="#d95f0e", label="SHAP")
    ax[1].set_yticks(yy)
    ax[1].set_yticklabels([names[j] for j in ko], fontsize=12)
    ax[1].set_xlabel("mean attribution")
    ax[1].set_title("(b) aggregate over detected cases", fontsize=12)
    ax[1].legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    out = ROOT / "figures" / "e4_casestudy.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nsaved figure -> {out}")

    print("\nTop-8 aggregate witness features:")
    for j in np.argsort(agg_w)[::-1][:8]:
        print(f"   {names[j]:30s}  {agg_w[j]:.3f}")


if __name__ == "__main__":
    main()
