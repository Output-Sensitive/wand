"""E6 -- Comparison with deep unsupervised detectors.

Reviewers asked for a comparison with recent deep / self-supervised anomaly
detectors.  We add three torch-based deep unsupervised baselines from
PyOD -- AutoEncoder, Variational AE, and Deep SVDD -- trained unsupervised
on the contaminated sample (no clean training set), and compare detection
ROC-AUC against \\textsc{WAND} and Isolation Forest on a representative
ADBench subset.  (Consistent with the ADBench study, deep tabular detectors
do not dominate strong shallow detectors; \\textsc{WAND} stays competitive
while additionally being explainable-by-design.)

Usage:  python src/exp_e6_deep.py
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from core.explain import WANDExplainer            # noqa: E402

SUBSET = ["breastw", "Pima", "WBC", "WDBC", "wine", "Ionosphere", "cardio",
          "Cardiotocography", "thyroid", "satellite", "satimage-2",
          "mammography", "optdigits", "musk", "pendigits", "letter"]


def load(short, max_n=10000, seed=0):
    import glob
    path = glob.glob(str(ROOT / "datasets" / "odds" / f"*_{short}.npz"))[0]
    d = np.load(path)
    X = d["X"].astype(np.float64); y = d["y"].ravel().astype(int)
    keep = X.std(0) > 1e-12; X = X[:, keep]
    X = (X - X.mean(0)) / X.std(0)
    if len(X) > max_n:
        rng = np.random.default_rng(seed)
        i = rng.choice(len(X), max_n, replace=False); X, y = X[i], y[i]
    return X, y


def deep_auc(name, X, y, seed):
    n, d = X.shape
    try:
        if name == "AutoEncoder":
            from pyod.models.auto_encoder import AutoEncoder
            clf = AutoEncoder(epoch_num=30, batch_size=min(256, n),
                              random_state=seed, verbose=0)
        elif name == "VAE":
            from pyod.models.vae import VAE
            clf = VAE(epoch_num=30, batch_size=min(256, n),
                      random_state=seed, verbose=0)
        elif name == "DeepSVDD":
            from pyod.models.deep_svdd import DeepSVDD
            clf = DeepSVDD(n_features=d, epochs=30, batch_size=min(256, n),
                           random_state=seed, verbose=0)
        clf.fit(X)
        return roc_auc_score(y, clf.decision_scores_)
    except Exception as e:
        print(f"   {name} failed: {str(e)[:90]}", flush=True)
        return float("nan")


def run(seed=0):
    from pyod.models.iforest import IForest
    rows = []
    for short in SUBSET:
        try:
            X, y = load(short)
        except Exception as e:
            print(f"[skip] {short}: {e}"); continue
        r = {"dataset": short, "n": len(X), "d": X.shape[1]}
        r["WAND"] = roc_auc_score(y, WANDExplainer(K=1024, seed=seed).fit(X).score(X))
        r["IForest"] = roc_auc_score(y, IForest(random_state=seed).fit(X).decision_scores_)
        for m in ["AutoEncoder", "VAE", "DeepSVDD"]:
            r[m] = deep_auc(m, X, y, seed)
        rows.append(r)
        print("[%-14s %6dx%-3d] " % (short, r["n"], r["d"])
              + "  ".join(f"{k}:{r[k]:.3f}" for k in
                          ["WAND", "IForest", "AutoEncoder", "VAE", "DeepSVDD"]),
              flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "e6_deep.csv"
    df.to_csv(out, index=False)
    cols = ["WAND", "IForest", "AutoEncoder", "VAE", "DeepSVDD"]
    print("\n=== mean ROC-AUC ===")
    print(df[cols].mean().round(3).to_string())
    print("\n=== mean rank (1=best) ===")
    print(df[cols].rank(axis=1, ascending=False).mean().round(2).to_string())
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    run()
