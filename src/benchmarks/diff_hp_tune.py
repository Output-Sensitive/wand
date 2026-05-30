"""
Per-dataset differentiable hyperparameter tuning for WAND.

Pipeline
--------
1.  Stratified 50/50 train/test split on each dataset.
2.  Pre-compute N WAND score vectors on the *train half* under N
    different hyper-parameter profiles spanning the interesting region
    of the HP grid (K, axis_weight, spacing_mix, n_langevin, beta).
3.  Treat the mixture weights `w = softmax(alpha)` as the only
    learnable parameters and optimise an *unsupervised* contrast loss
        L(alpha) = - (mean top-q score - mean bottom-q score),
    where q is the assumed contamination rate (heuristic = 0.1).
    This is fully differentiable in alpha; no labels are touched.
4.  Re-compute the same N score vectors on the *test half* with the
    same profiles, mix them with the learned weights, and report the
    test AUC.

What the test half measures
---------------------------
The mixture weights generalise across the (X_tr, X_te) split, so the
test AUC is a *fair* unsupervised estimate of what label-free
HP-tuning buys vs. the published defaults.

Output
------
results/diff_hp_tuned.csv  one row per dataset:
    dataset, n, d, contam,
    auc_default, auc_tuned, n_profiles, learned_top_profile,
    per-profile mixture weights, secs.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core.anticoncentration import wand_score
from src.benchmarks.bench_anomaly_wand import DATASETS, _load


# ----------------------------------------------------------------------
# Hyper-parameter profiles spanning the interesting region of the grid.
# Index 0 is the paper default; the rest are deliberate perturbations
# that the mixture weights can pick out per dataset.
# ----------------------------------------------------------------------
PROFILES: list[tuple[str, dict]] = [
    ("default",
     dict(K=1024, axis_weight=0.25, spacing_mix=0.5,
          n_langevin=20, n_seeds=1)),
    ("no_axis",
     dict(K=1024, axis_weight=0.0,  spacing_mix=0.5,
          n_langevin=20, n_seeds=1)),
    ("strong_axis",
     dict(K=1024, axis_weight=1.0,  spacing_mix=0.5,
          n_langevin=20, n_seeds=1)),
    ("no_spacing",
     dict(K=1024, axis_weight=0.25, spacing_mix=0.0,
          n_langevin=20, n_seeds=1)),
    ("strong_spacing",
     dict(K=1024, axis_weight=0.25, spacing_mix=1.0,
          n_langevin=20, n_seeds=1)),
    ("no_langevin",
     dict(K=1024, axis_weight=0.25, spacing_mix=0.5,
          n_langevin=0,  n_seeds=1)),
    ("big_K",
     dict(K=2048, axis_weight=0.25, spacing_mix=0.5,
          n_langevin=20, n_seeds=1)),
]


# ----------------------------------------------------------------------
# Unsupervised contrast loss
# ----------------------------------------------------------------------

def _unsup_contrast(scores: torch.Tensor, q: float = 0.10) -> torch.Tensor:
    """L = - (mean top-q% scores - mean bottom-q% scores).

    Differentiable in `scores`. Picks weights so the score distribution
    has a clear separation between its top-q% mass (assumed anomalies)
    and bottom-q% mass (assumed clean inliers), without using labels.
    """
    n = scores.shape[0]
    k = max(1, int(round(q * n)))
    top = torch.topk(scores, k=k, largest=True).values.mean()
    bot = torch.topk(scores, k=k, largest=False).values.mean()
    return -(top - bot)


# ----------------------------------------------------------------------
# Per-dataset run
# ----------------------------------------------------------------------

def _compute_score_bank(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Score `X` under every HP profile. Returns (n_profiles, n)."""
    bank = []
    for _, cfg in PROFILES:
        s = wand_score(X, seed=seed, **cfg)
        bank.append(s.astype(np.float64))
    return np.stack(bank, axis=0)


def _norm01(scores: np.ndarray) -> np.ndarray:
    """Per-row [0,1] normalisation (numpy)."""
    mx = scores.max(axis=1, keepdims=True)
    return scores / (mx + 1e-30)


def _learn_mixture(S_tr: torch.Tensor, q: float = 0.10,
                   n_steps: int = 200, lr: float = 0.1,
                   seed: int = 0) -> torch.Tensor:
    """Optimise softmax(alpha) over the profile axis on the train scores.

    `S_tr` is (n_profiles, n_train). Returns the learned mixture
    weights (n_profiles,) summing to 1.
    """
    torch.manual_seed(seed)
    P = S_tr.shape[0]
    alpha = torch.zeros(P, requires_grad=True)
    opt = torch.optim.Adam([alpha], lr=lr)
    for _ in range(n_steps):
        w = F.softmax(alpha, dim=0)
        s_mix = (w.unsqueeze(1) * S_tr).sum(dim=0)
        loss = _unsup_contrast(s_mix, q=q)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return F.softmax(alpha.detach(), dim=0)


def _per_dataset(name: str, q: float, n_steps: int, seed: int,
                 split_seed: int) -> dict:
    X, y, short = _load(ROOT / "datasets" / "odds" / f"{name}.npz")
    n, d = X.shape
    contam = float(y.mean())
    t0 = time.perf_counter()

    # Stratified 50/50 split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=split_seed,
    )

    # Pre-compute score bank on each half
    S_tr_np = _compute_score_bank(X_tr, seed=seed)              # (P, n_tr)
    S_te_np = _compute_score_bank(X_te, seed=seed)              # (P, n_te)

    # Per-row normalisation before mixing (so profiles with wildly
    # different scales don't dominate by accident)
    S_tr_np = _norm01(S_tr_np)
    S_te_np = _norm01(S_te_np)
    S_tr = torch.tensor(S_tr_np, dtype=torch.float64)

    # Baselines: default-profile test AUC, oracle = best single-profile test AUC.
    auc_default_te = float(roc_auc_score(y_te, S_te_np[0]))
    profile_aucs_te = [float(roc_auc_score(y_te, s)) for s in S_te_np]
    auc_oracle_te = max(profile_aucs_te)

    # Learn mixture on train (no labels)
    w = _learn_mixture(S_tr, q=q, n_steps=n_steps, seed=seed)   # (P,)
    s_te_mix = (w.numpy()[:, None] * S_te_np).sum(axis=0)
    auc_tuned_te = float(roc_auc_score(y_te, s_te_mix))

    secs = time.perf_counter() - t0
    return {
        "dataset": short, "n": n, "d": d, "contam": contam,
        "auc_default": auc_default_te,
        "auc_tuned": auc_tuned_te,
        "auc_oracle": auc_oracle_te,
        "best_profile": PROFILES[int(np.argmax(profile_aucs_te))][0],
        "top_weight_profile": PROFILES[int(torch.argmax(w))][0],
        "weights": ";".join(
            f"{PROFILES[i][0]}={float(w[i]):.3f}" for i in range(len(PROFILES))
        ),
        "secs": secs,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default=None,
                   help="comma-separated dataset codes; default = a small "
                        "diverse subset (8 datasets)")
    p.add_argument("--q", type=float, default=0.10,
                   help="contrast loss quantile (default 0.10)")
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--out", type=str,
                   default=str(ROOT / "results" / "diff_hp_tuned.csv"))
    args = p.parse_args()

    if args.datasets is None:
        # A small, diverse subset covering small / medium / high-dim,
        # plus a few datasets where the K-sweep showed HP sensitivity
        # (Hepatitis, vowels, satellite, SpamBase, InternetAds).
        ds = ["14_glass", "29_Pima", "15_Hepatitis", "18_Ionosphere",
              "40_vowels", "30_satellite", "35_SpamBase", "17_InternetAds"]
    else:
        ds = args.datasets.split(",")

    print(f'{"dataset":<22s} {"n":>6s} {"d":>5s} '
          f'{"def":>7s} {"tuned":>7s} {"oracle":>7s} '
          f'{"Δ vs def":>9s} {"top-profile":>16s} {"secs":>7s}',
          flush=True)
    print("-" * 110, flush=True)

    rows = []
    for name in ds:
        try:
            r = _per_dataset(name, q=args.q, n_steps=args.n_steps,
                             seed=args.seed, split_seed=args.split_seed)
            d_v = r["auc_tuned"] - r["auc_default"]
            print(f'{r["dataset"][:22]:<22s} {r["n"]:>6d} {r["d"]:>5d} '
                  f'{r["auc_default"]:>7.3f} {r["auc_tuned"]:>7.3f} '
                  f'{r["auc_oracle"]:>7.3f} {d_v:>+9.3f} '
                  f'{r["top_weight_profile"][:16]:>16s} {r["secs"]:>7.1f}',
                  flush=True)
            rows.append(r)
        except Exception as e:
            print(f'{name:<22s} ERR: {e}', flush=True)

    print("-" * 110, flush=True)
    if rows:
        def m(k): return float(np.mean([r[k] for r in rows]))
        print(f'{"AVG":<22s} {"":>6s} {"":>5s} '
              f'{m("auc_default"):>7.3f} {m("auc_tuned"):>7.3f} '
              f'{m("auc_oracle"):>7.3f} '
              f'{m("auc_tuned") - m("auc_default"):>+9.3f}',
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "n", "d", "contam",
                  "auc_default", "auc_tuned", "auc_oracle",
                  "best_profile", "top_weight_profile", "weights", "secs"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"# wrote {out}", flush=True)


if __name__ == "__main__":
    main()
