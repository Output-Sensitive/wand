"""
Gradient-based tuning of WAND's lambda (axis_weight) per dataset,
using a label-free top-q gap loss. POC on a representative dataset
subset; shows that the soft-extreme differentiability of WAND
(Sec. 5.3 of the paper) extends naturally to hyperparameter selection.

Setup per dataset:
  1. Stratified 50/50 split.
  2. On the TRAIN half, run WAND twice:
       s_train(0) = wand_score(X_tr, axis_weight=0)
       s_train(1) = wand_score(X_tr, axis_weight=1)
     The published combination is exactly linear in lambda:
       s_train(lambda) = s_train(0) + lambda * [s_train(1) - s_train(0)]
     so a single nn.Parameter lambda suffices to backprop through.
  3. Optimise the label-free top-q gap loss
       L(lambda) = -[ mean(top-q% of s_train(lambda))
                    - mean(bottom-(1-q)% of s_train(lambda)) ]
     for ~100 Adam steps (q = 0.10 by default, label-free heuristic).
  4. Apply the learned lambda to the TEST half exactly the same way
     (run lambda=0 / lambda=1, interpolate) and report test AUC.
  5. Compare against the published default lambda = 0.25.

Outputs:
  results/diff_hp_lambda.csv -- one row per dataset with default vs
                                tuned test AUC and the learned lambda.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
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
from src.benchmarks.bench_anomaly_wand import _load


DEFAULT_DS = [
    "39_vertebral",     # inverted geometry -- structurally hard
    "45_wine",          # small, baseline-strong
    "6_cardio",         # medium, well-behaved
    "30_satellite",     # medium, WAND wins
    "40_vowels",        # spacing-sensitive
    "2_annthyroid",     # medium, spacing-hurts
]


def _topq_gap_loss(s: torch.Tensor, q: float = 0.10) -> torch.Tensor:
    """Label-free analog of AUC: gap between top-q and bottom-(1-q) means.

    Differentiable; uses torch.topk for the top selection and a soft
    masked mean for the rest. Maximising this gap encourages a
    discriminative score distribution.
    """
    n = s.shape[0]
    k = max(1, int(round(q * n)))
    top_vals, _ = torch.topk(s, k=k, largest=True, sorted=False)
    bot_vals, _ = torch.topk(s, k=n - k, largest=False, sorted=False)
    return -(top_vals.mean() - bot_vals.mean())


def _two_score(X: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (s_lambda0, s_lambda1) for the dataset X.

    Each is a per-point WAND score, computed with axis_weight=0 and
    axis_weight=1 respectively. The published combination is
    `score(lambda) = s_lambda0 + lambda * (s_lambda1 - s_lambda0)`, so
    we obtain a closed-form linear parameterisation in lambda from just
    two forward passes.
    """
    s0 = wand_score(X, K=1024, seed=seed, axis_weight=0.0, n_seeds=1)
    s1 = wand_score(X, K=1024, seed=seed, axis_weight=1.0, n_seeds=1)
    return s0, s1


def _tune_lambda(s0_tr: np.ndarray, s1_tr: np.ndarray,
                 init: float = 0.25, n_steps: int = 100,
                 lr: float = 0.05, q: float = 0.10,
                 ) -> tuple[float, list[float]]:
    """Optimise lambda via gradient descent on the top-q gap loss.

    Returns the learned lambda (after a softplus clamp to >= 0) and the
    per-step loss trace.
    """
    s0_t = torch.tensor(s0_tr, dtype=torch.float64)
    s1_t = torch.tensor(s1_tr, dtype=torch.float64)
    # log_lambda for positivity reparam (softplus would also work).
    log_lam = torch.tensor(math.log(max(init, 1e-3)), dtype=torch.float64,
                           requires_grad=True)
    opt = torch.optim.Adam([log_lam], lr=lr)
    trace = []
    for step in range(n_steps):
        lam = torch.exp(log_lam)
        s = s0_t + lam * (s1_t - s0_t)
        loss = _topq_gap_loss(s, q=q)
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append(float(loss.detach()))
    return float(torch.exp(log_lam).detach()), trace


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str,
                   default=",".join(DEFAULT_DS))
    p.add_argument("--q", type=float, default=0.10,
                   help="contamination prior for the gap loss")
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=0,
                   help="random seed for the 50/50 stratified split")
    p.add_argument("--out", type=str,
                   default=str(ROOT / "results" / "diff_hp_lambda.csv"))
    args = p.parse_args()

    print(f'{"dataset":<14s} {"n":>6s} {"d":>4s} {"lam*":>7s}'
          f' {"AUC@0.25":>10s} {"AUC@lam*":>10s} {"ΔAUC":>8s}', flush=True)
    print("-" * 64, flush=True)

    rows = []
    for name in args.datasets.split(","):
        path = ROOT / "datasets" / "odds" / f"{name}.npz"
        if not path.exists():
            print(f"[skip] missing {path}", flush=True)
            continue
        X, y, short = _load(path)
        n, d = X.shape
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.5, stratify=y,
                random_state=args.split_seed,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.5, random_state=args.split_seed,
            )

        # --- Two base scores per half ---
        s0_tr, s1_tr = _two_score(X_tr, seed=args.seed)
        s0_te, s1_te = _two_score(X_te, seed=args.seed)

        # --- Gradient-tune lambda on train half ---
        lam_star, trace = _tune_lambda(
            s0_tr, s1_tr, init=0.25,
            n_steps=args.n_steps, lr=args.lr, q=args.q,
        )

        # --- Score test half at default and tuned lambda ---
        s_default_te = s0_te + 0.25 * (s1_te - s0_te)
        s_tuned_te = s0_te + lam_star * (s1_te - s0_te)
        auc_default = float(roc_auc_score(y_te, s_default_te))
        auc_tuned = float(roc_auc_score(y_te, s_tuned_te))
        delta = auc_tuned - auc_default

        print(f'{short[:14]:<14s} {n:>6d} {d:>4d} {lam_star:>7.3f}'
              f' {auc_default:>10.4f} {auc_tuned:>10.4f} {delta:>+8.4f}',
              flush=True)

        rows.append({
            "dataset": short, "n": n, "d": d,
            "lambda_tuned": lam_star,
            "AUC_default_test": auc_default,
            "AUC_tuned_test": auc_tuned,
            "delta_AUC": delta,
            "loss_init": trace[0], "loss_final": trace[-1],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["dataset", "n", "d", "lambda_tuned",
                           "AUC_default_test", "AUC_tuned_test",
                           "delta_AUC", "loss_init", "loss_final"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"# wrote {out}", flush=True)


if __name__ == "__main__":
    main()
