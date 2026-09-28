"""Directional-witness explanations for WAND.

WAND scores a point by how far the extreme of its projection sits beyond
a sub-Gaussian baseline along sampled directions.  Each direction is a
vector in feature space, so the directions that flag a point *are* its
explanation -- at no extra cost over scoring.  This module turns that
observation into feature attributions and provides two attribution modes
over one and the same scorer:

    witness   -- gradient-free.  Read off the probe directions that fire on
                 a point, weighted by how much they fire, projected to
                 ambient feature loadings.  Computed during scoring (free).

    gradient  -- |d score / d x| with the background statistics held fixed.
                 Uses the differentiability of the smooth-extreme score;
                 this is what makes the "fully differentiable" claim do
                 work instead of dangling.

The scorer here is *inductive* with a fixed background (fit once on a
reference sample, then score arbitrary query points).  This is what lets
SHAP / LIME -- which perturb single rows -- explain the same function the
witness/gradient modes explain, so all four are compared on equal footing.

Background statistics use hard median / MAD per direction (WAND's robust
calibration); because they are frozen at fit time, the query-to-score map
x -> s(x) stays differentiable in x.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

Array = Union[np.ndarray, torch.Tensor]


def _as_tensor(X: Array, dtype=torch.float64) -> torch.Tensor:
    if isinstance(X, torch.Tensor):
        return X.to(dtype)
    return torch.as_tensor(np.asarray(X), dtype=dtype)


def _subgauss_baseline(n: int) -> float:
    """c_d(n) = sqrt(2 log n) + log 2 / sqrt(2 log n)."""
    if n < 3:
        return 0.0
    g = math.sqrt(2.0 * math.log(n))
    return g + math.log(2.0) / (g + 1e-12)


def _l1_normalise(A: torch.Tensor) -> torch.Tensor:
    return A / A.abs().sum(dim=1, keepdim=True).clamp_min(1e-30)


class WANDExplainer:
    """Inductive fixed-background WAND scorer + directional-witness
    explanations.

    Parameters
    ----------
    K : probe budget (number of sampling directions).
    T : smooth-extreme temperature for the differentiable score.
    whiten : Mahalanobis-precondition so directions are isotropic.
    seed : RNG seed for the (uniform) direction draw.
    """

    def __init__(
        self,
        K: int = 1024,
        T: float = 0.1,
        whiten: bool = True,
        shrinkage: float = 1e-3,
        seed: int = 0,
        robust: bool = True,
        dtype=torch.float64,
    ):
        self.K = K
        self.T = T
        self.whiten = whiten
        self.shrinkage = shrinkage
        self.seed = seed
        self.robust = robust          # True: median/MAD; False: mean/std (ablation)
        self.dtype = dtype
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, X: Array) -> "WANDExplainer":
        """Cache the background: whitening, directions, per-direction
        median / MAD, sub-Gaussian baseline, and per-direction weights."""
        Xt = _as_tensor(X, self.dtype)
        n, d = Xt.shape
        self.d = d
        self.mean_ = Xt.mean(dim=0)

        # Mahalanobis whitening factor (Linv): Xw = (X - mean) @ Linv.T
        if self.whiten and d >= 2:
            Xc = Xt - self.mean_
            S = (Xc.t() @ Xc) / max(n - 1, 1)
            ridge = self.shrinkage * S.diag().mean()
            S = S + ridge * torch.eye(d, dtype=self.dtype)
            L = torch.linalg.cholesky(S)
            self.Linv_ = torch.linalg.solve_triangular(
                L, torch.eye(d, dtype=self.dtype), upper=False)
        else:
            self.Linv_ = torch.eye(d, dtype=self.dtype)

        # Uniform directions on S^{d-1}
        g = torch.Generator().manual_seed(self.seed)
        U = torch.randn(self.K, d, generator=g, dtype=self.dtype)
        self.U_ = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)   # (K,d)

        # Ambient feature loadings of each direction:  L_amb[k] = Linv^T U_k
        self.L_amb_ = self.U_ @ self.Linv_                            # (K,d)

        # Per-direction centre / scale on the background (frozen). Robust
        # median/MAD by default; mean/std is the non-robust ablation used to
        # show the MAD calibration is what survives heavy-tailed inliers.
        Z = self._project(Xt)                                         # (K,n)
        if self.robust:
            med = Z.median(dim=1).values
            mad = (Z - med.unsqueeze(1)).abs().median(dim=1).values
            self.med_ = med
            self.mad_ = (1.4826 * mad).clamp_min(1e-6)
        else:
            self.med_ = Z.mean(dim=1)
            self.mad_ = Z.std(dim=1).clamp_min(1e-6)
        self.c_ = _subgauss_baseline(n)

        # Per-direction weight = max tail excess on the background (paper w-by-delta)
        res = (Z - self.med_.unsqueeze(1)) / self.mad_.unsqueeze(1)
        tau_bg = (res.abs() - self.c_).clamp_min(0.0)
        delta = tau_bg.max(dim=1).values
        w = delta.clamp_min(0.0)
        if float(w.sum()) < 1e-12:
            w = torch.ones_like(delta)
        self.w_ = w
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def _project(self, Xq: torch.Tensor) -> torch.Tensor:
        """Whitened projections Z[k,i] = (Linv^T U_k) . (x_i - mean)."""
        Xw = (Xq - self.mean_) @ self.Linv_.t()          # (m,d)
        return self.U_ @ Xw.t()                          # (K,m)

    def _tau(self, Xq: torch.Tensor) -> torch.Tensor:
        """Smooth per-direction tail excess for query points. (K,m)"""
        Z = self._project(Xq)
        res = (Z - self.med_.unsqueeze(1)) / self.mad_.unsqueeze(1)
        T = self.T
        # T-smoothed ReLU -> max(., 0) as T -> 0; differentiable in Xq.
        return T * F.softplus((res.abs() - self.c_) / T)

    def score_torch(self, Xq: Array) -> torch.Tensor:
        Xt = _as_tensor(Xq, self.dtype)
        if Xt.ndim == 1:
            Xt = Xt.unsqueeze(0)
        tau = self._tau(Xt)                              # (K,m)
        return (self.w_.unsqueeze(1) * tau).sum(dim=0) / self.w_.sum().clamp_min(1e-30)

    def score(self, Xq: Array) -> np.ndarray:
        with torch.no_grad():
            return self.score_torch(Xq).cpu().numpy()

    # ------------------------------------------------------------------
    def witness_attribution(self, Xq: Array, signed: bool = False) -> np.ndarray:
        """Gradient-free attribution (m,d).

        A point's score is built from the directions that fire on it; each
        direction's whitened projection decomposes over features as
        z_k(x) = sum_j L_amb[k,j] (x_j - mean_j).  The feature-j share of
        the firing evidence is therefore |L_amb[k,j] (x_j - mean_j)|,
        weighted by how much direction k fires (w_k * tau_k).  Summing over
        directions and L1-normalising gives a per-point feature attribution.
        The (x - mean) factor localises to the features this point actually
        deviates on, which is what makes it survive Mahalanobis rotation.

        With ``signed=True`` the magnitudes are unchanged (so every
        detection / ground-truth-recovery result is identical) but each
        entry carries sign(x_j - centre_j): a feature is marked anomalously
        *high* (+) or *low* (-).  The sign is free -- it reuses the
        deviation already formed for the magnitude."""
        Xt = _as_tensor(Xq, self.dtype)
        if Xt.ndim == 1:
            Xt = Xt.unsqueeze(0)
        with torch.no_grad():
            tau = self._tau(Xt)                          # (K,m)
            contrib = self.w_.unsqueeze(1) * tau         # (K,m)
            dev = Xt - self.mean_                         # (m,d) signed deviation
            A = dev.abs() * (contrib.t() @ self.L_amb_.abs())  # (m,d) magnitude
            A = _l1_normalise(A)
            if signed:
                A = A * torch.sign(dev)
        return A.cpu().numpy()

    def dominant_witness(self, Xq: Array) -> np.ndarray:
        """The single most-firing ambient direction per point (m,d), signed."""
        Xt = _as_tensor(Xq, self.dtype)
        if Xt.ndim == 1:
            Xt = Xt.unsqueeze(0)
        with torch.no_grad():
            tau = self._tau(Xt)
            contrib = self.w_.unsqueeze(1) * tau         # (K,m)
            kstar = contrib.argmax(dim=0)                # (m,)
            return self.L_amb_[kstar].cpu().numpy()

    def gradient_attribution(self, Xq: Array, times_input: bool = True) -> np.ndarray:
        """Saliency attribution (m,d) using the differentiability of the
        score.  Returns |(x - mean) . d score/d x| (gradient x input,
        default) or |d score/d x| (times_input=False).  The background
        statistics are frozen, so the gradient is a clean per-point
        derivative."""
        Xt = _as_tensor(Xq, self.dtype)
        if Xt.ndim == 1:
            Xt = Xt.unsqueeze(0)
        Xr = Xt.clone().detach().requires_grad_(True)
        s = self.score_torch(Xr)
        g, = torch.autograd.grad(s.sum(), Xr)            # background frozen => clean diag
        g = g.detach()
        if times_input:
            g = g * (Xt - self.mean_)
        A = _l1_normalise(g.abs())
        return A.cpu().numpy()


def rank_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """Mean per-row Spearman rank correlation between two attribution sets."""
    from scipy.stats import spearmanr
    vals = []
    for a, b in zip(A, B):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        rho, _ = spearmanr(a, b)
        if not np.isnan(rho):
            vals.append(rho)
    return float(np.mean(vals)) if vals else float("nan")


def _selftest() -> None:
    rng = np.random.default_rng(0)
    n, d = 600, 12
    Xin = rng.normal(size=(n, d))
    k = 30
    Xout = rng.normal(size=(k, d))
    true_feats = [2, 5]
    Xout[:, true_feats] += 8.0
    X = np.vstack([Xin, Xout])
    y = np.r_[np.zeros(n), np.ones(k)]

    from sklearn.metrics import roc_auc_score
    expl = WANDExplainer(K=512, seed=0).fit(X)
    s = expl.score(X)
    print(f"  AUC = {roc_auc_score(y, s):.3f}")

    Xa = X[y == 1]
    A_w = expl.witness_attribution(Xa)
    A_g = expl.gradient_attribution(Xa)
    top_w = A_w.mean(0).argsort()[::-1][:2]
    top_g = A_g.mean(0).argsort()[::-1][:2]
    print(f"  witness  top-2 features: {sorted(top_w.tolist())}  (true {true_feats})")
    print(f"  gradient top-2 features: {sorted(top_g.tolist())}  (true {true_feats})")
    print(f"  witness/gradient rank corr: {rank_correlation(A_w, A_g):.3f}")


if __name__ == "__main__":
    print("=== explain.py self-test ===")
    _selftest()
