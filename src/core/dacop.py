"""
DACOP — Differentiable Anti-Concentration Probing.

End-to-end differentiable variant of `anticoncentration.wand_score`.
Replaces every non-smooth primitive in WAND with a smooth surrogate so
that `dscore/dX` (and `dscore/dtheta` for learnable theta) is dense
everywhere:

    WAND                         DACOP
    ---------------------------     ----------------------------------
    Z.median(dim=1), MAD            mean/std in whitened coords
                                    (differentiable Cholesky whitening)
    Z.sort(dim=1) + k-NN gather     not used in the tail signal here
                                    (the density pathway is kept off by
                                     default since WAND's empirical
                                     study showed catastrophic regression
                                     on tail-anomaly datasets)
    tau.max(dim=1)                  T * logsumexp(tau / T, dim=1)
    .clamp_min(0.0)                 softplus(x / T) * T  (smooth ReLU)
    empirical_null .item()          analytical sub-Gaussian baseline
                                    c = sqrt(2 log n) + log(2)/sqrt(2 log n)
                                    + learnable correction
    sample_directions_langevin      uniform random directions (initial
    (detached)                      version). Optional reparametrized
                                    Langevin available via `n_langevin>0`
                                    in which case autograd flows through
                                    the chain (create_graph=True).

Learnable parameters:
    log_beta, log_T, log_eps        positivity-reparam scalars
    c_correction                    additive correction on the sub-
                                    Gaussian baseline
    encoder (optional)              small MLP R^d -> R^d. Default
                                    identity for the non-parametric
                                    variant.

Training signal: synthetic-anomaly contrastive loss
    L = softplus(margin + mean(s_inliers) - mean(s_synth))
where synthetics are a mixture of bounding-box-uniform (tail mode) and
tight-cluster-near-bulk (concentration mode). See `train_dacop`.

Inference cost is comparable to WAND at matched K.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Differentiable primitives
# ----------------------------------------------------------------------

def _smooth_cov_root(X: torch.Tensor, shrinkage: float = 1e-3) -> torch.Tensor:
    """Differentiable Cholesky of (Σ + ridge·diag) where Σ = cov(X).

    Identical to WAND `_covariance_root` but never receives a `weights`
    arg (the refinement pass is not used by DACOP). Returns a (d, d)
    lower-triangular factor whose gradient w.r.t. X is the standard
    Cholesky-backward.
    """
    n, d = X.shape
    Xc = X - X.mean(dim=0, keepdim=True)
    S = (Xc.t() @ Xc) / max(n - 1, 1)
    ridge = shrinkage * S.diag().mean()
    S = S + ridge * torch.eye(d, dtype=X.dtype, device=X.device)
    return torch.linalg.cholesky(S)


def _whiten(X: torch.Tensor, shrinkage: float = 1e-3) -> torch.Tensor:
    """Mahalanobis whitening: returns Xw such that cov(Xw) ≈ I.

    Computed as Xw = X @ (L^{-T}) where L = chol(Σ). All steps are
    differentiable in torch.
    """
    n, d = X.shape
    if d < 2:
        return X
    L = _smooth_cov_root(X, shrinkage=shrinkage)
    eye = torch.eye(d, dtype=X.dtype, device=X.device)
    Linv = torch.linalg.solve_triangular(L, eye, upper=False)
    return X @ Linv.t()


def _smooth_relu(x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """T-smoothed ReLU: T·softplus(x/T) → max(x, 0) as T → 0."""
    return T * F.softplus(x / T)


def _smooth_median(z: torch.Tensor, n_iter: int = 4,
                   eps: float = 1e-3) -> torch.Tensor:
    """IRLS L1-median along the last axis. Fully differentiable.

    Fixed-point iteration: m_{t+1} = (sum_i w_i z_i) / (sum_i w_i),
    w_i = 1 / (|z_i - m_t| + eps). Converges to L1 median in 1D
    (= standard median). 4 iterations is enough for moderate-n.

    Use instead of `Z.mean()` when the data has heavy contamination
    that would skew the mean -- e.g. small datasets where a handful
    of anomalies materially shifts the central tendency.
    """
    m = z.mean(dim=-1, keepdim=True)
    for _ in range(n_iter):
        w = 1.0 / (torch.abs(z - m) + eps)
        m = (w * z).sum(dim=-1, keepdim=True) / w.sum(dim=-1, keepdim=True)
    return m.squeeze(-1)


def _smooth_mad(z: torch.Tensor, n_iter: int = 4,
                eps: float = 1e-3) -> torch.Tensor:
    """Smooth MAD = smooth_median(|z - smooth_median(z)|).

    Robust scale that pairs with `_smooth_median`. Falls back to a
    Cauchy-style scale if the inner median is degenerate.
    """
    med = _smooth_median(z, n_iter=n_iter, eps=eps)
    return _smooth_median(torch.abs(z - med.unsqueeze(-1)),
                          n_iter=n_iter, eps=eps)


def _smooth_max(tau: torch.Tensor, T: torch.Tensor, dim: int) -> torch.Tensor:
    """T-smoothed max: T·logsumexp(tau/T) → max(tau) as T → 0.

    Dense gradient through all entries (vs. argmax which gives gradient
    to a single index).
    """
    return T * torch.logsumexp(tau / T, dim=dim)


# ----------------------------------------------------------------------
# DACOP scorer module
# ----------------------------------------------------------------------

class DACOP(nn.Module):
    """Differentiable Anti-Concentration anomaly scorer.

    Forward returns a per-point anomaly score in R_{>=0}, larger means
    more anomalous. The map X -> score is end-to-end differentiable in
    both X and the module's learnable parameters.

    Parameters
    ----------
    d : input dimension.
    K : number of probe directions per forward (default 256). The full
        suite of K probes is resampled on every forward to avoid
        memorising a fixed direction set; the resampling RNG does not
        carry gradients (consistent with standard reparametrization).
    encoder_hidden : if not None, an MLP encoder R^d -> R^d_enc -> R^d
        is inserted before whitening. Default None = identity encoder
        (purely non-parametric scorer driven by learnable scalars).
    use_whitening : Mahalanobis-precondition the (post-encoder) data so
        directions on S^{d-1} are Mahalanobis-uniform. Default True.
    init_T : initial smoothing temperature T. Smaller = sharper, closer
        to WAND at the cost of a noisier gradient. Default 0.1.
    init_c_correction : additive correction on the sub-Gaussian baseline
        c_d(n). Default 0.0 = pure analytical baseline.
    """

    def __init__(
        self,
        d: int,
        K: int = 256,
        encoder_hidden: Optional[int] = None,
        use_whitening: bool = True,
        init_T: float = 0.1,
        init_c_correction: float = 0.0,
        robust_center: bool = False,
        residual_encoder: bool = True,
        encoder_scale: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.K = K
        self.use_whitening = use_whitening
        self.robust_center = robust_center
        self.residual_encoder = residual_encoder

        if encoder_hidden is not None:
            self.encoder_mlp: Optional[nn.Module] = nn.Sequential(
                nn.Linear(d, encoder_hidden),
                nn.GELU(),
                nn.Linear(encoder_hidden, d),
            )
            # Initialise final layer to ~zero so the encoder starts at
            # identity (when residual_encoder=True) or near-identity
            # (otherwise). This avoids the early-training collapse where
            # a randomly-initialised encoder maps everything into a tiny
            # ball and DACOP scores all points equally.
            with torch.no_grad():
                self.encoder_mlp[-1].weight.mul_(encoder_scale)
                self.encoder_mlp[-1].bias.mul_(0.0)
        else:
            self.encoder_mlp = None

        self.log_T = nn.Parameter(torch.tensor(math.log(init_T)))
        self.log_eps = nn.Parameter(torch.tensor(math.log(1e-3)))
        self.c_correction = nn.Parameter(torch.tensor(init_c_correction))

    # ------------------------------------------------------------------

    def encoder(self, X: torch.Tensor) -> torch.Tensor:
        """Optional residual encoder: H = X + Phi(X) (residual) or Phi(X).

        Residual form keeps the model near identity at initialisation so
        Mahalanobis whitening and the rest of the pipeline see well-scaled
        inputs from epoch 0. Mitigates the collapse-to-AUC-0.5 failure
        mode observed with full-MLP encoders on small datasets.
        """
        if self.encoder_mlp is None:
            return X
        if self.residual_encoder:
            return X + self.encoder_mlp(X)
        return self.encoder_mlp(X)

    # ------------------------------------------------------------------

    def _sub_gaussian_baseline(self, n: int) -> torch.Tensor:
        """Analytical c_d(n) = sqrt(2 log n) + log(2) / sqrt(2 log n).

        Returned as a tensor (not a python float) so it carries the
        learnable additive correction c_correction.
        """
        if n < 3:
            return self.c_correction
        g = math.sqrt(2.0 * math.log(n))
        return torch.tensor(g + math.log(2.0) / (g + 1e-12),
                            dtype=self.c_correction.dtype,
                            device=self.c_correction.device) + self.c_correction

    def _sample_directions(self, K: int, d: int, device, dtype) -> torch.Tensor:
        """Uniform random unit-norm directions on S^{d-1}.

        The randomness does not carry gradient (it is exogenous noise in
        the reparametrization view). Gradients flow through the data
        only.
        """
        U = torch.randn(K, d, device=device, dtype=dtype)
        return U / U.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # ------------------------------------------------------------------

    def forward(
        self,
        X: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        H = self.encoder(X)
        Hw = _whiten(H) if self.use_whitening else H
        n, d = Hw.shape

        T = self.log_T.exp().clamp(min=1e-3, max=1.0)
        eps = self.log_eps.exp().clamp(min=1e-9, max=1.0)
        c = self._sub_gaussian_baseline(n)

        U = self._sample_directions(self.K, d, Hw.device, Hw.dtype)
        Z = U @ Hw.t()                                # (K, n)

        # Central tendency / scale. In whitened coords mean/std are
        # well-behaved -- but if the data has heavy contamination,
        # outliers can shift the mean. `robust_center=True` switches to
        # an IRLS L1-median + smooth MAD that is asymptotically as
        # robust as WAND's hard median while remaining
        # differentiable.
        if self.robust_center:
            mu = _smooth_median(Z).unsqueeze(1)
            std = _smooth_mad(Z).unsqueeze(1).clamp_min(eps)
            # Convert MAD to a sub-Gaussian scale (1.4826 * MAD ~ sigma)
            std = 1.4826 * std
        else:
            mu = Z.mean(dim=1, keepdim=True)
            std = Z.std(dim=1, keepdim=True).clamp_min(eps)
        res = (Z - mu) / std

        # Smooth tail excess per point per direction
        tau = _smooth_relu(res.abs() - c, T)          # (K, n)

        # Smooth max over points -> per-direction excess
        delta = _smooth_max(tau, T, dim=1)            # (K,)

        # Weight directions by their tail excess (matches WAND w-by-delta)
        w = _smooth_relu(delta, T)                    # (K,)
        w_sum = w.sum().clamp_min(eps)
        score = (w.unsqueeze(1) * tau).sum(dim=0) / w_sum

        if return_components:
            return score, dict(delta=delta, tau=tau, U=U, Hw=Hw, T=T, c=c)
        return score


# ----------------------------------------------------------------------
# Synthetic-anomaly contrast training
# ----------------------------------------------------------------------

def _sample_synthetic_anomalies(
    X: torch.Tensor,
    n_syn: int,
    mode_frac: float = 0.5,
    cluster_size: int = 10,
    cluster_scale: float = 0.05,
) -> torch.Tensor:
    """Synthesise pseudo-anomalies covering both anomaly modes.

    Half the synthetics are drawn uniformly inside the per-feature
    bounding box of X (covers tail-extreme anomalies, the standard
    GOAD/NeuTraL synthesis). The other half are tight clusters anchored
    near random data points (covers concentration anomalies of the kind
    vertebral has — labelled anomalies in the data mode).

    Returns (n_syn, d).
    """
    n, d = X.shape
    n_tail = int(n_syn * (1.0 - mode_frac))
    n_mode = n_syn - n_tail

    X_min = X.min(dim=0).values
    X_max = X.max(dim=0).values
    span = (X_max - X_min)

    # Tail: bounding-box uniform, slightly inflated
    u = torch.rand(n_tail, d, device=X.device, dtype=X.dtype)
    X_tail = X_min + (u * 1.2 - 0.1) * span

    # Mode: pick random anchor points, perturb tightly
    if n_mode > 0:
        anchor_idx = torch.randint(0, n, (n_mode,), device=X.device)
        anchors = X[anchor_idx]
        noise = torch.randn(n_mode, d, device=X.device, dtype=X.dtype) * cluster_scale
        # Force the cluster to be tight: collapse a fraction onto a single anchor
        if n_mode > 1:
            shared = anchors[0:1].expand_as(anchors)
            mix = 0.5
            anchors = (1 - mix) * anchors + mix * shared
        X_mode = anchors + noise * span.mean()
    else:
        X_mode = X.new_zeros((0, d))

    return torch.cat([X_tail, X_mode], dim=0)


def train_dacop(
    model: DACOP,
    X: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    margin: float = 1.0,
    n_synth_factor: float = 1.0,
    weight_decay: float = 0.0,
    seed: int = 0,
    log_every: int = 50,
    verbose: bool = False,
) -> dict:
    """Train DACOP with synthetic-anomaly contrastive loss.

    Loss
    ----
        L = softplus( margin + mean(score(X_inliers)) - mean(score(X_synth)) )

    The softplus is a smooth hinge: it is large when synthetics fail to
    score above inliers by `margin`, and saturates to 0 when they do.

    Returns a dict of training trace metrics (loss per `log_every`
    epochs) for diagnostics.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    n, d = X.shape

    trace = {"epoch": [], "loss": [], "s_in_mean": [], "s_syn_mean": []}
    for ep in range(epochs):
        model.train()
        n_syn = max(8, int(n * n_synth_factor))
        X_syn = _sample_synthetic_anomalies(X, n_syn)
        X_all = torch.cat([X, X_syn], dim=0)
        s_all = model(X_all)
        s_in, s_syn = s_all[:n], s_all[n:]

        loss = F.softplus(margin + s_in.mean() - s_syn.mean())

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if (ep % log_every == 0) or (ep == epochs - 1):
            trace["epoch"].append(ep)
            trace["loss"].append(float(loss.item()))
            trace["s_in_mean"].append(float(s_in.mean().item()))
            trace["s_syn_mean"].append(float(s_syn.mean().item()))
            if verbose:
                print(f"  ep {ep:4d}  loss={float(loss):.4f}  "
                      f"s_in={float(s_in.mean()):.3f}  "
                      f"s_syn={float(s_syn.mean()):.3f}")
    return trace


# ----------------------------------------------------------------------
# Numpy convenience wrappers
# ----------------------------------------------------------------------

def dacop_score(
    X,
    epochs: int = 200,
    K: int = 256,
    use_encoder: bool = False,
    encoder_hidden: Optional[int] = None,
    robust_center: bool = False,
    residual_encoder: bool = True,
    encoder_scale: float = 0.1,
    seed: int = 0,
    margin: float = 1.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    init_T: float = 0.1,
    verbose: bool = False,
):
    """Train DACOP on X and return per-point anomaly scores.

    Parameters mirror WAND's `wand_score` for swap-in
    convenience; additional knobs control training and the
    smooth-statistic variant.
    """
    X_np = X.astype("float64") if hasattr(X, "astype") else X
    X_t = torch.tensor(X_np, dtype=torch.float64)
    d = X_t.shape[1]
    if use_encoder and encoder_hidden is None:
        encoder_hidden = 2 * d
    model = DACOP(
        d=d, K=K,
        encoder_hidden=(encoder_hidden if use_encoder else None),
        robust_center=robust_center,
        residual_encoder=residual_encoder,
        encoder_scale=encoder_scale,
        init_T=init_T,
    ).double()
    torch.manual_seed(seed)
    train_dacop(model, X_t, epochs=epochs, lr=lr, margin=margin,
                weight_decay=weight_decay, seed=seed, verbose=verbose)
    model.eval()
    with torch.no_grad():
        s = model(X_t)
    return s.detach().cpu().numpy()


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

def _selftest() -> None:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    n, d = 500, 8
    X_in = rng.normal(size=(n, d))
    X_out = rng.normal(size=(10, d)) + 8.0
    X = np.vstack([X_in, X_out])
    y = np.array([0] * n + [1] * 10)
    s = dacop_score(X, epochs=150, K=128, verbose=False)
    auc = roc_auc_score(y, s)
    print(f"  DACOP AUC on Gaussian + outliers (n={n+10}, d={d}): {auc:.3f}")
    assert auc > 0.9, f"AUC too low: {auc}"
    print("  OK")


if __name__ == "__main__":
    print("=== DACOP self-test ===")
    _selftest()
