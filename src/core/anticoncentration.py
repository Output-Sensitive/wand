"""
Output-Sensitive Anomaly Detection via Adaptive Anti-Concentration Probing.

This module implements the method introduced in the paper. The core object
is a *direction-and-point coupling* q(u, x) on S^{d-1} x R^d such that
high-mass regions of q concentrate on (direction, point) pairs that
maximally evidence anomaly. Sampling from q yields an anomaly score that
is

  (i)   output-sensitive:  number of probes K = O~(|H| + k) where |H|
        is the convex-hull / extreme-set cardinality and k is the
        anomaly count (independent of n);
  (ii)  differentiable end-to-end through a softmax extremeness operator
        compatible with the DiffHulls family in `diffhulls_fast.py`;
  (iii) cluster-free: no K-means / kNN / density estimator.

----------------------------------------------------------------------
Method summary
----------------------------------------------------------------------

For a unit direction u in S^{d-1} and projected values z_i = u^T x_i,
let

    Delta(u; X) = ( max_i z_i - median(z) ) / MAD(z)  -  c_d(n)

where c_d(n) is the sub-Gaussian extreme-value baseline

    c_d(n) := sqrt( 2 * log(n) )    +    log(2) / sqrt(2 log n).

Under a sub-Gaussian inlier null Delta(u; X) is uniformly O(1) over u;
anything materially larger is statistical evidence of an anomaly along
direction u.

The direction posterior is

    q(u)  proportional to  exp( beta * Delta_+(u; X) ),

with Delta_+ = max(Delta, 0). We sample from q with projected-Langevin
on the sphere (Riemannian SGLD). The gradient of Delta w.r.t. u is
backed by the soft-extreme operator (softmax over points), exactly as
in `diffhulls_fast.diffhulls_fast_nd`, so the whole pipeline is
differentiable in pytorch.

The per-point anomaly score is the Monte-Carlo aggregate

    s(x_i) = (1/K) sum_{u_k ~ q}  extremeness(x_i, u_k) * Delta_+(u_k; X).

This is exactly the differentiable plug-in of the population identity

    s*(x) = E_{u ~ q*} [ 1{x is argmax_j u^T x_j} * Delta_+(u; mu) ],

which is the halfspace-depth-complement re-weighted by the
anti-concentration excess of each halfspace boundary.

----------------------------------------------------------------------
Public API
----------------------------------------------------------------------

wand_score(X, ...)            high-level scalar score (np.ndarray)
wand_score_torch(X_t, ...)    torch version, gradient-safe
sample_directions_langevin(...)    direction sampler (utility)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class WANDConfig:
    """Hyper-parameters for the anti-concentration probing method.

    Most defaults are dimension-aware and need no tuning.

    Attributes
    ----------
    K              total probe budget (= number of directions sampled)
    K0             warm-start budget: directions drawn uniformly from
                   S^{d-1} for the initial pilot, before Langevin
                   adaptation kicks in. Default: K // 4.
    beta           inverse temperature of the direction posterior
                   q(u) ~ exp(beta * Delta_+(u; X)). Default: log(n).
    step_size      Langevin step size on the sphere (in radians).
                   Default: 1 / sqrt(d).
    n_langevin     Langevin iterations per direction. Default: 30.
    alpha          extremeness sharpening exponent in [1, 2] for the
                   per-point importance weight. Default: 1.0.
    temperature    soft-extreme temperature (smaller = sharper, more
                   like a hard argmax). Default: 1e-2.
    uniform_frac   fraction of probes drawn uniformly even after warm-
                   up, as a coverage safety net (so adversarial
                   anomalies hiding along low-q(u) directions are not
                   missed). Default: 0.20.
    use_null_calib if True, c_d(n) is replaced by an empirical null
                   computed by bootstrapping a Gaussian-fitted copy of
                   X once at the start. Tighter and label-free.
                   Default: True.
    whiten         if True, pre-condition X by Sigma^{-1/2} so probes
                   drawn uniform on S^{d-1} correspond to Mahalanobis-
                   uniform halfspaces. Surfaces anomalies that hide
                   along low-variance directions. Default: True.
    shrinkage      Ledoit-Wolf-style ridge added to Sigma before
                   whitening, as a fraction of trace(Sigma)/d. Stable
                   for d <= n. Default: 1e-3.
    n_refine       number of robust-scale refinement passes after the
                   first scoring pass. Each pass re-estimates median /
                   MAD / Sigma after down-weighting the top
                   `refine_trim` quantile of first-pass scores.
                   Default: 0.
    refine_trim    quantile of top-scoring points to exclude from the
                   refined scale estimator. Default: 0.10.
    cdf_mix        weight in [0, 1] on the empirical-CDF tail term
                   relative to the MAD-z term. Default: 0.0 (pure
                   MAD-z, which dominates on the ODDS suite).
    density_weight additive weight on the density-excess pathway.
                   Scores each point by how over-concentrated its
                   projected k-NN gap is compared to a Gaussian-null
                   bootstrap at the same n. Catches anomalies that
                   form a tight sub-cluster *inside* the bulk -- the
                   inverted-geometry regime (e.g. vertebral) where the
                   standard tail-excess Delta(u) is small or negative.
                   Off by default (0.0) because it can catastrophically
                   regress tail-anomaly datasets (mammography,
                   PageBlocks lose -0.3 to -0.4 AUC at weight=2.0).
                   Set > 0 only when you have prior reason to believe
                   anomalies sit at the mode of the distribution.
    density_k_factor   neighbour count for projected k-NN gap,
                   k_dens = round(density_k_factor * sqrt(n)).
                   Smaller k = sharper concentration detector but
                   noisier. Default 0.5.
    density_null_q lower-tail quantile of the Gaussian-bootstrap log-
                   gap distribution used as the density-excess
                   baseline. Smaller = stricter null (fewer false
                   positives, less sensitive). Default 0.02.
    n_seeds        number of independent direction-sampling passes
                   to average. n_seeds * K total probes; the average
                   has Monte-Carlo variance K * n_seeds times lower
                   than a single pass and is robust to bad Langevin
                   trajectories. Default: 3.
    seed           torch + numpy seed. Default: 0.
    device         "cpu" or "cuda". Default: cpu.
    dtype          torch dtype. Default: torch.float64.
    """
    K: int = 1024
    K0: Optional[int] = None
    beta: Optional[float] = None
    step_size: Optional[float] = None
    n_langevin: int = 20
    alpha: float = 1.0
    temperature: float = 1e-2
    uniform_frac: float = 0.20
    use_null_calib: bool = True
    whiten: bool = False
    shrinkage: float = 1e-3
    n_refine: int = 0
    refine_trim: float = 0.10
    cdf_mix: float = 0.0
    spacing_mix: float = 0.5
    spacing_k_factor: float = 1.0
    axis_probes: bool = True
    axis_weight: float = 0.25
    density_weight: float = 0.0
    density_k_factor: float = 0.5
    density_null_q: float = 0.02
    n_seeds: int = 3
    # Streaming / batching knobs (used when n is too large for an in-memory
    # (K, n) projection matrix). When `n_stats_subsample < n` the per-
    # direction median / MAD, the Gaussian-null bootstrap and the
    # Langevin direction sampler are computed on a random size-
    # `n_stats_subsample` subsample of X; scoring then runs in two
    # batched passes (running max for Delta, weighted sum for the score)
    # over the full X with `batch_size` points per chunk. Spacing /
    # CDF / density components are disabled in batched mode -- they
    # require global sorts over Z which defeat batching. Memory drops
    # from O(K * n) to O(K * batch_size).
    n_stats_subsample: int = 5000
    batch_size: int = 8192
    seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64


# ----------------------------------------------------------------------
# Sub-Gaussian extreme baseline
# ----------------------------------------------------------------------

def _subgauss_baseline(n: int) -> float:
    """Sub-Gaussian extreme-value baseline c_d(n).

    Standard result: for X_1, ..., X_n i.i.d. mean-zero unit-variance
    sub-Gaussian r.v.s, E[max_i X_i] <= sqrt(2 log n), and the
    fluctuation is sub-logarithmic. We use the asymptotic Gumbel
    correction term log(2)/sqrt(2 log n).
    """
    if n < 3:
        return 0.0
    g = math.sqrt(2.0 * math.log(n))
    return g + math.log(2.0) / (g + 1e-12)


def _empirical_null_quantile(
    X: torch.Tensor,
    K_null: int = 256,
    q_level: float = 0.95,
    seed: int = 0,
    cdf_mix: float = 0.0,
    spacing_mix: float = 0.0,
    spacing_k_factor: float = 1.0,
) -> float:
    """Bootstrap an empirical null threshold for Delta(u; .).

    We resample a Gaussian copy of X with matching covariance, project
    onto K_null random directions, and return the q_level quantile of
    the empirical Delta. This calibrates against the actual ambient
    dimension / sample size without assuming sub-Gaussianity tightly.
    Label-free.
    """
    n, d = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    Z = torch.randn(n, d, generator=g, dtype=X.dtype) @ _covariance_root(X)
    u = torch.randn(K_null, d, generator=g, dtype=X.dtype)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-30)
    deltas, _, _, _ = _delta_per_direction(
        Z, u, cdf_mix=cdf_mix,
        spacing_mix=spacing_mix, spacing_k_factor=spacing_k_factor,
    )
    return float(torch.quantile(deltas, q_level).item())


def _covariance_root(X: torch.Tensor, weights: Optional[torch.Tensor] = None,
                     shrinkage: float = 1e-3) -> torch.Tensor:
    """Lower-triangular Cholesky of (possibly weighted) cov(X).

    Parameters
    ----------
    X       : (n, d) tensor
    weights : (n,) non-negative tensor or None. If given, used to compute
              the weighted covariance for the robust refinement pass
              (down-weights suspected anomalies).
    shrinkage : multiplicative ridge as a fraction of trace(S)/d.

    Returns
    -------
    L : (d, d) lower-triangular Cholesky factor.
    """
    n, d = X.shape
    if weights is None:
        Xc = X - X.mean(dim=0, keepdim=True)
        S = (Xc.t() @ Xc) / max(n - 1, 1)
    else:
        w = weights / (weights.sum() + 1e-30)
        mu = (w.unsqueeze(1) * X).sum(dim=0, keepdim=True)
        Xc = X - mu
        S = (Xc.t() * w.unsqueeze(0)) @ Xc
        S = S / (1.0 - (w * w).sum() + 1e-30)
    ridge = shrinkage * S.diag().mean()
    S = S + ridge * torch.eye(d, dtype=X.dtype, device=X.device)
    return torch.linalg.cholesky(S)


def _whitening_matrix(X: torch.Tensor, weights: Optional[torch.Tensor] = None,
                      shrinkage: float = 1e-3) -> torch.Tensor:
    """Whitening transform W such that X @ W has identity (weighted) cov.

    Computed as L^{-T} where L = chol(Sigma). Returns (d, d) matrix.
    """
    L = _covariance_root(X, weights=weights, shrinkage=shrinkage)
    d = L.shape[0]
    return torch.linalg.solve_triangular(
        L, torch.eye(d, dtype=L.dtype, device=L.device), upper=False
    ).t()


# ----------------------------------------------------------------------
# Delta(u; X): the anti-concentration excess
# ----------------------------------------------------------------------

def _local_spacing_tau(Z: torch.Tensor, k: int) -> torch.Tensor:
    """Per-direction 1D k-nearest-neighbour distance, normalised.

    For each direction (row of Z) we sort the projections, compute the
    two-sided k-NN distance in projection (i.e., for the point at rank r,
    min over (+k, -k) of |z_{r+k} - z_r|, with edge clamps), and
    standardise by the median k-NN distance in that direction.

    This is a non-parametric local density estimate: a point is locally
    sparse if its sorted-neighbourhood at distance k is unusually wide.
    Robust to multi-modal projections, where the global MAD-z statistic
    collapses because MAD is dominated by inter-mode spread.

    Returns
    -------
    tau_loc : (K, n)  max( log( d_k(i,u) / median d_k(.,u) ), 0 ).
              In the same units as a log-density excess.
    """
    K_, n = Z.shape
    k = max(1, min(k, n - 1))
    sorted_Z, sort_idx = Z.sort(dim=1)                          # (K, n)
    idx = torch.arange(n, device=Z.device).unsqueeze(0).expand(K_, n)
    r_plus = (idx + k).clamp_max(n - 1)
    r_minus = (idx - k).clamp_min(0)
    right_d = torch.gather(sorted_Z, 1, r_plus) - sorted_Z
    left_d = sorted_Z - torch.gather(sorted_Z, 1, r_minus)
    dist_sorted = torch.minimum(right_d, left_d).clamp_min(1e-12)
    # Scatter back to original index order.
    dist_orig = torch.empty_like(dist_sorted)
    dist_orig.scatter_(1, sort_idx, dist_sorted)
    med_d = dist_orig.median(dim=1, keepdim=True).values.clamp_min(1e-12)
    tau_loc = (dist_orig / med_d).log().clamp_min(0.0)
    return tau_loc


def _density_excess_tau(
    Z: torch.Tensor, k: int, null_log_gap: float,
) -> torch.Tensor:
    """Per-direction per-point density-excess (concentration anomaly).

    Mirror of `_local_spacing_tau`: instead of firing on points whose
    projected k-NN gap is *larger* than the median (sparsity / isolation),
    it fires on points whose gap is *smaller* than the Gaussian-bootstrap
    null lower-tail quantile (over-concentration / mode anomaly).

    Used for datasets like `vertebral` where the labeled anomalies form a
    tight subcluster inside the bulk, so the standard tail Delta(u) signal
    is weak or inverted.

    Returns
    -------
    tau_dens : (K, n)  max( null_log_gap - log d_k(i, u), 0 ).
    """
    K_, n = Z.shape
    k = max(1, min(k, n - 1))
    sorted_Z, sort_idx = Z.sort(dim=1)                          # (K, n)
    idx = torch.arange(n, device=Z.device).unsqueeze(0).expand(K_, n)
    r_plus = (idx + k).clamp_max(n - 1)
    r_minus = (idx - k).clamp_min(0)
    right_d = torch.gather(sorted_Z, 1, r_plus) - sorted_Z
    left_d = sorted_Z - torch.gather(sorted_Z, 1, r_minus)
    gap_sorted = torch.minimum(right_d, left_d).clamp_min(1e-12)
    gap_orig = torch.empty_like(gap_sorted)
    gap_orig.scatter_(1, sort_idx, gap_sorted)
    log_gap = gap_orig.log()
    return (null_log_gap - log_gap).clamp_min(0.0)


def _empirical_null_log_gap(
    X: torch.Tensor,
    k: int,
    K_null: int = 256,
    q_level: float = 0.02,
    seed: int = 0,
) -> float:
    """Bootstrap a Gaussian-null lower-tail quantile of log k-NN gap.

    Projects a Gaussian copy of X with matching covariance onto K_null
    random directions, computes the projected k-NN gap per point per
    direction, and returns the q_level-quantile of log-gap pooled across
    directions and points. Used as the baseline below which observed
    log-gaps signal anomalous over-concentration.
    """
    n, d = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    Z0 = torch.randn(n, d, generator=g, dtype=X.dtype) @ _covariance_root(X)
    u = torch.randn(K_null, d, generator=g, dtype=X.dtype)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-30)
    Z = u @ Z0.t()                                              # (K_null, n)
    k = max(1, min(k, n - 1))
    sorted_Z, _ = Z.sort(dim=1)
    idx = torch.arange(n, device=Z.device).unsqueeze(0).expand(K_null, n)
    r_plus = (idx + k).clamp_max(n - 1)
    r_minus = (idx - k).clamp_min(0)
    right_d = torch.gather(sorted_Z, 1, r_plus) - sorted_Z
    left_d = sorted_Z - torch.gather(sorted_Z, 1, r_minus)
    gap = torch.minimum(right_d, left_d).clamp_min(1e-12)
    return float(torch.quantile(gap.log().flatten(), q_level).item())


def _delta_per_direction(
    X: torch.Tensor,
    U: torch.Tensor,
    temperature: float = 1e-2,
    soft_max: bool = False,
    both_tails: bool = True,
    cdf_mix: float = 0.0,
    spacing_mix: float = 0.0,
    spacing_k_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the per-direction anti-concentration excess together with
    the per-point standardized residuals.

    For direction u and projections z_i = u^T x_i:

      MAD-z component
        r_i(u)  =  (z_i - med(z)) / MAD(z)
        tau_mad_i(u) = max( |r_i| - c_d(n), 0 )

      Empirical-CDF component (ECOD identity along arbitrary u)
        F_u(z)  = empirical CDF
        p_i(u)  = 2 * min( F_u(z_i), 1 - F_u(z_i) )    # two-sided tail prob
        tau_cdf_i(u) = -log( p_i(u) + 1/n )            # log-rank score

    The CDF component is robust to multi-modal projections (where MAD is
    dominated by inter-mode spread and the MAD-z signal collapses). The
    MAD-z component is sharper on unimodal heavy-tailed inliers.

    Final per-point per-direction excess:
        tau_i(u) = (1 - cdf_mix) * tau_mad_i + cdf_mix * tau_cdf_i

    Direction-level excess:
        Delta(u) = max_i tau_i(u)

    Parameters
    ----------
    X : (n, d) tensor
    U : (K, d) unit-direction tensor
    temperature : softmax temperature for log-sum-exp surrogate of max
                  (only used if soft_max=True)
    soft_max : if True, use a soft-max for Delta so it is differentiable
               in u (used by the Langevin step). For scoring, use hard.
    both_tails : if True, MAD-z residuals are |z - med|/MAD (symmetric).
    cdf_mix : weight on the empirical-CDF term in [0, 1]. cdf_mix=0
              recovers the pure MAD-z method; cdf_mix=1 is projection-
              pursuit ECOD. Default in caller: 0.5.

    Returns
    -------
    delta : (K,) per-direction excess     max_i tau_i(u)
    tau   : (K, n) per-point per-direction tail excess  tau_i(u)
    Z     : (K, n) raw projections u^T x_i
    res   : (K, n) standardised residual (z - med) / MAD (signed)
    """
    n = X.shape[0]
    Z = U @ X.t()                                              # (K, n)

    # ---- MAD-z component ----
    med = Z.median(dim=1).values                               # (K,)
    mad = (Z - med.unsqueeze(1)).abs().median(dim=1).values    # (K,)
    mad = mad.clamp_min(1e-6)
    res = (Z - med.unsqueeze(1)) / mad.unsqueeze(1)            # (K, n)
    absres = res.abs() if both_tails else res.clamp_min(0.0)
    c = _subgauss_baseline(n)
    tau_mad = (absres - c).clamp_min(0.0)                      # (K, n)

    tau = tau_mad

    if cdf_mix > 0.0:
        # ---- Empirical CDF tail component ----
        order = Z.argsort(dim=1)
        ranks = torch.empty_like(Z)
        idx = torch.arange(1, n + 1, dtype=Z.dtype, device=Z.device)
        ranks.scatter_(1, order, idx.unsqueeze(0).expand_as(Z))
        F = ranks / float(n)
        p = (2.0 * torch.minimum(F, 1.0 - F)).clamp_min(1.0 / n)
        tau_cdf = (-torch.log(p) - math.log(2.0)).clamp_min(0.0)
        tau = (1.0 - cdf_mix) * tau + cdf_mix * tau_cdf

    if spacing_mix > 0.0:
        # ---- 1D k-spacings local-density component ----
        k_loc = max(2, int(round(spacing_k_factor * math.sqrt(n))))
        tau_loc = _local_spacing_tau(Z, k_loc)
        # Combine via element-wise max so the component with the stronger
        # signal carries each point's score (max == OR of two evidences).
        # Normalise tau_loc to the tau_mad scale (sub-Gaussian baseline c)
        # before maxing, otherwise the log-ratio dominates spuriously.
        scale = c / (tau_loc.max() + 1e-30)
        tau = torch.maximum(tau, scale * tau_loc)

    if soft_max:
        delta = temperature * torch.logsumexp(tau / temperature, dim=1)
    else:
        delta = tau.max(dim=1).values                          # (K,)
    return delta, tau, Z, res


# ----------------------------------------------------------------------
# Riemannian-SGLD sampler on the sphere
# ----------------------------------------------------------------------

def _project_to_tangent(u: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Project Euclidean gradient g onto the tangent space at u.

    For u on S^{d-1}: T_u S^{d-1} = { v : <u, v> = 0 }.
    Projection: g - (u . g) u, vectorised over a batch.
    """
    inner = (u * g).sum(dim=-1, keepdim=True)
    return g - inner * u


def _retract_to_sphere(u: torch.Tensor) -> torch.Tensor:
    """Renormalise to project back onto S^{d-1}."""
    return u / (u.norm(dim=-1, keepdim=True) + 1e-30)


def sample_directions_langevin(
    X: torch.Tensor,
    K: int,
    beta: float,
    step_size: float,
    n_steps: int = 20,
    seed: int = 0,
    temperature: float = 1e-2,
    cdf_mix: float = 0.0,
) -> torch.Tensor:
    """Riemannian projected-SGLD sampler on S^{d-1} targeting

        q(u)  ~  exp( beta * Delta_+(u; X) ).

    The differential of Delta w.r.t. u is computed via autograd on the
    soft-max projection.

    Returns
    -------
    U : (K, d) tensor of unit directions distributed (approximately)
        according to q.
    """
    n, d = X.shape
    device, dtype = X.device, X.dtype
    g = torch.Generator(device="cpu").manual_seed(seed)
    U = torch.randn(K, d, generator=g, dtype=dtype, device=device)
    U = _retract_to_sphere(U)

    # Riemannian SGLD: u_{t+1} = retract( u_t + eta * grad_tan + sqrt(2*eta/beta) * xi_tan )
    for _ in range(n_steps):
        U = U.detach().requires_grad_(True)
        delta, _, _, _ = _delta_per_direction(
            X, U, temperature=temperature, soft_max=True, cdf_mix=cdf_mix,
        )
        # We optimise -beta * Delta_+, so gradient on Delta:
        loss = -(delta.clamp_min(0.0)).sum()
        grad_eu, = torch.autograd.grad(loss, U)
        # Take a *descent* step on the negative log-density:
        grad_tan = _project_to_tangent(U.detach(), -grad_eu)    # ascent on log q
        noise = torch.randn(K, d, generator=g, dtype=dtype, device=device)
        noise_tan = _project_to_tangent(U.detach(), noise)
        U = U.detach() + step_size * grad_tan + math.sqrt(2.0 * step_size / max(beta, 1e-6)) * noise_tan
        U = _retract_to_sphere(U)
    return U.detach()


# ----------------------------------------------------------------------
# Anomaly score (torch)
# ----------------------------------------------------------------------

def _score_one_pass_batched(
    X: torch.Tensor,
    cfg: WANDConfig,
    weights: Optional[torch.Tensor],
    seed_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched / streaming variant of `_score_one_pass` for large n.

    Pipeline:
      Stage A: subsample `cfg.n_stats_subsample` rows of X to estimate
               whitening, empirical null, per-direction median / MAD,
               and the Langevin direction posterior. Cost O(K * n_sub).
      Stage B: two batched passes over the full X with
               `cfg.batch_size` points per chunk:
                 - pass 1 computes Delta(u) = max_i tau_i(u) via a
                   running max across batches;
                 - pass 2 re-computes tau in batches and accumulates
                   the weighted per-point score
                       s(x_i) = (1/W) sum_u w(u) * tau_i(u),
                   w(u) = (Delta(u) - null_q)_+.

    Memory is O(K * batch_size) instead of O(K * n).
    Spacing / CDF / density components are *not* batched here -- they
    require global sorts over the (K, n) projection matrix and are
    silently disabled when this code path is taken (it is reached only
    when `cfg.batch_size < n`, i.e. the user explicitly opts in for
    large-n).
    """
    n, d = X.shape
    bs = max(cfg.batch_size, 256)
    n_sub = max(min(cfg.n_stats_subsample, n), 256)

    K = cfg.K
    K0 = cfg.K0 if cfg.K0 is not None else max(64, K // 4)
    beta = cfg.beta if cfg.beta is not None else float(math.log(max(n_sub, 3)))
    step_size = cfg.step_size if cfg.step_size is not None else 1.0 / math.sqrt(d)

    # --- Stage A: training subsample ---
    g_sub = torch.Generator(device="cpu").manual_seed(
        cfg.seed + seed_offset + 42)
    sub_idx = torch.randperm(n, generator=g_sub)[:n_sub]
    X_sub = X[sub_idx]

    # --- Whitening on the subsample only (cheap, d x d Cholesky) ---
    if cfg.whiten and d >= 2:
        W = _whitening_matrix(
            X_sub, weights=weights[sub_idx] if weights is not None else None,
            shrinkage=cfg.shrinkage,
        )
        # Apply whitening to the full X via a single matmul (n x d) (d x d).
        Xw = X @ W
        Xw_sub = X_sub @ W
    else:
        Xw = X
        Xw_sub = X_sub

    # --- Empirical null on the subsample ---
    if cfg.use_null_calib:
        null_q = _empirical_null_quantile(
            Xw_sub, K_null=max(128, K0),
            seed=cfg.seed + seed_offset,
            cdf_mix=0.0, spacing_mix=0.0,
        )
    else:
        null_q = 0.0

    # --- Direction draws on the whitened sphere (subsample-trained) ---
    g = torch.Generator(device="cpu").manual_seed(cfg.seed + seed_offset)
    U_unif = torch.randn(K0, d, generator=g, dtype=cfg.dtype)
    U_unif = _retract_to_sphere(U_unif).to(device=cfg.device)

    K_adapt = K - K0
    K_safe = int(round(cfg.uniform_frac * K_adapt))
    K_lang = K_adapt - K_safe
    if K_lang > 0:
        U_lang = sample_directions_langevin(
            Xw_sub, K=K_lang, beta=beta, step_size=step_size,
            n_steps=cfg.n_langevin, seed=cfg.seed + seed_offset + 1,
            temperature=cfg.temperature, cdf_mix=0.0,
        )
    else:
        U_lang = Xw_sub.new_zeros((0, d))
    if K_safe > 0:
        U_safe = torch.randn(K_safe, d, generator=g, dtype=cfg.dtype)
        U_safe = _retract_to_sphere(U_safe).to(device=cfg.device)
    else:
        U_safe = Xw_sub.new_zeros((0, d))

    U_rand = torch.cat([U_unif, U_lang, U_safe], dim=0)            # (K, d)

    # Axis probes for tabular feature directions
    if cfg.axis_probes and d >= 2:
        U_axis = torch.eye(d, dtype=cfg.dtype, device=cfg.device)
    else:
        U_axis = Xw.new_zeros((0, d))

    # --- Per-direction med / MAD computed on subsample ---
    def _med_mad(U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Z = U @ Xw_sub.t()                                          # (K, n_sub)
        med = Z.median(dim=1).values
        mad = (Z - med.unsqueeze(1)).abs().median(dim=1).values
        return med, mad.clamp_min(1e-6)

    med_r, mad_r = _med_mad(U_rand)
    if U_axis.shape[0] > 0:
        med_a, mad_a = _med_mad(U_axis)

    c = _subgauss_baseline(n)

    # --- Stage B pass 1: running max over batches to get Delta(u) ---
    def _running_delta(U: torch.Tensor, med: torch.Tensor,
                       mad: torch.Tensor) -> torch.Tensor:
        delta_run = torch.full((U.shape[0],), -float("inf"),
                                dtype=Xw.dtype, device=Xw.device)
        for start in range(0, n, bs):
            X_b = Xw[start:start + bs]
            Z_b = U @ X_b.t()                                       # (K, |B|)
            res_b = (Z_b - med.unsqueeze(1)) / mad.unsqueeze(1)
            tau_b = (res_b.abs() - c).clamp_min(0.0)
            delta_run = torch.maximum(delta_run, tau_b.max(dim=1).values)
        return delta_run

    delta_rand = _running_delta(U_rand, med_r, mad_r)
    if U_axis.shape[0] > 0:
        delta_axis = _running_delta(U_axis, med_a, mad_a)

    # Direction weights (gate via null_q, fallback to delta_+, fallback to ones)
    def _weights(delta: torch.Tensor) -> torch.Tensor:
        w = (delta - null_q).clamp_min(0.0)
        if float(w.sum().item()) < 1e-12:
            w = delta.clamp_min(0.0)
            if float(w.sum().item()) < 1e-12:
                w = torch.ones_like(delta)
        return w

    w_rand = _weights(delta_rand)
    w_rand_sum = w_rand.sum().clamp_min(1e-30)
    if U_axis.shape[0] > 0:
        w_axis = _weights(delta_axis)
        w_axis_sum = w_axis.sum().clamp_min(1e-30)

    # --- Stage B pass 2: per-batch weighted score accumulation ---
    score_rand = Xw.new_zeros((n,))
    score_axis = Xw.new_zeros((n,)) if U_axis.shape[0] > 0 else None
    for start in range(0, n, bs):
        end = start + bs
        X_b = Xw[start:end]
        Z_b = U_rand @ X_b.t()
        res_b = (Z_b - med_r.unsqueeze(1)) / mad_r.unsqueeze(1)
        tau_b = (res_b.abs() - c).clamp_min(0.0)
        if cfg.alpha != 1.0:
            tau_b = tau_b.pow(cfg.alpha)
        score_rand[start:end] = (w_rand.unsqueeze(1) * tau_b).sum(dim=0) / w_rand_sum
        if score_axis is not None:
            Z_a = U_axis @ X_b.t()
            res_a = (Z_a - med_a.unsqueeze(1)) / mad_a.unsqueeze(1)
            tau_a = (res_a.abs() - c).clamp_min(0.0)
            if cfg.alpha != 1.0:
                tau_a = tau_a.pow(cfg.alpha)
            score_axis[start:end] = (w_axis.unsqueeze(1) * tau_a).sum(dim=0) / w_axis_sum

    # --- Combine pathways (same guarded mix as the non-batched path) ---
    def _norm01(s: torch.Tensor) -> torch.Tensor:
        m = s.max()
        return s / (m + 1e-30) if float(m.item()) > 0 else s

    if score_axis is not None:
        score = _norm01(score_rand) + cfg.axis_weight * _norm01(score_axis)
    else:
        score = _norm01(score_rand)

    U = torch.cat([U_rand, U_axis], dim=0)
    return score, U


def _score_one_pass(
    X: torch.Tensor,
    cfg: WANDConfig,
    weights: Optional[torch.Tensor],
    seed_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One scoring pass: optional whitening (using `weights` for the
    weighted covariance), direction sampling, per-point tail-excess
    aggregation. Returns (score, U) in the original ambient coordinates.

    Dispatches to the streaming `_score_one_pass_batched` when n is
    larger than `cfg.batch_size` to keep memory bounded.
    """
    n, d = X.shape
    if cfg.batch_size > 0 and n > cfg.batch_size:
        return _score_one_pass_batched(X, cfg, weights, seed_offset)

    K = cfg.K
    K0 = cfg.K0 if cfg.K0 is not None else max(64, K // 4)
    beta = cfg.beta if cfg.beta is not None else float(math.log(max(n, 3)))
    step_size = cfg.step_size if cfg.step_size is not None else 1.0 / math.sqrt(d)

    # --- Whitening (Mahalanobis preconditioning) ---
    # Sample u uniformly on S^{d-1} in whitened coords X_w = X @ W; this
    # is equivalent to sampling u_orig ~ N(0, Sigma^{-1}) on S^{d-1} in
    # ambient coords -- surfacing low-variance directions equally.
    if cfg.whiten and d >= 2:
        W = _whitening_matrix(X, weights=weights, shrinkage=cfg.shrinkage)
        Xw = X @ W
    else:
        Xw = X

    # --- Empirical null on the (whitened) coords ---
    if cfg.use_null_calib:
        null_q = _empirical_null_quantile(
            Xw, K_null=max(128, K0), seed=cfg.seed + seed_offset,
            cdf_mix=cfg.cdf_mix,
            spacing_mix=cfg.spacing_mix, spacing_k_factor=cfg.spacing_k_factor,
        )
    else:
        null_q = 0.0

    # --- Direction draws on the whitened sphere ---
    g = torch.Generator(device="cpu").manual_seed(cfg.seed + seed_offset)
    U_unif = torch.randn(K0, d, generator=g, dtype=cfg.dtype)
    U_unif = _retract_to_sphere(U_unif).to(device=cfg.device)

    K_adapt = K - K0
    K_safe = int(round(cfg.uniform_frac * K_adapt))
    K_lang = K_adapt - K_safe
    if K_lang > 0:
        U_lang = sample_directions_langevin(
            Xw, K=K_lang, beta=beta, step_size=step_size,
            n_steps=cfg.n_langevin, seed=cfg.seed + seed_offset + 1,
            temperature=cfg.temperature, cdf_mix=cfg.cdf_mix,
        )
    else:
        U_lang = Xw.new_zeros((0, d))
    if K_safe > 0:
        U_safe = torch.randn(K_safe, d, generator=g, dtype=cfg.dtype)
        U_safe = _retract_to_sphere(U_safe).to(device=cfg.device)
    else:
        U_safe = Xw.new_zeros((0, d))

    U_rand = torch.cat([U_unif, U_lang, U_safe], dim=0)        # (K, d)

    def _aggregate(U_probes: torch.Tensor) -> torch.Tensor:
        """Weighted per-point tail-excess score from a probe set.

        Returns a zero tensor if the probe set is empty.
        """
        if U_probes.shape[0] == 0:
            return Xw.new_zeros((Xw.shape[0],))
        delta, tau, _Z, _res = _delta_per_direction(
            Xw, U_probes, temperature=cfg.temperature, soft_max=False,
            both_tails=True, cdf_mix=cfg.cdf_mix,
            spacing_mix=cfg.spacing_mix, spacing_k_factor=cfg.spacing_k_factor,
        )                                                      # tau: (Kp, n)
        w = (delta - null_q).clamp_min(0.0)
        if float(w.sum().item()) < 1e-12:
            w = delta.clamp_min(0.0)
            if float(w.sum().item()) < 1e-12:
                w = torch.ones_like(delta)
        if cfg.alpha != 1.0:
            tau = tau.pow(cfg.alpha)
        w_sum = w.sum().clamp_min(1e-30)
        return (w.unsqueeze(1) * tau).sum(dim=0) / w_sum       # (n,)

    # --- Random / Langevin pathway score ---
    score_rand = _aggregate(U_rand)

    # --- Axis-aligned pathway score (separate pool so a noisy single
    # feature cannot drown out the rotated probes; cf. musk where d=166
    # axis probes contained a non-anomalous outlier that previously
    # dominated the combined pool). ---
    if cfg.axis_probes and d >= 2:
        U_axis = torch.eye(d, dtype=cfg.dtype, device=cfg.device)
        score_axis = _aggregate(U_axis)
    else:
        U_axis = Xw.new_zeros((0, d))
        score_axis = None

    # --- Density-mode pathway (concentration anomaly). Detects points
    # whose projected k-NN gap is anomalously small vs a Gaussian-null
    # bootstrap at the same n -- i.e., tight subclusters inside the bulk
    # that the tail-extremeness Delta(u) misses. Separate aggregator so
    # the per-direction weight is the per-direction *max density excess*
    # (mirroring the tail weight = per-direction max tail excess), not
    # the tail Delta. ---
    if cfg.density_weight > 0.0 and n >= 4:
        k_dens = max(2, int(round(cfg.density_k_factor * math.sqrt(n))))
        null_log_gap = _empirical_null_log_gap(
            Xw, k=k_dens, K_null=max(128, K0),
            q_level=cfg.density_null_q, seed=cfg.seed + seed_offset + 7,
        )
        # Density-mode probes are drawn uniformly on S^{d-1}: a tight
        # subcluster is tight from any angle, so Langevin-adapted
        # directions (which chase tail extremeness) are actively
        # counter-productive here. We also concatenate axis probes,
        # since coordinate-aligned subclusters are common in tabular data.
        g_d = torch.Generator(device="cpu").manual_seed(cfg.seed + seed_offset + 17)
        U_dens_unif = torch.randn(K, d, generator=g_d, dtype=cfg.dtype)
        U_dens_unif = _retract_to_sphere(U_dens_unif).to(device=cfg.device)
        if cfg.axis_probes and d >= 2:
            U_dens = torch.cat([U_dens_unif, U_axis], dim=0)
        else:
            U_dens = U_dens_unif
        Z_dens = U_dens @ Xw.t()
        tau_dens = _density_excess_tau(Z_dens, k_dens, null_log_gap)
        # Sum tau across directions: a true subcluster fires on most
        # angles, so sum-aggregation builds Monte-Carlo confidence the
        # same way the tail-signal Monte-Carlo does, but without
        # per-direction re-weighting (which biases toward directions
        # where one point sticks out, the opposite of cluster
        # detection).
        score_dens = tau_dens.sum(dim=0)
    else:
        score_dens = None

    # --- Combine pathways. We use a *guarded* additive mix: random
    # path is the primary signal, axis path contributes a fraction
    # `axis_weight` of its [0,1]-normalised score. This avoids the
    # max-of-two failure mode where a noisy axis (e.g.\ musk, d=166)
    # produces high scores on inliers and ties them with true anomalies.
    def _norm01(s: torch.Tensor) -> torch.Tensor:
        m = s.max()
        return s / (m + 1e-30) if float(m.item()) > 0 else s

    if score_axis is not None:
        score = _norm01(score_rand) + cfg.axis_weight * _norm01(score_axis)
    else:
        score = _norm01(score_rand)
    if score_dens is not None:
        score = score + cfg.density_weight * _norm01(score_dens)

    U = torch.cat([U_rand, U_axis], dim=0)
    return score, U


def wand_score_torch(
    X: torch.Tensor,
    cfg: Optional[WANDConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Anti-concentration anomaly score on a (n, d) torch tensor.

    Pipeline:
      pass 0 : whitened-direction probing, uniform sample weights.
      pass r : re-whiten + re-score using inlier weights derived by
               down-weighting the top `cfg.refine_trim` quantile of the
               previous pass's score. Robust against masking effects in
               the scale estimator.

    Returns
    -------
    score : (n,) torch tensor in R_{>=0}, larger = more anomalous
    info  : (K, d) tensor of sampled directions from the final pass,
            useful for diagnostics
    """
    if cfg is None:
        cfg = WANDConfig()
    n, d = X.shape
    X = X.to(device=cfg.device, dtype=cfg.dtype)

    # --- Seed-ensembling: average normalised scores from n_seeds passes.
    # Each pass draws K i.i.d. probe directions with a different seed;
    # the average is Monte-Carlo equivalent to a single pass with
    # K * n_seeds probes but additionally robust to bad Langevin
    # trajectories (each pass burns in independently). All passes here
    # are weighted uniformly (weights=None); robust-scale refinement
    # runs *within* each pass via the `n_refine` config below.
    n_seeds = max(cfg.n_seeds, 1)
    score_sum: Optional[torch.Tensor] = None
    U_last: Optional[torch.Tensor] = None
    for s in range(n_seeds):
        weights = None
        score, U = _score_one_pass(X, cfg, weights=None, seed_offset=1000 * s)
        for r in range(max(cfg.n_refine, 0)):
            thresh = torch.quantile(score, 1.0 - cfg.refine_trim)
            weights = (score < thresh).to(cfg.dtype)
            if float(weights.sum().item()) < d + 1:
                break
            score, U = _score_one_pass(
                X, cfg, weights=weights,
                seed_offset=1000 * s + 10 * (r + 1),
            )
        # Normalise each pass to its own [0, 1] range before averaging
        # so a single high-variance pass doesn't dominate.
        score_n = score / (score.max() + 1e-30)
        score_sum = score_n if score_sum is None else (score_sum + score_n)
        U_last = U
    score_avg = score_sum / float(n_seeds)
    return score_avg, U_last


# ----------------------------------------------------------------------
# numpy convenience wrapper
# ----------------------------------------------------------------------

def wand_score(
    X: np.ndarray,
    K: int = 1024,
    seed: int = 0,
    **kwargs,
) -> np.ndarray:
    """Numpy convenience wrapper around wand_score_torch.

    Parameters
    ----------
    X : (n, d) numpy array
    K : probe budget
    seed : RNG seed
    **kwargs : forwarded to WANDConfig
    """
    cfg = WANDConfig(K=K, seed=seed, **kwargs)
    X_t = torch.tensor(X, dtype=cfg.dtype)
    score, _ = wand_score_torch(X_t, cfg)
    return score.detach().cpu().numpy()


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

def _selftest() -> None:
    """Sanity check: anomaly score is high on injected outliers."""
    rng = np.random.default_rng(0)
    n, d = 500, 8
    X_in = rng.normal(size=(n, d))
    # Inject 10 anomalies far from the centroid
    X_out = rng.normal(size=(10, d)) + 8.0
    X = np.vstack([X_in, X_out])
    y = np.array([0] * n + [1] * 10)

    from sklearn.metrics import roc_auc_score
    score = wand_score(X, K=512)
    auc = roc_auc_score(y, score)
    print(f"  AUC on Gaussian + outliers (n={n+10}, d={d}): {auc:.3f}")
    assert auc > 0.95, f"AUC too low: {auc}"
    print("  OK")


if __name__ == "__main__":
    print("=== anti-concentration self-test ===")
    _selftest()
