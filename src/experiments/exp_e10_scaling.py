"""E10 -- Large-n scaling of the WAND scorer.

Rebuts the "no runtime improvement / won't scale" concern with evidence.
WAND is INDUCTIVE: it calibrates once on a reference sample (per-direction
median/MAD + weights), then scores arbitrary query points in a single
linear pass. The probe budget K is independent of n, so the deployed
scorer is O(K n d) -- LINEAR in n, the same complexity class as Isolation
Forest / ECOD / HBOS (and unlike the O(n^2) neighbour methods). The scan
is also embarrassingly parallel over rows (we score in chunks).

We measure, on synthetic Gaussian data (d=20), the wall-clock to score
n = 10^4 .. 10^7 points after a one-time calibration, for two probe
budgets (K=256 "lite", K=1024 default). Output:
  results/e10_scaling.csv
  figures/scaling.pdf  (wall-clock vs n, log-log, with a slope-1 ref)

Usage:  python src/exp_e10_scaling.py
"""
from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch                                          # noqa: E402
from core.explain import WANDExplainer                     # noqa: E402

N_THREADS = 8
D = 20
N_BG = 50_000                       # reference (calibration) sample size
CHUNK = 100_000                     # rows scored per batch (bounds memory)
SIZES = [10_000, 30_000, 100_000, 300_000,
         1_000_000, 3_000_000, 10_000_000]
KS = [256, 1024]


def main():
    torch.set_num_threads(N_THREADS)
    rng = np.random.default_rng(0)
    print(f"threads={N_THREADS}  d={D}  n_bg={N_BG}  chunk={CHUNK}")

    Xbg = rng.standard_normal((N_BG, D))
    # one big query pool (reused across sizes/K); slice prefixes
    print(f"allocating query pool {max(SIZES):,}x{D} ...", flush=True)
    Xpool = rng.standard_normal((max(SIZES), D))

    # Correctness: chunked (streaming) scoring is bit-for-bit identical to a
    # single call -- WAND is inductive (background stats frozen at fit), so
    # each point's score is independent. Batching only bounds memory.
    _ex = WANDExplainer(K=512, seed=0).fit(Xbg)
    _Xv = Xpool[:200_000]
    _full = _ex.score(_Xv)
    _chunked = np.concatenate([_ex.score(_Xv[i:i + CHUNK])
                               for i in range(0, len(_Xv), CHUNK)])
    assert np.allclose(_full, _chunked, atol=1e-12), "batched != single-call!"
    print(f"check: chunked == single-call on 200k pts "
          f"(max|diff|={np.abs(_full - _chunked).max():.1e})", flush=True)

    rows = []
    for K in KS:
        t = time.perf_counter()
        ex = WANDExplainer(K=K, seed=0).fit(Xbg)
        fit_t = time.perf_counter() - t
        print(f"\nK={K}: calibration on {N_BG:,} pts = {fit_t:.2f}s", flush=True)
        for n in SIZES:
            Xq = Xpool[:n]
            t = time.perf_counter()
            out = np.empty(n)
            for i in range(0, n, CHUNK):
                out[i:i + CHUNK] = ex.score(Xq[i:i + CHUNK])
            dt = time.perf_counter() - t
            thr = n / dt
            rows.append(dict(K=K, n=n, fit_s=fit_t, score_s=dt, pts_per_s=thr))
            print(f"  n={n:>10,}  score={dt:8.2f}s  {thr:>10,.0f} pts/s",
                  flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "e10_scaling.csv", index=False)

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    col = {256: "#41b6c4", 1024: "#2c7fb8"}
    for K in KS:
        d = df[df.K == K]
        ax[0].plot(d.n, d.score_s, "o-", color=col[K], label=f"WAND (K={K})")
        ax[1].plot(d.n, d.pts_per_s / 1e3, "o-", color=col[K],
                   label=f"K={K}")
    # slope-1 (linear) reference through the K=1024 first point
    d1 = df[df.K == 1024].sort_values("n")
    x0, y0 = d1.n.iloc[0], d1.score_s.iloc[0]
    xs = np.array([d1.n.min(), d1.n.max()])
    ax[0].plot(xs, y0 * xs / x0, "k--", lw=1.0, alpha=0.7, label="linear (slope 1)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("number of points $n$")
    ax[0].set_ylabel("scoring wall-clock (s)")
    ax[0].set_title("(a) linear scan to $n{=}10^7$")
    ax[0].grid(alpha=0.3, which="both"); ax[0].legend()
    ax[1].set_xscale("log")
    ax[1].set_xlabel("number of points $n$")
    ax[1].set_ylabel("throughput ($10^3$ pts/s)")
    ax[1].set_title("(b) constant throughput")
    ax[1].grid(alpha=0.3, which="both"); ax[1].legend()
    fig.tight_layout()
    out = ROOT / "figures" / "scaling.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nwrote {out}")
    # headline numbers
    big = df[(df.K == 1024) & (df.n == max(SIZES))].iloc[0]
    print(f"HEADLINE: K=1024 scores n={max(SIZES):,} in {big.score_s:.1f}s "
          f"({big.pts_per_s:,.0f} pts/s) after {big.fit_s:.1f}s calibration")


if __name__ == "__main__":
    main()
