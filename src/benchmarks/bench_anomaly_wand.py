"""
Benchmark: Anti-Concentration Probing vs. SOTA PyOD / ADBench baselines.

We compare our output-sensitive anti-concentration anomaly detector
(`src.anticoncentration.wand_score`) against eight widely used
unsupervised baselines, on the ODDS / ADBench tabular suite:

   shallow / classical
     - IForest   (Isolation Forest, Liu et al. 2008)
     - LOF       (Local Outlier Factor, Breunig et al. 2000)
     - OCSVM     (One-class SVM, Schoelkopf et al. 2001)
     - KNN       (Ramaswamy et al. 2000)
     - PCA       (reconstruction-error, Shyu et al. 2003)
     - HBOS      (Histogram-based Outlier Score, Goldstein 2012)
   distribution-aware
     - ECOD      (Empirical CDF, Li et al. 2022; published in TKDE)
     - COPOD     (Copula-based, Li et al. 2020; published in ICDM)

All baselines are pulled from PyOD (Zhao et al., JMLR 2019), the
reference implementation that ADBench (Han et al., NeurIPS 2022) uses
to publish its leaderboard. Datasets are the .npz files under
`datasets/odds/`, identical to the ADBench distribution.

Metrics
-------
   ROC-AUC          area under the ROC curve (primary)
   AP               average precision (= area under PR curve)
   secs             wall time for *.fit + decision_function*

Output
------
   results/bench_anomaly_wand.csv     long-form CSV
   stdout                                 pretty table per dataset
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np


class _MethodTimeout(Exception):
    """Raised when a per-method run exceeds --per_method_timeout seconds."""


@contextlib.contextmanager
def _alarm(seconds: int):
    """Per-method SIGALRM-based timeout.

    Only fires when `seconds > 0`. On UNIX/Linux only (signal.SIGALRM is
    not available on Windows; the bench script is Linux-only via the
    PyOD/NumPy dependencies it pulls in).
    """
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise _MethodTimeout(f"timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core.anticoncentration import WANDConfig, wand_score


# ----------------------------------------------------------------------
# ADBench tabular suite (47 of the 48 DIPSMiner datasets; 48_arrhythmia
# unavailable on the ADBench mirror at the time of writing). Grouped by
# scale category as in DIPSMiner (Tables 2-5).
# ----------------------------------------------------------------------

DATASETS = [
    # --- Small-scale (n <= 1k) ---
    "4_breastw",
    "14_glass",
    "15_Hepatitis",
    "18_Ionosphere",
    "21_Lymphography",
    "29_Pima",
    "37_Stamps",
    "39_vertebral",
    "42_WBC",
    "43_WDBC",
    "45_wine",
    "46_WPBC",
    # --- Medium-scale (~1k - 10k) ---
    "2_annthyroid",
    "6_cardio",
    "7_Cardiotocography",
    "12_fault",
    "19_landsat",
    "20_letter",
    "27_PageBlocks",
    "28_pendigits",
    "30_satellite",
    "31_satimage-2",
    "38_thyroid",
    "40_vowels",
    "41_Waveform",
    "44_Wilt",
    "47_yeast",
    # --- Large-scale (>10k) ---
    "1_ALOI",
    "8_celeba",
    "10_cover",
    "11_donors",
    "13_fraud",
    "16_http",
    "22_magic.gamma",
    "23_mammography",
    "32_shuttle",
    "33_skin",
    "34_smtp",
    # --- High-dimensional (large d) ---
    "3_backdoor",
    "5_campaign",
    "9_census",
    "17_InternetAds",
    "24_mnist",
    "25_musk",
    "26_optdigits",
    "35_SpamBase",
    "36_speech",
]


# ----------------------------------------------------------------------
# Baseline registry
# ----------------------------------------------------------------------

def _make_baselines(seed: int = 0):
    """Instantiate the PyOD baseline registry.

    Kept inside a function so import errors are scoped + reported per
    model rather than at module load time. Includes the 9 classical
    PyOD detectors plus 7 newer SOTA shallow detectors (ABOD, COF, SOD,
    INNE, LODA, LSCP, KDE). SUOD is excluded since it requires the
    separate `suod` package.
    """
    from pyod.models.abod import ABOD
    from pyod.models.cof import COF
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest
    from pyod.models.inne import INNE
    from pyod.models.kde import KDE
    from pyod.models.knn import KNN
    from pyod.models.loda import LODA
    from pyod.models.lof import LOF
    from pyod.models.lscp import LSCP
    from pyod.models.ocsvm import OCSVM
    from pyod.models.pca import PCA
    from pyod.models.sod import SOD

    def _lscp():
        # LSCP needs a list of >= 2 heterogeneous base detectors.
        return LSCP(
            detector_list=[LOF(n_neighbors=15), LOF(n_neighbors=30),
                           LOF(n_neighbors=45), HBOS(), HBOS(n_bins=20)],
            random_state=seed,
        )

    return {
        "IForest": lambda: IForest(random_state=seed),
        "LOF":     lambda: LOF(),
        "OCSVM":   lambda: OCSVM(),
        "KNN":     lambda: KNN(),
        "PCA":     lambda: PCA(random_state=seed),
        "HBOS":    lambda: HBOS(),
        "ECOD":    lambda: ECOD(),
        "COPOD":   lambda: COPOD(),
        "ABOD":    lambda: ABOD(),
        "COF":     lambda: COF(),
        "SOD":     lambda: SOD(),
        "INNE":    lambda: INNE(random_state=seed),
        "LODA":    lambda: LODA(),
        "LSCP":    _lscp,
        "KDE":     lambda: KDE(),
        "PIDForest": lambda: _make_pidforest(seed),
    }


def _make_pidforest(seed):
    from src.core.pidforest_wrapper import PIDForest
    return PIDForest(random_state=seed)


# ----------------------------------------------------------------------
# Data loader (ADBench .npz format)
# ----------------------------------------------------------------------

def _load(path: Path):
    """Load an ADBench .npz file, z-score features, return (X, y, name).

    Constant (zero-variance) columns are dropped before standardisation;
    otherwise PyOD-PCA's internal SVD overflows when a column is divided
    by its near-zero std.
    """
    d = np.load(path)
    X = d["X"].astype(np.float64)
    y = d["y"].astype(int)
    keep = X.std(axis=0) > 1e-10
    if not keep.all():
        X = X[:, keep]
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    X = (X - mu) / sd
    name = path.stem.split("_", 1)[1]
    return X, y, name


# ----------------------------------------------------------------------
# Single-dataset evaluation
# ----------------------------------------------------------------------

def _score_baseline(name: str, model_factory, X: np.ndarray, timeout: int = 0):
    """Fit a PyOD baseline and return (score, seconds).

    Raises `_MethodTimeout` if `timeout > 0` and the fit + scoring exceeds
    that wall-clock budget. Caller is expected to catch and mark the cell.
    """
    t0 = time.perf_counter()
    with _alarm(timeout):
        m = model_factory()
        m.fit(X)
        s = m.decision_function(X)
    return s, time.perf_counter() - t0


def _score_wand(X: np.ndarray, cfg: WANDConfig, timeout: int = 0):
    """Run our method, return (score, seconds).

    Forwards the *entire* WANDConfig (modulo K, seed which the
    wrapper consumes positionally) so newer fields -- spacing_mix,
    spacing_k_factor, axis_probes, n_seeds, cdf_mix, whiten, n_refine --
    don't get silently dropped. Wrapped in the same SIGALRM timeout as
    the PyOD baselines.
    """
    t0 = time.perf_counter()
    with _alarm(timeout):
        s = wand_score(
            X, K=cfg.K, seed=cfg.seed,
            beta=cfg.beta, alpha=cfg.alpha,
            temperature=cfg.temperature,
            uniform_frac=cfg.uniform_frac,
            use_null_calib=cfg.use_null_calib,
            n_langevin=cfg.n_langevin,
            whiten=cfg.whiten, shrinkage=cfg.shrinkage,
            n_refine=cfg.n_refine, refine_trim=cfg.refine_trim,
            cdf_mix=cfg.cdf_mix,
            spacing_mix=cfg.spacing_mix, spacing_k_factor=cfg.spacing_k_factor,
            axis_probes=cfg.axis_probes, axis_weight=cfg.axis_weight,
            density_weight=cfg.density_weight,
            density_k_factor=cfg.density_k_factor,
            density_null_q=cfg.density_null_q,
            n_seeds=cfg.n_seeds,
            n_stats_subsample=cfg.n_stats_subsample,
            batch_size=cfg.batch_size,
        )
    return s, time.perf_counter() - t0


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=None,
                        help="comma-separated dataset short codes (e.g. '14_glass,42_WBC')")
    parser.add_argument("--K", type=int, default=2048,
                        help="probe budget for anti-concentration (default 2048)")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="comma-separated seeds (default '0,1,2'); per-cell"
                             " results are averaged across seeds.")
    parser.add_argument("--out", type=str,
                        default=str(ROOT / "results" / "bench_anomaly_wand.csv"))
    parser.add_argument("--out_raw", type=str,
                        default=str(ROOT / "results"
                                    / "bench_anomaly_wand_raw.csv"),
                        help="raw per-seed CSV (long form)")
    parser.add_argument("--max_n_quadratic", type=int, default=30000,
                        help="skip O(n^2)-memory models on datasets above this size")
    parser.add_argument("--per_method_timeout", type=int, default=300,
                        help="per-method SIGALRM timeout in seconds (default 300 = 5 min);"
                             " set 0 to disable")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    quadratic_models = {"LOF", "KNN", "COF", "SOD", "ABOD", "LSCP", "KDE"}

    # Reference method list (uses seeds[0] only; rebuilt per-seed inside the loop)
    method_names = list(_make_baselines(seed=seeds[0]).keys()) + ["WAND"]

    if args.datasets is not None:
        dataset_list = args.datasets.split(",")
    else:
        dataset_list = DATASETS

    header = f'{"dataset":<14s} {"n":>7s} {"d":>4s} {"%anom":>6s}'
    for m in method_names:
        header += f' {m:>9s}'
    print(header, flush=True)
    print("-" * len(header), flush=True)

    rows = []
    raw_rows = []          # long form, one row per (dataset, model, seed)
    dataset_dir = ROOT / "datasets" / "odds"
    for name in dataset_list:
        path = dataset_dir / f"{name}.npz"
        if not path.exists():
            print(f"[skip] missing {path}", flush=True)
            continue
        X, y, short = _load(path)
        n, d = X.shape
        contam = float(y.mean())
        line_auc = f'{short[:14]:<14s} {n:>7d} {d:>4d} {contam*100:>5.1f}%'
        row = {"dataset": short, "n": n, "d": d, "contam": contam}

        # Skip a model entirely on this dataset if the model has known
        # quadratic memory and n exceeds the cap, *before* we waste any
        # time trying it.
        for m in method_names:
            if m in quadratic_models and n > args.max_n_quadratic:
                line_auc += f' {"skip":>9s}'
                row[f"{m}_auc"] = None
                row[f"{m}_ap"]  = None
                row[f"{m}_secs"] = None
                continue

            per_seed_auc, per_seed_ap, per_seed_secs = [], [], []
            for s in seeds:
                try:
                    if m == "WAND":
                        cfg = WANDConfig(K=args.K, seed=s)
                        score, secs = _score_wand(
                            X, cfg, timeout=args.per_method_timeout)
                    else:
                        baselines_s = _make_baselines(seed=s)
                        score, secs = _score_baseline(
                            m, baselines_s[m], X,
                            timeout=args.per_method_timeout)
                    auc = float(roc_auc_score(y, score))
                    ap  = float(average_precision_score(y, score))
                    per_seed_auc.append(auc)
                    per_seed_ap.append(ap)
                    per_seed_secs.append(secs)
                    raw_rows.append({
                        "dataset": short, "n": n, "d": d, "contam": contam,
                        "method": m, "seed": s,
                        "auc": auc, "ap": ap, "secs": secs,
                    })
                except Exception as e:
                    raw_rows.append({
                        "dataset": short, "n": n, "d": d, "contam": contam,
                        "method": m, "seed": s,
                        "auc": None, "ap": None, "secs": None,
                    })
                    print(f"  [warn] {m} on {short} seed {s}: {e}", flush=True)
            if per_seed_auc:
                auc_mean = float(np.mean(per_seed_auc))
                ap_mean  = float(np.mean(per_seed_ap))
                secs_mean = float(np.mean(per_seed_secs))
                line_auc += f' {auc_mean:>9.3f}'
                row[f"{m}_auc"]  = auc_mean
                row[f"{m}_ap"]   = ap_mean
                row[f"{m}_secs"] = secs_mean
            else:
                line_auc += f' {"ERR":>9s}'
                row[f"{m}_auc"] = None
                row[f"{m}_ap"]  = None
                row[f"{m}_secs"] = None
        print(line_auc, flush=True)
        rows.append(row)

        # incremental save -- the run is long; we want partial outputs.
        _write_csv_mean(args.out, method_names, rows)
        _write_csv_raw(args.out_raw, raw_rows)

    # ---------- average summary ----------
    if rows:
        print("\n" + "=" * len(header), flush=True)
        avg_line = f'{"AVG (ROC-AUC)":<14s} {"":>7s} {"":>4s} {"":>6s}'
        for m in method_names:
            vals = [r[f"{m}_auc"] for r in rows if r.get(f"{m}_auc") is not None]
            avg_line += f' {np.mean(vals):>9.3f}' if vals else f' {"--":>9s}'
        print(avg_line, flush=True)

        rank_line = f'{"AVG rank":<14s} {"":>7s} {"":>4s} {"":>6s}'
        # Compute mean rank per method, 1 = best.
        auc_mat = np.array([[r[f"{m}_auc"] if r.get(f"{m}_auc") is not None else np.nan
                             for m in method_names] for r in rows])
        # Rank within each row (higher AUC -> lower rank). NaN -> middle rank.
        ranks = np.zeros_like(auc_mat)
        for i in range(auc_mat.shape[0]):
            row_v = auc_mat[i]
            valid = ~np.isnan(row_v)
            order = np.argsort(-row_v[valid])    # descending AUC
            r_v = np.empty(valid.sum())
            r_v[order] = np.arange(1, valid.sum() + 1)
            full = np.full_like(row_v, np.nan, dtype=float)
            full[valid] = r_v
            ranks[i] = full
        for j, m in enumerate(method_names):
            v = ranks[:, j]
            v = v[~np.isnan(v)]
            rank_line += f' {np.mean(v):>9.2f}' if v.size else f' {"--":>9s}'
        print(rank_line, flush=True)

    # ---------- final CSV write (incremental saves happened in the loop) ----------
    _write_csv_mean(args.out, method_names, rows)
    _write_csv_raw(args.out_raw, raw_rows)
    print(f"\n# wrote {args.out}", flush=True)
    print(f"# wrote {args.out_raw}", flush=True)


def _write_csv_mean(out_path: str, method_names, rows):
    """Wide-form CSV: one row per dataset, mean over seeds per (method, metric)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "n", "d", "contam"]
    for m in method_names:
        fieldnames += [f"{m}_auc", f"{m}_ap", f"{m}_secs"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_csv_raw(out_path: str, raw_rows):
    """Long-form CSV: one row per (dataset, method, seed)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "n", "d", "contam", "method", "seed",
                  "auc", "ap", "secs"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(raw_rows)


if __name__ == "__main__":
    main()
