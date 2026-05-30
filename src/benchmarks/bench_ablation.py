"""
Ablation bench for WAND on the full ODDS suite.

Each row of Table II of the paper is one WAND configuration; the
PyOD baselines are constant across variants and read from the cached
v5 CSV at results/bench_anomaly_wand.csv. For each variant we
compute WAND's per-dataset AUC and rank (vs. the eight baselines),
then average across 23 datasets.

The variants are *incremental*: each row adds one mechanism over the
previous. The final row is the published configuration.

Output
------
   results/bench_ablation.csv             long-form per-variant stats
   tables/tab_ablation.tex        Table II body (auto-generated)

Run
---
   python -m src.bench_ablation
"""

from __future__ import annotations

import csv
import sys
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core.anticoncentration import WANDConfig, wand_score
from src.benchmarks.bench_anomaly_wand import DATASETS, _load


BASELINE_METHODS = ["IForest", "LOF", "OCSVM", "KNN", "PCA", "HBOS", "ECOD", "COPOD"]


# ----------------------------------------------------------------------
# Variants -- each is an incremental override over the previous
# ----------------------------------------------------------------------

@dataclass
class Variant:
    name:     str                # short name for the row
    label:    str                # LaTeX label for Table II
    cfg:      WANDConfig     # WAND config


def _make_variants() -> list[Variant]:
    """Build the ablation cascade.

    Each variant turns ON one more mechanism than the previous, so the
    diff column-to-column attributes a delta to a single feature.
    """
    base = WANDConfig(
        K=1024, n_seeds=1, n_langevin=20,
        spacing_mix=0.0, axis_probes=False,
        uniform_frac=1.0,           # disables Langevin: 100% uniform draws
        use_null_calib=True,
    )
    variants: list[Variant] = []
    # 1. MAD-z only, uniform sphere probes
    variants.append(Variant("base", "Base (MAD-z, uniform probes)", replace(base)))
    # 2. Add Langevin direction posterior (anti-concentration q(u))
    variants.append(Variant("langevin", "+ Langevin posterior $q(u)$",
                            replace(base, uniform_frac=0.20)))
    # 3. Add 1D spacing component (multi-modal-projection fix)
    variants.append(Variant("spacing", "+ Spacing component \\eqref{eq:tau-spc}",
                            replace(base, uniform_frac=0.20, spacing_mix=0.5)))
    # 4. Add axis-aligned probes (split pathway, w=0.25)
    variants.append(Variant("axis", "+ Axis probes (split pathway)",
                            replace(base, uniform_frac=0.20,
                                    spacing_mix=0.5, axis_probes=True,
                                    axis_weight=0.25)))
    # 5. Add seed ensemble (full method)
    variants.append(Variant("full", "+ Seed ensemble ($S{=}3$, full)",
                            replace(base, uniform_frac=0.20,
                                    spacing_mix=0.5, axis_probes=True,
                                    axis_weight=0.25, n_seeds=3)))
    return variants


# ----------------------------------------------------------------------
# Cached baseline loader
# ----------------------------------------------------------------------

def _load_baseline_cache():
    csv_path = ROOT / "results" / "bench_anomaly_wand.csv"
    cache: dict[str, dict] = {}
    if not csv_path.exists():
        return cache
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            cache[row["dataset"]] = row
    return cache


def _baseline_auc_row(cache_row: dict) -> np.ndarray:
    """Return shape (8,) ROC-AUC values from cache row, NaN if missing."""
    out = np.full(len(BASELINE_METHODS), np.nan)
    for j, m in enumerate(BASELINE_METHODS):
        v = cache_row.get(f"{m}_auc", "")
        if v in ("", "None", None):
            continue
        try:
            out[j] = float(v)
        except (TypeError, ValueError):
            pass
    return out


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def main():
    cache = _load_baseline_cache()
    if not cache:
        print("[!] no cached baselines; run bench_anomaly_wand.py first")
        sys.exit(1)

    variants = _make_variants()

    # Collect WAND AUC per (variant, dataset) and baseline AUC per dataset.
    n_var = len(variants)
    n_data = 0
    dataset_names: list[str] = []
    baseline_aucs: list[np.ndarray] = []
    antic_aucs: dict[int, list[float]] = {i: [] for i in range(n_var)}
    times: dict[int, list[float]] = {i: [] for i in range(n_var)}

    dataset_dir = ROOT / "datasets" / "odds"
    for name in DATASETS:
        path = dataset_dir / f"{name}.npz"
        if not path.exists():
            continue
        cached = cache.get(name.split("_", 1)[1])
        if cached is None:
            continue
        X, y, short = _load(path)
        dataset_names.append(short)
        baseline_aucs.append(_baseline_auc_row(cached))
        for i, v in enumerate(variants):
            t0 = time.perf_counter()
            score = wand_score(
                X, K=v.cfg.K, seed=v.cfg.seed,
                beta=v.cfg.beta, alpha=v.cfg.alpha,
                temperature=v.cfg.temperature,
                uniform_frac=v.cfg.uniform_frac,
                use_null_calib=v.cfg.use_null_calib,
                n_langevin=v.cfg.n_langevin,
                whiten=v.cfg.whiten, shrinkage=v.cfg.shrinkage,
                n_refine=v.cfg.n_refine, refine_trim=v.cfg.refine_trim,
                cdf_mix=v.cfg.cdf_mix,
                spacing_mix=v.cfg.spacing_mix,
                spacing_k_factor=v.cfg.spacing_k_factor,
                axis_probes=v.cfg.axis_probes,
                axis_weight=v.cfg.axis_weight,
                n_seeds=v.cfg.n_seeds,
            )
            auc = float(roc_auc_score(y, score))
            antic_aucs[i].append(auc)
            times[i].append(time.perf_counter() - t0)
        n_data += 1
        print(f"  {short:<14s}  " +
              "  ".join(f"{v.name}={antic_aucs[i][-1]:.3f}" for i, v in enumerate(variants)),
              flush=True)

    # ----- summary: mean AUC + mean rank vs the 8 baselines -----
    baseline_aucs = np.array(baseline_aucs)              # (n_data, 8)
    rows = []
    print(flush=True)
    print(f'{"variant":<35s}  {"mean AUC":>10s}  {"mean rank":>10s}  {"secs/ds":>10s}')
    print("-" * 72)
    for i, v in enumerate(variants):
        ac = np.array(antic_aucs[i])                    # (n_data,)
        # rank of WAND among the 9 methods (1 = best) per dataset
        ranks = []
        for d in range(len(ac)):
            row = np.concatenate([baseline_aucs[d], [ac[d]]])  # (9,)
            valid = ~np.isnan(row)
            order = np.argsort(-row[valid])
            r = np.empty(valid.sum())
            r[order] = np.arange(1, valid.sum() + 1)
            full = np.full(9, np.nan)
            full[valid] = r
            ranks.append(full[-1])  # WAND is the last column
        mean_auc = float(np.nanmean(ac))
        mean_rank = float(np.nanmean(ranks))
        mean_secs = float(np.mean(times[i]))
        rows.append({"name": v.name, "label": v.label,
                     "mean_auc": mean_auc, "mean_rank": mean_rank,
                     "secs": mean_secs})
        print(f"  {v.label:<33s}  {mean_auc:>10.3f}  {mean_rank:>10.2f}  {mean_secs:>10.3f}")

    # ----- write CSV -----
    out_csv = ROOT / "results" / "bench_ablation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "label",
                                          "mean_auc", "mean_rank", "secs"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n# wrote {out_csv}", flush=True)

    # ----- write Table II body (\input target) -----
    aucs = np.array([r["mean_auc"] for r in rows])
    rks  = np.array([r["mean_rank"] for r in rows])
    # bold best, underline second-best, on each column
    auc_order = np.argsort(-aucs); auc_best, auc_second = auc_order[0], auc_order[1]
    rk_order  = np.argsort(rks);   rk_best,  rk_second  = rk_order[0],  rk_order[1]

    def fmt_auc(v, i):
        if i == auc_best:    return f"$\\mathbf{{{v:.3f}}}$"
        if i == auc_second:  return f"$\\underline{{{v:.3f}}}$"
        return f"${v:.3f}$"
    def fmt_rk(v, i):
        if i == rk_best:     return f"$\\mathbf{{{v:.2f}}}$"
        if i == rk_second:   return f"$\\underline{{{v:.2f}}}$"
        return f"${v:.2f}$"

    body_lines = []
    for i, r in enumerate(rows):
        body_lines.append(
            f"{r['label']} & {fmt_auc(r['mean_auc'], i)} & "
            f"{fmt_rk(r['mean_rank'], i)} & ${r['secs']:.2f}$\\,s \\\\"
        )

    tab = (
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Variant & Mean AUC & Mean rank & Wall time \\\\\n"
        "\\midrule\n"
        + "\n".join(body_lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    out_tab = ROOT / "tables" / "tab_ablation.tex"
    out_tab.parent.mkdir(parents=True, exist_ok=True)
    out_tab.write_text(tab)
    print(f"# wrote {out_tab}", flush=True)


if __name__ == "__main__":
    main()
