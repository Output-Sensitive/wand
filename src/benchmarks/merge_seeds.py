"""
Average AUC / AP / secs across multiple seed sidecars and overwrite the
4 stochastic-method columns in bench_anomaly_wand.csv.

Reads sidecars produced by `run_one_seed.py` (one per seed). For each
(dataset, method) cell:
  - collects the AUC/AP/secs values from every sidecar that has them,
  - averages the values whose AUC is a finite number (sentinel "nan"
    values are dropped before averaging),
  - writes the averaged AUC/AP/secs back into the main CSV.

If *every* sidecar has a sentinel for a cell, the main CSV cell is set
to "nan" (the sentinel propagates).

Other columns in the main CSV (the 11 deterministic baselines + the
WAND columns) are left untouched.

Usage
-----
   python -m src.merge_seeds \
       --inputs results/seed0.csv,results/seed1.csv,results/seed2.csv

Optional:
   --main PATH    path to main CSV (default: results/bench_anomaly_wand.csv)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

STOCHASTIC = ["IForest", "PCA", "INNE", "LSCP"]
ROOT = Path(__file__).resolve().parents[2]


def _to_float(v):
    if v in ("", "None", None, "nan", "NaN"):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--main", type=str,
        default=str(ROOT / "results" / "bench_anomaly_wand.csv"),
    )
    p.add_argument("--inputs", type=str, required=True,
                   help="comma-separated paths to seed sidecar CSVs")
    args = p.parse_args()

    main_path = Path(args.main)
    sidecar_paths = [Path(x.strip()) for x in args.inputs.split(",") if x.strip()]
    if not sidecar_paths:
        raise SystemExit("--inputs requires at least one sidecar CSV")

    # Load main and sidecars.
    main_rows = list(csv.DictReader(main_path.open()))
    if not main_rows:
        raise SystemExit(f"main CSV {main_path} is empty")
    fieldnames = list(main_rows[0].keys())
    main_idx = {r["dataset"]: r for r in main_rows}

    sidecars = []
    for sp in sidecar_paths:
        if not sp.exists():
            raise SystemExit(f"sidecar not found: {sp}")
        sc_rows = list(csv.DictReader(sp.open()))
        sidecars.append({r["dataset"]: r for r in sc_rows})

    print(f"merging {len(sidecars)} sidecar(s) into {main_path}")
    print(f"methods averaged: {STOCHASTIC}\n")
    print(f"{'dataset':<22s} {'method':<8s} {'mean_AUC':>9s} "
          f"{'std':>7s} {'n_seeds':>8s}")
    print("-" * 60)

    n_changed = 0
    for ds, row in main_idx.items():
        for m in STOCHASTIC:
            aucs, aps, secs_ = [], [], []
            for sc in sidecars:
                r = sc.get(ds)
                if r is None:
                    continue
                aucs.append(_to_float(r.get(f"{m}_auc", "")))
                aps.append(_to_float(r.get(f"{m}_ap", "")))
                secs_.append(_to_float(r.get(f"{m}_secs", "")))

            aucs_ok = [a for a in aucs if not np.isnan(a)]
            aps_ok = [a for a in aps if not np.isnan(a)]
            secs_ok = [s for s in secs_ if not np.isnan(s)]

            if not aucs_ok:
                # All sentinels -> propagate sentinel.
                row[f"{m}_auc"] = "nan"
                row[f"{m}_ap"] = "nan"
                row[f"{m}_secs"] = "nan"
                print(f"{ds:<22s} {m:<8s} {'nan':>9s} "
                      f"{'--':>7s} {'0':>8s}")
                continue

            mean_auc = float(np.mean(aucs_ok))
            std_auc = float(np.std(aucs_ok)) if len(aucs_ok) > 1 else 0.0
            row[f"{m}_auc"] = f"{mean_auc:.6f}"
            row[f"{m}_ap"] = f"{np.mean(aps_ok):.6f}" if aps_ok else "nan"
            row[f"{m}_secs"] = f"{np.mean(secs_ok):.6f}" if secs_ok else "nan"
            n_changed += 1
            print(f"{ds:<22s} {m:<8s} {mean_auc:>9.4f} "
                  f"{std_auc:>7.4f} {len(aucs_ok):>8d}")

    with main_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(main_rows)
    print(f"\n# updated {n_changed} cell(s); wrote {main_path}")


if __name__ == "__main__":
    main()
