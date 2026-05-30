"""
WAND-only sweep across all 47 ADBench datasets, batched.

Useful for two scenarios:
  - validating the batched scoring path scales to the giants
    (n up to 619k) on a memory-limited box;
  - producing a clean per-dataset WAND AUC / time table when the
    PyOD baselines have already been run separately (so we don't
    re-pay their cost).

Output: results/bench_wand_only.csv  -- one row per dataset with
  dataset, n, d, contam, WAND_auc, WAND_ap, WAND_secs.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core.anticoncentration import WANDConfig, wand_score
from src.benchmarks.bench_anomaly_wand import DATASETS, _load


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default=None,
                   help="comma-separated dataset codes; default = full DATASETS")
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--n_stats_subsample", type=int, default=5000)
    p.add_argument("--out", type=str,
                   default=str(ROOT / "results" / "bench_wand_only.csv"))
    args = p.parse_args()

    ds_list = args.datasets.split(",") if args.datasets else DATASETS

    print(f'{"dataset":<22s} {"n":>8s} {"d":>5s} {"%anom":>7s}'
          f' {"AUC":>7s} {"AP":>7s} {"secs":>8s}', flush=True)
    print("-" * 70, flush=True)

    rows = []
    ds_dir = ROOT / "datasets" / "odds"
    for name in ds_list:
        path = ds_dir / f"{name}.npz"
        if not path.exists():
            print(f"[skip] missing {path}", flush=True)
            continue
        X, y, short = _load(path)
        n, d = X.shape
        contam = float(y.mean())
        t0 = time.perf_counter()
        try:
            cfg = WANDConfig(
                K=args.K, seed=args.seed,
                batch_size=args.batch_size,
                n_stats_subsample=args.n_stats_subsample,
                n_seeds=1,
            )
            score = wand_score(
                X, K=cfg.K, seed=cfg.seed,
                batch_size=cfg.batch_size,
                n_stats_subsample=cfg.n_stats_subsample,
                n_seeds=cfg.n_seeds,
            )
            secs = time.perf_counter() - t0
            auc = float(roc_auc_score(y, score))
            ap = float(average_precision_score(y, score))
            print(f'{short[:22]:<22s} {n:>8d} {d:>5d} {100*contam:>6.2f}%'
                  f' {auc:>7.3f} {ap:>7.3f} {secs:>8.1f}', flush=True)
            rows.append({
                "dataset": short, "n": n, "d": d, "contam": contam,
                "WAND_auc": auc, "WAND_ap": ap, "WAND_secs": secs,
            })
        except Exception as e:
            secs = time.perf_counter() - t0
            print(f'{short[:22]:<22s} {n:>8d} {d:>5d} {100*contam:>6.2f}%'
                  f' {"ERR":>7s} {"ERR":>7s} {secs:>8.1f}    {e}', flush=True)
            rows.append({
                "dataset": short, "n": n, "d": d, "contam": contam,
                "WAND_auc": None, "WAND_ap": None,
                "WAND_secs": secs,
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["dataset", "n", "d", "contam",
                           "WAND_auc", "WAND_ap", "WAND_secs"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\n# wrote {out}", flush=True)


if __name__ == "__main__":
    main()
