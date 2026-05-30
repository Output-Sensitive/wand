"""
Fill the PIDForest_auc / PIDForest_ap / PIDForest_secs columns in
results/bench_anomaly_wand.csv with the mean over three external
RNG seeds (0, 1, 2), matching the protocol used for the other
stochastic baselines.

Per (dataset, seed) we run with a wall-clock budget and a virtual
address-space cap; rows that exceed either are stamped "nan" and the
mean is taken over the remaining seeds. If all three seeds fail, the
row's PIDForest cells stay "nan".

Usage
-----
   python -m src.run_pidforest [--budget 600] [--max_mem_gb 16]
       [--datasets a,b,c] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmarks._baseline_run import record_flag, safe_score, sanitize_X
from src.benchmarks.bench_anomaly_wand import _load
from src.core.pidforest_wrapper import PIDForest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=600.0,
                   help="per-(dataset, seed) wall-clock cap, seconds")
    p.add_argument("--max_mem_gb", type=float, default=16.0,
                   help="virtual address space cap, GB (0 disables)")
    p.add_argument("--csv", type=str,
                   default=str(ROOT / "results" / "bench_anomaly_wand.csv"))
    p.add_argument("--retry_log", type=str,
                   default=str(ROOT / "results" / "retry_flags.csv"))
    p.add_argument("--datasets", type=str, default=None,
                   help="restrict to comma-separated short names")
    p.add_argument("--seeds", type=str, default="0,1,2")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    max_mem = args.max_mem_gb if args.max_mem_gb > 0 else None
    csv_path = Path(args.csv)
    rows = list(csv.DictReader(csv_path.open()))
    fieldnames = list(rows[0].keys())
    if args.datasets:
        keep = set(args.datasets.split(","))
        target_rows = [r for r in rows if r["dataset"] in keep]
    else:
        target_rows = rows
    dataset_dir = ROOT / "datasets" / "odds"

    print(f"PIDForest sweep: seeds={seeds} budget={args.budget:.0f}s "
          f"mem_cap={args.max_mem_gb} GB", flush=True)
    print(f"{'dataset':<22s} {'n':>7s} {'d':>5s}   per-seed AUC", flush=True)

    for row in target_rows:
        short = row["dataset"]
        path = next((p_ for p_ in dataset_dir.glob(f"*_{short}.npz")), None)
        if path is None:
            print(f"  [skip] {short}: no .npz", flush=True)
            continue
        try:
            X, y, _ = _load(path)
            X = sanitize_X(X)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {short}: load failed ({e})", flush=True)
            continue
        n, d = X.shape

        aucs, aps, secs_list, flags = [], [], [], []
        for s in seeds:
            factory = lambda s=s: PIDForest(random_state=s)
            auc, ap, secs, flag = safe_score(
                factory, X, y,
                budget=args.budget, max_mem_gb=max_mem,
                sanitize_input=False,  # already done above
            )
            if auc is not None:
                aucs.append(auc); aps.append(ap); secs_list.append(secs)
            flags.append(flag if flag else "OK")
            if flag in ("TIMEOUT", "OOM"):
                record_flag(args.retry_log, short, "PIDForest", s, flag)
            gc.collect()

        if aucs:
            row["PIDForest_auc"]  = f"{float(np.mean(aucs)):.6f}"
            row["PIDForest_ap"]   = f"{float(np.mean(aps)):.6f}"
            row["PIDForest_secs"] = f"{float(np.mean(secs_list)):.6f}"
            tag = f" mean_AUC={float(np.mean(aucs)):.3f} over {len(aucs)} seeds"
        else:
            row["PIDForest_auc"] = "nan"
            row["PIDForest_ap"]  = "nan"
            row["PIDForest_secs"] = "nan"
            tag = " (all seeds failed)"
        print(f"  {short[:22]:<22s} {n:>7d} {d:>5d}   "
              f"flags={'/'.join(flags)}{tag}", flush=True)

        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        del X, y
        gc.collect()

    print(f"\n# wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
