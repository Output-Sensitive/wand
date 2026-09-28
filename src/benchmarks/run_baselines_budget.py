"""
Fill in PyOD baseline cells for rows under a per-(dataset, method)
wall-clock + memory budget. Defaults: 1 h wall-clock, 16 GiB virtual
address space. Cells that complete are persisted; cells that hit a
timeout or the memory cap are stamped ``"nan"`` *and* appended to a
retry log (results/retry_flags.csv) so a future run with more compute
can pick them up.

NaN / Inf in the input matrix is sanitised in-place (column-median fill
plus percentile clipping) so flaky datasets still feed every method
rather than being dropped at PyOD's input-validation layer. NaN / Inf
in a method's decision_function output is also patched (median fill of
the finite scores) so the method's score can still feed the AUC / AP.

Usage
-----
   python -m src.run_baselines_budget [--budget 3600] [--max_mem_gb 16]
       [--datasets a,b,c] [--skip_methods OCSVM,...]

In-place updates results/bench_anomaly_wand.csv. Existing non-empty
baseline cells are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmarks._baseline_run import record_flag, safe_score, sanitize_X
from src.benchmarks.bench_anomaly_wand import _load, _make_baselines


# Methods known to be O(n^2) in time and/or memory; cheap to skip when
# n is large enough that no one-hour run is plausible on a 16 GiB box.
_QUADRATIC = {"LOF", "OCSVM", "KNN", "ABOD", "COF", "SOD", "KDE", "LSCP"}


def _should_attempt(method: str, n: int, n_cap: int) -> bool:
    if method in _QUADRATIC and n > n_cap:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=3600.0,
                        help="per-(dataset, method) wall-clock cap, seconds "
                             "(default 3600 = 1 h)")
    parser.add_argument("--max_mem_gb", type=float, default=16.0,
                        help="RLIMIT_AS cap in GiB; allocations beyond this "
                             "raise MemoryError and the cell is flagged for "
                             "future retry (default 16, 0 disables)")
    parser.add_argument("--n_cap_quad", type=int, default=20_000,
                        help="auto-skip O(n^2) methods for n > this "
                             "(default 20k; cell stays empty so a future "
                             "bigger-machine run can fill it)")
    parser.add_argument(
        "--csv", type=str,
        default=str(ROOT / "results" / "bench_anomaly_wand.csv"),
    )
    parser.add_argument("--retry_log", type=str,
                        default=str(ROOT / "results" / "retry_flags.csv"),
                        help="CSV that collects TIMEOUT / OOM flags for "
                             "future runs")
    parser.add_argument("--datasets", type=str, default=None,
                        help="restrict to comma-separated short names")
    parser.add_argument("--skip_methods", type=str, default="",
                        help="comma-separated methods to skip entirely")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    skip = set(s for s in args.skip_methods.split(",") if s)
    max_mem = args.max_mem_gb if args.max_mem_gb > 0 else None

    csv_path = Path(args.csv)
    rows = list(csv.DictReader(csv_path.open()))
    fieldnames = list(rows[0].keys())
    methods = [f[:-4] for f in fieldnames
               if f.endswith("_auc") and f != "WAND_auc" and f[:-4] not in skip]
    if args.datasets:
        keep = set(args.datasets.split(","))
        target_rows = [r for r in rows if r["dataset"] in keep]
    else:
        target_rows = rows

    dataset_dir = ROOT / "datasets" / "odds"
    factories = _make_baselines(seed=args.seed)

    print(f"budget = {args.budget:.0f}s, mem_cap = {args.max_mem_gb} GiB, "
          f"methods = {methods}")
    print(f"{'dataset':<22s} {'n':>7s} {'d':>5s}   results", flush=True)

    for row in target_rows:
        short = row["dataset"]
        missing = [m for m in methods
                   if not row.get(f"{m}_auc")
                   or row[f"{m}_auc"] in ("", "None")]
        if not missing:
            continue

        path = next((p for p in dataset_dir.glob(f"*_{short}.npz")), None)
        if path is None:
            print(f"  [skip] {short}: no .npz found", flush=True)
            continue

        try:
            X, y, _ = _load(path)
            X = sanitize_X(X)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {short}: load failed ({e})", flush=True)
            continue
        n, d = X.shape

        wins, losses = [], []
        for m in missing:
            if not _should_attempt(m, n, args.n_cap_quad):
                losses.append(f"{m}=skip")
                continue
            auc, ap, secs, flag = safe_score(
                factories[m], X, y,
                budget=args.budget, max_mem_gb=max_mem,
            )
            if auc is not None:
                row[f"{m}_auc"] = f"{auc:.6f}"
                row[f"{m}_ap"] = f"{ap:.6f}"
                row[f"{m}_secs"] = f"{secs:.6f}"
                tag = "*" if flag == "SALVAGED" else ""
                wins.append(f"{m}{tag}({auc:.3f}|{secs*1000:.0f}ms)")
            else:
                row[f"{m}_auc"] = row[f"{m}_ap"] = row[f"{m}_secs"] = "nan"
                losses.append(f"{m}={flag}")
                if flag in ("TIMEOUT", "OOM"):
                    record_flag(args.retry_log, short, m, args.seed, flag)
            gc.collect()

        line = f"{short[:22]:<22s} {n:>7d} {d:>5d}   "
        if wins:
            line += " " + " ".join(wins)
        if losses:
            line += "    miss: " + ",".join(losses)
        print(line, flush=True)

        # Incremental persistence: rewrite the full CSV after every
        # dataset so we never lose work if a later row OOMs / segfaults.
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        del X, y
        gc.collect()

    print(f"\n# wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
