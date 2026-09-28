"""
Shared safe-execution wrapper for the PyOD baseline fillers.

Provides:
  - `safe_score(factory, X, y, *, budget, max_mem_gb=None)`:
      run a PyOD baseline with a wall-clock cap and an optional virtual-
      address-space cap, sanitise NaN / inf in the raw decision_function
      output if possible, and return (auc, ap, secs, flag).
  - `record_flag(retry_path, dataset, method, seed, flag)`:
      append a flagged cell to a CSV of (dataset, method, seed, flag,
      timestamp) tuples so a future run can pick the cell up with more
      compute.

The four `flag` outcomes are:
  None        - clean success (numeric AUC / AP, finite scores)
  'SALVAGED'  - success but NaN / inf appeared in the raw scores; we
                replaced them with the median of the finite scores so
                the metrics could still be computed. AUC is degraded
                quality but not arbitrary.
  'TIMEOUT'   - wall time exceeded `budget`. Cell value left empty;
                appended to retry list.
  'OOM'       - allocation hit `max_mem_gb` cap. Cell value left empty;
                appended to retry list.
  'ERR'       - PyOD raised, or output was entirely non-finite. Cell
                value stamped "nan" (permanent skip).
"""

from __future__ import annotations

import contextlib
import resource
import signal
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class _Timeout(Exception):
    pass


@contextlib.contextmanager
def _wall(seconds: float):
    """Per-call wall-clock timeout via setitimer(ITIMER_REAL)."""
    def _h(signum, frame):  # noqa: ARG001
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 1e-3))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def sanitize_X(X: np.ndarray) -> np.ndarray:
    """Replace NaN / Inf in X with the column-median of finite values
    (or zero if a column is entirely non-finite) and clip extreme
    magnitudes that trip PyOD's PCA / ABOD float-overflow checks.
    Used to keep flaky datasets running through PyOD baselines that
    refuse non-finite input.
    """
    X = np.asarray(X, dtype=np.float64)
    if np.isfinite(X).all():
        return X
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        finite_mask = np.isfinite(col)
        if finite_mask.all():
            continue
        fill = float(np.median(col[finite_mask])) if finite_mask.any() else 0.0
        col_c = np.where(finite_mask, col, fill)
        finite_f = col_c[np.isfinite(col_c)]
        if finite_f.size:
            lo, hi = np.percentile(finite_f, [0.01, 99.99])
            spread = max(hi - lo, 1.0)
            col_c = np.clip(col_c, lo - 10 * spread, hi + 10 * spread)
        X[:, j] = col_c
    return X


def _set_mem_limit(gb: float) -> None:
    """Cap RLIMIT_AS to `gb` gigabytes for the current process.

    `RLIMIT_AS` controls virtual address space, so an overshoot causes
    `malloc()` (and thus numpy / sklearn allocations) to fail with
    MemoryError -- which we catch. Idempotent: a non-privileged process
    cannot raise a soft limit once set, so calling this with a *larger*
    `gb` than the current soft limit is silently a no-op.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = int(gb * (1024 ** 3))
    if soft != resource.RLIM_INFINITY and new_soft >= soft:
        return
    new_hard = hard if hard != resource.RLIM_INFINITY else new_soft
    try:
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))
    except (ValueError, OSError):
        pass


def safe_score(
    factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    *,
    budget: float,
    max_mem_gb: Optional[float] = None,
    sanitize_input: bool = True,
) -> Tuple[Optional[float], Optional[float], float, Optional[str]]:
    """Fit + decision_function under wall-clock / memory caps.

    The method is allowed to misbehave: if its decision_function returns
    NaN / inf for some points we silently replace them with the median
    of the finite scores and compute the metrics anyway. This lets
    flaky PyOD detectors (ABOD, LSCP, PCA on certain rows) still
    contribute a usable number to the table instead of being dropped.

    Returns (auc, ap, secs, flag). On TIMEOUT / OOM the (auc, ap) pair
    is (None, None) and `secs` reports the wall time consumed so far.
    """
    if max_mem_gb is not None:
        _set_mem_limit(max_mem_gb)
    if sanitize_input:
        X = sanitize_X(X)

    t0 = time.perf_counter()
    try:
        with _wall(budget):
            mdl = factory()
            mdl.fit(X)
            s = mdl.decision_function(X)
        secs = time.perf_counter() - t0
    except _Timeout:
        return None, None, float(budget), "TIMEOUT"
    except MemoryError:
        return None, None, time.perf_counter() - t0, "OOM"
    except Exception:  # noqa: BLE001
        return None, None, time.perf_counter() - t0, "ERR"

    s = np.asarray(s, dtype=np.float64)
    flag: Optional[str] = None
    if not np.all(np.isfinite(s)):
        finite = s[np.isfinite(s)]
        if len(finite) < 2 or float(np.std(finite)) == 0.0:
            return None, None, secs, "ERR"
        s = np.where(np.isfinite(s), s, float(np.median(finite)))
        flag = "SALVAGED"

    try:
        auc = float(roc_auc_score(y, s))
        ap = float(average_precision_score(y, s))
    except Exception:  # noqa: BLE001
        return None, None, secs, "ERR"
    return auc, ap, secs, flag


def record_flag(
    retry_path: Path | str,
    dataset: str,
    method: str,
    seed: int,
    flag: str,
) -> None:
    """Append a flagged (dataset, method, seed) to a retry CSV."""
    p = Path(retry_path)
    new = not p.exists()
    with p.open("a") as f:
        if new:
            f.write("dataset,method,seed,flag,timestamp\n")
        f.write(f"{dataset},{method},{seed},{flag},{int(time.time())}\n")
