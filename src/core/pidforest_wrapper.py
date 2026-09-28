"""
PyOD-style wrapper around the reference PIDForest implementation
(https://github.com/vatsalsharan/pidforest, Gopalan, Sharan, Wieder,
NeurIPS 2019).

Exposes a class with `fit(X)` and `decision_function(X)` so the bench
scripts can call it the same way they call any PyOD estimator. The
underlying Forest is fed `X.T` (features-by-samples in the reference
implementation) and the raw score (lower = more anomalous) is
\emph{negated} to match the PyOD convention (higher = more anomalous).

Constructor accepts a `random_state` keyword (seeded via
`numpy.random.seed`) so the seed sweep in `run_baselines_budget.py`
and `run_one_seed.py` works without changes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Location of the reference PIDForest code. Override with the
# WAND_PIDFOREST_PATH env var; otherwise default to a `third_party`
# clone under the repo root (see README), i.e.
#   git clone https://github.com/vatsalsharan/pidforest third_party/pidforest
# It is imported lazily so a missing clone surfaces as a clean
# ImportError only when PIDForest is actually used.
_DEFAULT_PIDFOREST = (
    Path(__file__).resolve().parents[2] / "third_party" / "pidforest" / "code"
)
_PIDFOREST_PATH = os.environ.get("WAND_PIDFOREST_PATH", str(_DEFAULT_PIDFOREST))


def _import_forest():
    if _PIDFOREST_PATH not in sys.path:
        sys.path.insert(0, _PIDFOREST_PATH)
    from scripts.forest import Forest  # noqa: WPS433
    return Forest


class PIDForest:
    """PyOD-style PIDForest detector.

    Default hyperparameters mirror the values used in the reference
    classification experiments (max_depth=10, n_trees=50, max_samples=100,
    max_buckets=3, epsilon=0.1, sample_axis=1, threshold=0).
    """

    def __init__(
        self,
        n_trees: int = 50,
        max_depth: int = 10,
        max_samples: int = 100,
        max_buckets: int = 3,
        epsilon: float = 0.1,
        sample_axis: float = 1.0,
        threshold: float = 0.0,
        err: float = 0.1,
        pct: int = 50,
        random_state: int = 0,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.max_buckets = max_buckets
        self.epsilon = epsilon
        self.sample_axis = sample_axis
        self.threshold = threshold
        self.err = err
        self.pct = pct
        self.random_state = random_state
        self._forest = None

    def fit(self, X: np.ndarray) -> "PIDForest":
        Forest = _import_forest()
        np.random.seed(self.random_state)
        # max_samples must be <= n; clamp defensively
        n = X.shape[0]
        ms = max(2, min(self.max_samples, n))
        self._forest = Forest(
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            max_samples=ms,
            max_buckets=self.max_buckets,
            epsilon=self.epsilon,
            sample_axis=self.sample_axis,
            threshold=self.threshold,
        )
        self._forest.fit(np.asarray(X, dtype=float).T)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self._forest is None:
            raise RuntimeError("PIDForest must be fitted before decision_function")
        _, _, _, _, scores = self._forest.predict(
            np.asarray(X, dtype=float).T,
            err=self.err,
            pct=self.pct,
        )
        # PIDForest convention: lower score = more anomalous. PyOD: higher = more anomalous.
        return -np.asarray(scores, dtype=float)
