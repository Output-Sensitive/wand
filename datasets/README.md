# Datasets

The raw data is **not** bundled (size / redistribution). Put it here as
described below; the bundled `results/*.csv` already cover a full run, so
figures and tables regenerate without re-downloading anything.

## ADBench / ODDS (`datasets/odds/`)

The 47 tabular tasks of [ADBench](https://github.com/Minqi824/ADBench). Each
file is an `.npz` with two arrays:

- `X` — `(n, d)` float feature matrix
- `y` — `(n,)` binary labels (`1` = anomaly), used for evaluation only

Name the files `<index>_<short>.npz`, zero-padded index in benchmark order:

```
datasets/odds/
├── 01_breastw.npz
├── 02_cardio.npz
├── ...
└── 46_speech.npz
```

The benchmark drivers (`src/benchmarks/`) discover datasets by this pattern.

### Conversion template

```python
import numpy as np
# `arr` from any source with features and labels:
np.savez(f"datasets/odds/{idx:02d}_{name}.npz", X=X.astype(np.float32), y=y.astype(int))
```

## AnoCUB (`datasets/cub/`)

Derived from [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/).
Download and extract the dataset archive into `datasets/cub/`, then run:

```bash
python src/experiments/exp_e7_anocub.py
```

This builds and caches the exact AnoCUB task (`datasets/cub/anocub_task.npz`,
a 1259×312 named-concept anomaly-detection task) and renders the concept-level
explanation; `exp_e8` / `exp_e9` add the pixel-saliency and gallery views.
