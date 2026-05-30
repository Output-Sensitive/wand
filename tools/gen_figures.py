"""
Generate three figures for the main paper:
  - cd_auc.pdf       : Nemenyi critical-difference diagram, ranks by AUC
  - cd_ap.pdf        : Nemenyi critical-difference diagram, ranks by AP
  - time_auc.pdf     : Time-vs-AUC scatter, one marker per method

Reads results/bench_anomaly_wand.csv. Pure matplotlib; no extra deps.

Run: python paper/figures/_gen_figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "bench_anomaly_wand.csv"
OUT = ROOT / "figures"

METHODS = [
    "IForest", "LOF", "OCSVM", "KNN", "PCA", "HBOS", "ECOD", "COPOD",
    "ABOD", "COF", "SOD", "INNE", "LODA", "LSCP", "KDE", "PIDForest",
    "WAND",
]
HIGHLIGHT = "WAND"
# Display names: the CSV uses "WAND" as the column key (legacy), but
# the paper calls the method WAND. Map at draw time only.
DISPLAY = {"WAND": "WAND"}

# Nemenyi critical values q_alpha at alpha = 0.05.
# Index = k (number of methods). Source: Demsar 2006, table 5(a).
_Q_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
    14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
    20: 3.544,
}


def _ranks(df: pd.DataFrame, metric: str) -> tuple[pd.Series, int]:
    """Per-method mean Friedman rank, using the same NaN-tolerant
    methodology as Table~\\ref{tab:main}. We rank per row over the
    methods that have a value (NaN cells are skipped, not penalised),
    then average per method over the rows where that method ran. The
    `n` we return is the size of the fair-comparison subset (rows
    where Isolation Forest, our universal sub-quadratic baseline,
    completed within budget) -- the same subset whose row count
    Table~\\ref{tab:main} reports its summary lines over.
    """
    cols = [f"{m}_{metric}" for m in METHODS]
    sub = df[cols].copy()
    sub.columns = METHODS
    if "IForest" in METHODS:
        sub = sub[sub["IForest"].notna()]
    ranks = sub.rank(axis=1, ascending=False, method="average")
    return ranks.mean(skipna=True), len(sub)


def _cd(k: int, n: int, alpha: float = 0.05) -> float:
    q = _Q_005.get(k)
    if q is None:
        raise ValueError(f"no q-value for k={k}")
    return q * math.sqrt(k * (k + 1) / (6.0 * n))


def _draw_cd(ax, mean_rank: pd.Series, n: int, title: str, fscale: float = 1.0,
             line_pitch: float = 0.45, label_drop: float = 0.32) -> None:
    """Demsar-style CD diagram.

    Layout: rank axis at the top, methods listed in two columns at the
    outer edges (best ranks on the right, worst on the left). Labels are
    sorted from the axis midpoint outward in each column so the per-
    method drop-lines never cross. Cliques of methods that are NOT
    significantly different (within one CD) are drawn as thick black
    horizontal bars just below the axis.
    """
    methods = list(mean_rank.index)
    k = len(methods)
    cd = _cd(k, n)
    ranks = mean_rank.to_numpy()

    # Use the actual rank range rather than the full 1..k window: the
    # ranks all live in a sub-band of [1, k] and showing the empty tick
    # margins wastes horizontal space, squashing the diagram. We still
    # draw the integer ticks that fall inside the visible band so the
    # absolute rank scale is preserved.
    rmin = float(np.floor(ranks.min() - 0.4))
    rmax = float(np.ceil(ranks.max() + 0.4))
    lo, hi = max(1.0, rmin), min(float(k), rmax)
    # Reverse the x-axis so the best rank sits on the right.
    ax.set_xlim(hi + 0.4, lo - 0.4)
    # label_drop: vertical gap before any drop-line begins;
    # line_pitch: vertical pitch between adjacent labels (both args).
    # Count actual labels per side -- (k+1)//2 under-counts whenever the
    # split is uneven (e.g. k=16 with 9 ranks below the midpoint), which
    # would put the bottom-most label below the figure's y-limit and
    # clip it.
    mid_for_sizing = (lo + hi) / 2.0
    n_per_side = max(int(np.sum(ranks <  mid_for_sizing)),
                     int(np.sum(ranks >= mid_for_sizing)),
                     1)
    bottom_y = -(label_drop + (n_per_side + 0.5) * line_pitch + 0.3)
    ax.set_ylim(bottom_y, 1.65)
    ax.set_axis_off()

    # Main rank axis + ticks (integer ticks inside the visible band).
    ax.plot([lo, hi], [0, 0], "k-", lw=1.2)
    for r in range(int(lo), int(hi) + 1):
        ax.plot([r, r], [0, 0.10], "k-", lw=1.0)
        ax.text(r, 0.30, str(r), ha="center", va="bottom", fontsize=8 * fscale)

    # Title + CD reference bar (centred above the axis)
    ax.text((lo + hi) / 2, 1.30, title, ha="center", va="bottom",
            fontsize=10 * fscale, fontweight="bold")
    bar_y = 0.95
    bar_x0 = (lo + hi) / 2 + cd / 2
    bar_x1 = (lo + hi) / 2 - cd / 2
    ax.plot([bar_x0, bar_x1], [bar_y, bar_y], "k-", lw=2.2)
    ax.plot([bar_x0, bar_x0], [bar_y - 0.08, bar_y + 0.08], "k-", lw=1.5)
    ax.plot([bar_x1, bar_x1], [bar_y - 0.08, bar_y + 0.08], "k-", lw=1.5)
    ax.text((bar_x0 + bar_x1) / 2, bar_y + 0.13,
            f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8 * fscale)

    # Split methods at the rank-midpoint; place each side so the
    # topmost label is the one *closest to its own margin* on the axis
    # (shortest drop-line). Result: with the x-axis reversed (best on
    # the right), best ranks sit at the top of the right column and
    # worst ranks at the top of the left column -- no line ever crosses
    # another label, and the visual order matches the rank order.
    mid = (lo + hi) / 2.0
    pairs = list(zip(methods, ranks))
    left_side  = [(m, r) for m, r in pairs if r >= mid]   # worst-half
    right_side = [(m, r) for m, r in pairs if r <  mid]   # best-half
    # Top label = closest to that side's outer margin on the axis.
    # Left column sits at x = hi+0.30 (visually leftmost margin since
    # the axis is reversed), so its closest rank is the highest one;
    # sort descending. Right column sits at x = lo-0.30 (visually
    # rightmost margin), so its closest rank is rank 1; sort ascending.
    left_side  = sorted(left_side,  key=lambda p: -p[1])
    right_side = sorted(right_side, key=lambda p:  p[1])

    # Place labels OUTSIDE the rank-axis band, anchored on the side
    # that faces into the figure. The horizontal stub thus terminates
    # at the label's screen edge and never crosses the text glyphs,
    # so no white-bbox masking is needed and the line is fully visible.
    def _place(col_items, x_label, ha):
        for idx, (m, r) in enumerate(col_items):
            y = -(label_drop + (idx + 1) * line_pitch)
            colour = "C3" if m == HIGHLIGHT else "0.10"
            weight = "bold" if m == HIGHLIGHT else "normal"
            disp = DISPLAY.get(m, m)
            ax.plot([r, r], [0, y], color=colour, lw=1.0,
                    alpha=0.85, zorder=2)
            ax.plot([r, x_label], [y, y], color=colour, lw=1.0,
                    alpha=0.85, zorder=2)
            ax.text(x_label, y, f"  {disp}  ({r:.2f})  ",
                    ha=ha, va="center",
                    fontsize=8.5 * fscale, color=colour, fontweight=weight,
                    zorder=4)

    # Left column = worst-half ranks. x_label sits just past the right
    # end of the (reversed) axis -> visually the leftmost margin; with
    # ha="right" the text right-edge anchors there and extends further
    # left in screen, i.e. AWAY from the line, off the axis.
    _place(left_side,  hi + 0.30, ha="right")
    _place(right_side, lo - 0.30, ha="left")

    # Cliques of methods within one CD of each other, drawn as thin
    # horizontal bars just below the axis. We pack the bars onto as few
    # rows as possible by greedy-fitting non-overlapping cliques on each
    # row -- two bars sharing a row only if their x ranges are disjoint.
    order = np.argsort(ranks)
    sorted_ranks = ranks[order]
    raw_cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and (sorted_ranks[j + 1] - sorted_ranks[i]) <= cd:
            j += 1
        if j > i:
            raw_cliques.append((sorted_ranks[i], sorted_ranks[j]))
    # Keep maximal cliques only.
    pruned = []
    for a, b in raw_cliques:
        if any(aa <= a and bb >= b for aa, bb in pruned):
            continue
        pruned = [c for c in pruned if not (c[0] >= a and c[1] <= b)]
        pruned.append((a, b))
    pruned.sort()

    # Greedy row-packing: each row holds a set of non-overlapping ranges
    # (a, b). Bars on the same row don't touch (gap >= 0.10 in data
    # units).
    rows: list[list[tuple[float, float]]] = []
    bar_row = []  # per-clique row index, parallel to `pruned`
    for a, b in pruned:
        placed = False
        for ri, row in enumerate(rows):
            if all(b < ra - 0.10 or a > rb + 0.10 for ra, rb in row):
                row.append((a, b))
                bar_row.append(ri)
                placed = True
                break
        if not placed:
            rows.append([(a, b)])
            bar_row.append(len(rows) - 1)

    bar_y0 = -0.06
    bar_dy = 0.18
    for (a, b), ri in zip(pruned, bar_row):
        y = bar_y0 - bar_dy * ri
        ax.plot([a - 0.06, b + 0.06], [y, y], "k-", lw=2.0,
                solid_capstyle="round", zorder=2.5)


def _save_cd(metric: str, label: str, out_path: Path) -> None:
    df = pd.read_csv(CSV)
    mean_rank, n = _ranks(df, metric)
    # Width scales with number of methods so labels never overlap.
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    _draw_cd(ax, mean_rank, n, f"{label}  (n = {n} datasets)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"# wrote {out_path}  (n={n}, methods={len(mean_rank)})")


def _save_cd_combined(out_path: Path) -> None:
    """Stack both CD diagrams (AUC + AP) in one figure rendered at the
    target two-column display size. With k=16 methods, a single-column
    width forces the rank axis into a too-narrow band and labels
    collapse; the figure goes into figure*[t] in main.tex and is shown
    at near-textwidth in two-column layout."""
    df = pd.read_csv(CSV)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6))
    for ax, (metric, label) in zip(axes, [
        ("auc", "Critical-difference diagram (ROC-AUC)"),
        ("ap",  "Critical-difference diagram (AUPR / AP)"),
    ]):
        mean_rank, n = _ranks(df, metric)
        _draw_cd(ax, mean_rank, n, f"{label}  (n = {n})")
    fig.tight_layout(h_pad=1.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"# wrote {out_path}")


# ----------------------------------------------------------------------
# Time-vs-AUC scatter
# ----------------------------------------------------------------------

_MARKERS = {
    "IForest":  ("o",  "C0"),
    "LOF":      ("s",  "C1"),
    "OCSVM":    ("^",  "C2"),
    "KNN":      ("v",  "C3"),
    "PCA":      ("D",  "C4"),
    "HBOS":     ("P",  "C5"),
    "ECOD":     ("X",  "C6"),
    "COPOD":    ("*",  "C7"),
    "ABOD":     ("p",  "C8"),
    "COF":      ("h",  "C9"),
    "SOD":      ("<",  "tab:brown"),
    "INNE":     (">",  "tab:pink"),
    "LODA":     ("d",  "tab:olive"),
    "LSCP":     ("H",  "tab:cyan"),
    "KDE":      ("8",  "tab:gray"),
    "PIDForest":("o",  "tab:purple"),
    "WAND": ("*",  "red"),
}


def _save_time_auc(out_path: Path) -> None:
    """Mean AUC vs mean runtime, one marker per method.

    Renders at a near-textwidth size so the per-method labels stay
    legible at 8.5 pt when the figure is included in a two-column
    figure*. Uses the same DISPLAY rename map ("WAND" -> "WAND")
    as the CD diagrams. Marker for WAND is a large red star with a
    black edge; the others are smaller and use distinct
    matplotlib palette colours.
    """
    df = pd.read_csv(CSV)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    # Fair comparison: average every method over the SAME datasets, namely
    # those completed by *all* methods. Otherwise each method's mean is
    # taken over its own (often easier) subset -- the slow neighbour-based
    # methods skip the large datasets and look artificially strong.
    common = pd.Series(True, index=df.index)
    for m in METHODS:
        auc = pd.to_numeric(df[f"{m}_auc"], errors="coerce")
        sec = pd.to_numeric(df[f"{m}_secs"], errors="coerce")
        common &= auc.notna() & sec.notna() & (sec > 0)
    n_common = int(common.sum())

    # Collect per-method (mean_sec, mean_auc) on the common subset so we
    # can place labels with anti-overlap offsets.
    points = []
    for m in METHODS:
        auc = pd.to_numeric(df[f"{m}_auc"], errors="coerce")[common]
        sec = pd.to_numeric(df[f"{m}_secs"], errors="coerce")[common]
        if len(auc) == 0:
            continue
        points.append((m, float(sec.mean()), float(auc.mean())))

    # Per-method label offsets in (dx, dy) points. Auto-tuned to fan
    # out the labels in the dense low-time / mid-AUC cluster (PCA,
    # COPOD, ECOD, HBOS, OCSVM, IForest) so each name sits beside its
    # own marker without overlapping any neighbour.
    label_off = {
        "WAND": (  9,   1),   # bold, right of the star (top of frontier)
        "IForest":  (-16, -12),
        "INNE":     (  8,   5),
        "KNN":      (  6, -13),
        "KDE":      (  2,   9),
        "LSCP":     (-40,   6),
        "SOD":      (  8,   5),
        "COF":      (  8,   5),
        "ABOD":     (  8,  -3),
        "LOF":      (  8,  -3),
        "LODA":     ( -8, -13),
        "OCSVM":    (  6, -13),
        "PCA":      (  9,  -3),
        "HBOS":     (  4,   7),
        "ECOD":     (  8, -11),
        "COPOD":    (  3,   8),
        "PIDForest":(  8,  -3),
    }
    label_dy = {m: label_off.get(m, (7, 7)) for m, _, _ in points}

    for m, x, y in points:
        marker, colour = _MARKERS.get(m, ("o", "black"))
        is_hi = m == HIGHLIGHT
        size = 260 if is_hi else 110
        edge = "black" if is_hi else "white"
        ax.scatter(x, y, s=size, marker=marker, color=colour,
                   edgecolor=edge, linewidths=0.9,
                   zorder=4 if is_hi else 2.5, label=DISPLAY.get(m, m))
        disp = DISPLAY.get(m, m)
        dx, dy = label_dy[m]
        weight = "bold" if is_hi else "normal"
        ax.annotate(disp, (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=11,
                    fontweight=weight,
                    color="C3" if is_hi else "black")

    ax.set_xscale("log")
    ax.set_xlabel("Mean wall-clock per dataset (s, log scale)",
                  fontsize=12)
    ax.set_ylabel("Mean ROC-AUC across datasets", fontsize=12)
    ax.set_ylim(0.638, 0.775)   # headroom so the WAND star clears the title
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.set_title(f"AUC vs. runtime ({n_common} datasets run by all methods)",
                 fontsize=12, fontweight="bold")
    # "Better" arrow pointing to the lower-time / higher-AUC corner.
    ax.annotate("better",
                xy=(0.02, 0.96), xytext=(0.15, 0.80),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=11, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"# wrote {out_path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _save_cd_combined(OUT / "cd_both.pdf")
    _save_time_auc(OUT / "time_auc.pdf")


if __name__ == "__main__":
    main()
