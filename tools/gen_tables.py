"""Regenerate the main results table from the bench CSV.

Run: python paper/sections/_gen_tables.py
Writes:
  paper/sections/tab_main.tex   -- AUC matrix, 23 datasets x 9 methods
  paper/sections/tab_summary.tex -- avg AUC / rank / #wins row
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "bench_anomaly_wand.csv"
OUT_DIR = ROOT / "tables"

METHODS = [
    "IForest", "LOF", "OCSVM", "KNN", "PCA", "HBOS", "ECOD", "COPOD",
    "ABOD", "COF", "SOD", "INNE", "LODA", "LSCP", "KDE", "PIDForest",
    "WAND",
]


def main() -> None:
    df = pd.read_csv(CSV)
    auc = df[[f"{m}_auc" for m in METHODS]].copy()
    auc.columns = METHODS

    # Identify the "fair-comparison" subset: rows where the universal
    # baseline (IForest) ran. That gives exactly the original 27
    # Small + Medium ADBench rows that all of the PyOD methods saw.
    # The remaining 20 rows are WAND-only entries from the batched
    # sweep on the large / high-dim datasets; we keep them in the body
    # but exclude them from the per-method rank / share-of-wins
    # summary so WAND doesn't get free rank-1 credit on rows where
    # no competitor ran. Sporadic single-method failures (e.g. ABOD
    # erroring on a small dataset) are tolerated by the per-method
    # mean-with-skipna and the rank computation, which already ignore
    # NaN cells row-wise.
    fair_mask = auc["IForest"].notna()
    n_fair = int(fair_mask.sum())
    n_total = len(auc)

    # --- per-row best/second-best for bolding (with ties) ---
    # We round to 3 decimals first, so values that *display* as identical
    # (e.g. five methods all reporting 1.000 on musk) are all bolded.
    # The second-best tier is then any value strictly below the displayed
    # max but tied for the next-highest displayed value.
    arr = auc.to_numpy()
    nan_mask = np.isnan(arr)
    arr_rounded = np.where(nan_mask, -np.inf, np.round(arr, 3))
    row_max = arr_rounded.max(axis=1, keepdims=True)
    is_best = (arr_rounded == row_max) & ~nan_mask
    # second tier: max among values strictly less than row_max
    masked_second = np.where(arr_rounded < row_max, arr_rounded, -np.inf)
    row_second = masked_second.max(axis=1, keepdims=True)
    has_second = (row_second > -np.inf)
    is_second = (arr_rounded == row_second) & has_second & ~nan_mask

    # Shorten the longest dataset label to keep the body inside the page
    # margin without shrinking the per-cell font.
    DISPLAY = {"Cardiotocography": "Cardiotoco."}

    # --- main table body ---
    body = []
    for i, row in df.iterrows():
        label = DISPLAY.get(row["dataset"], row["dataset"])
        cells = [f"\\texttt{{{label}}}",
                 f"{int(row['n'])}",
                 f"{int(row['d'])}",
                 f"{100 * row['contam']:.1f}\\%"]
        for j, m in enumerate(METHODS):
            v = row[f"{m}_auc"]
            if pd.isna(v):
                cells.append("--")
            elif is_best[i, j]:
                cells.append(f"\\textbf{{{v:.3f}}}")
            elif is_second[i, j]:
                cells.append(f"\\underline{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        body.append(" & ".join(cells) + " \\\\")

    # --- summary statistics (tie-aware), computed on the fair-subset ---
    # `rank(..., method='average')` already splits ranks fairly on ties:
    # k methods tied for the top position all receive rank (1+...+k)/k.
    auc_fair = auc.loc[fair_mask]
    ranks_fair = auc_fair.rank(axis=1, ascending=False, method="average")
    avg_auc = auc_fair.mean()
    avg_rank = ranks_fair.mean()

    # Fair "share-of-wins": on a row where k methods tie for the best
    # (after rounding to display precision), each tied method receives
    # 1/k credit. Restricted to the fair-subset rows for the same
    # reason as the rank computation.
    fair_arr = arr[fair_mask.values]
    fair_nan_mask = np.isnan(fair_arr)
    arr_round = np.where(fair_nan_mask, -np.inf, np.round(fair_arr, 3))
    row_max_val = arr_round.max(axis=1, keepdims=True)
    tied_winners = (arr_round == row_max_val) & ~fair_nan_mask   # (n_fair, n_methods)
    n_tied = tied_winners.sum(axis=1, keepdims=True).clip(min=1)
    win_credit = tied_winners.astype(float) / n_tied             # (n_fair, n_methods)
    win_shares = win_credit.sum(axis=0)                          # (n_methods,)
    wins = pd.Series(win_shares, index=METHODS)

    def fmt_auc(v, is_best, is_second):
        return (f"\\textbf{{{v:.3f}}}" if is_best
                else f"\\underline{{{v:.3f}}}" if is_second
                else f"{v:.3f}")
    def fmt_rank(v, is_best, is_second):
        return (f"\\textbf{{{v:.2f}}}" if is_best
                else f"\\underline{{{v:.2f}}}" if is_second
                else f"{v:.2f}")
    def fmt_wins(v, is_best, is_second):
        s = f"{v:.1f}" if abs(v - round(v)) > 1e-6 else f"{int(round(v))}"
        return (f"\\textbf{{{s}}}" if is_best
                else f"\\underline{{{s}}}" if is_second
                else s)

    # Tied bold/underline for each summary row.
    def best_and_second(series: pd.Series, direction: str = "max"):
        """Return (best_set, second_set) given a Series.

        `direction='max'` picks the highest value(s); 'min' the lowest.
        Ties at the displayed precision are grouped. The second set is
        empty if the best group covers everyone.
        """
        if direction == "max":
            top_val = series.max()
            best = set(series[np.isclose(series, top_val, atol=5e-4)].index)
            below = series[~series.index.isin(best)]
            if below.empty:
                return best, set()
            sec_val = below.max()
            second = set(below[np.isclose(below, sec_val, atol=5e-4)].index)
            return best, second
        else:  # "min"
            top_val = series.min()
            best = set(series[np.isclose(series, top_val, atol=5e-3)].index)
            below = series[~series.index.isin(best)]
            if below.empty:
                return best, set()
            sec_val = below.min()
            second = set(below[np.isclose(below, sec_val, atol=5e-3)].index)
            return best, second

    auc_best,  auc_second  = best_and_second(avg_auc,  "max")
    rk_best,   rk_second   = best_and_second(avg_rank, "min")
    wins_best, wins_second = best_and_second(wins,     "max")

    # Four ampersands separate the label cell + 3 empty cells from the
    # first method value (4 leading non-method columns: Dataset/n/d/contam).
    pad = " & & & & "
    line_auc = (f"Avg.\\ AUC" + pad
                + " & ".join(fmt_auc(avg_auc[m], m in auc_best, m in auc_second)
                             for m in METHODS) + " \\\\")
    line_rank = (f"Avg.\\ rank" + pad
                 + " & ".join(fmt_rank(avg_rank[m], m in rk_best, m in rk_second)
                              for m in METHODS) + " \\\\")
    line_wins = (f"Share of wins" + pad
                 + " & ".join(fmt_wins(wins[m], m in wins_best, m in wins_second)
                              for m in METHODS) + " \\\\")

    # NOTE: the standalone "WAND avg AUC" row and the per-dataset
    # 1-D oracle row are no longer rendered inside tab_main, since
    # competing methods do not have an equivalent. The full-suite and
    # oracle AUC are reported as callouts in the body text.

    header = ("Dataset & $n$ & $d$ & Contam. & "
              + " & ".join(METHODS) + " \\\\")
    header = header.replace("WAND", "\\textbf{WAND}")

    table_body = "\n".join(body)
    summary_lines = [line_auc, line_rank, line_wins]
    summary_body = "\n".join(summary_lines)

    # Write the body and summary as two snippets so we can lay out the
    # table in main.tex with a nice layout.
    # No trailing newline -- otherwise LaTeX's \input adds an extra
    # space token before the following \midrule, which trips up
    # booktabs's \noalign placement.
    (OUT_DIR / "tab_main_header.tex").write_text(header)
    (OUT_DIR / "tab_main_body.tex").write_text(table_body)
    (OUT_DIR / "tab_main_summary.tex").write_text(summary_body)

    # Emit a single .tex that *includes* \begin{tabular}..\end{tabular}.
    # Column spec adapts to the number of methods so we can grow the
    # comparison without re-tuning by hand.
    col_spec = "lrrr" + "c" * len(METHODS)
    tabular = (
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        + header + "\n"
        "\\midrule\n"
        + table_body + "\n"
        "\\midrule\n"
        + summary_body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    (OUT_DIR / "tab_main.tex").write_text(tabular)

    print(f"# wrote {OUT_DIR / 'tab_main.tex'}")

    def _pretty(group, series, fmt=".3f"):
        return "{" + ", ".join(
            f"{m}={series[m]:{fmt}}" for m in METHODS if m in group
        ) + "}"

    print()
    print("Summary (ties are aggregated):")
    print(f"  Avg AUC      -- best={_pretty(auc_best,  avg_auc, '.3f')}  "
          f"2nd={_pretty(auc_second, avg_auc, '.3f')}")
    print(f"  Avg rank     -- best={_pretty(rk_best,   avg_rank, '.2f')}  "
          f"2nd={_pretty(rk_second,  avg_rank, '.2f')}")
    print(f"  Share-of-wins-- best={_pretty(wins_best,  wins, '.2f')}  "
          f"2nd={_pretty(wins_second, wins, '.2f')}")


if __name__ == "__main__":
    main()
