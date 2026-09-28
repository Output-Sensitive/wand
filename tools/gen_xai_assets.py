"""Generate explanation-quality table + figures for the paper from the
E1 (two-regime synthetic), E2 (faithfulness) and E5 (heavy-tail) CSVs.

Outputs:
  tables/tab_xai.tex     -- main explanation-quality + cost table
  figures/xai_scaling.pdf  -- (a) attribution-AUC by regime;
                                    (b) faithfulness vs query cost
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

PRETTY = {"witness": r"\textsc{Wand}-witness",
          "gradient": r"\textsc{Wand}-gradient",
          "ecod": "ECOD (native)", "shap": "SHAP (post-hoc)",
          "lime": "LIME (post-hoc)", "random": "Random"}
ORDER = ["witness", "gradient", "ecod", "shap", "lime", "random"]


def main():
    e1 = pd.read_csv(RES / "e1_synthetic.csv")
    e2 = pd.read_csv(RES / "e2_faithfulness.csv")
    nd = e2.dataset.nunique()

    # E1: attribution-AUC per regime; detection-AUC per detector per regime
    attr = e1.groupby(["method", "kind"])["attr_auc"].mean().unstack()
    det = e1.groupby(["method", "kind"])["det_auc"].mean().unstack()

    # E2: mean faithfulness, win-vs-SHAP, cost
    piv = e2.pivot(index="dataset", columns="method", values="faithful")
    g2 = e2.groupby("method").agg(faith=("faithful", "mean"),
                                  q2=("queries", "mean"),
                                  sec=("sec_per_expl", "mean"))
    win = {m: (piv[m] >= piv["shap"]).mean() for m in piv.columns}

    # ---------------- table ----------------
    L = [r"\begin{tabular}{lccccccc}", r"\toprule",
         r"& \multicolumn{2}{c}{Synthetic attr-AUC} & \multicolumn{2}{c}{Real faithfulness} & \multicolumn{2}{c}{Cost / expl.} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Method & axis & oblique & mean & $\geq$SHAP & queries & ms \\",
         r"\midrule"]
    best_ax = attr.loc[[m for m in ORDER if m in attr.index], "axis"].max()
    best_ob = attr.loc[[m for m in ORDER if m in attr.index], "oblique"].max()
    for m in ORDER:
        ax = f"{attr.loc[m,'axis']:.3f}" if m in attr.index else "--"
        ob = f"{attr.loc[m,'oblique']:.3f}" if m in attr.index else "--"
        if m in attr.index and attr.loc[m, "axis"] >= best_ax - 1e-9:
            ax = f"\\textbf{{{ax}}}"
        if m in attr.index and attr.loc[m, "oblique"] >= best_ob - 1e-9:
            ob = f"\\textbf{{{ob}}}"
        fa = f"{g2.loc[m,'faith']:.3f}" if m in g2.index else "--"
        wn = "--" if m == "shap" else (f"{100*win[m]:.0f}\\%" if m in win else "--")
        q = g2.loc[m, "q2"] if m in g2.index else 0
        qq = "$0$" if q < 1 else f"${q:,.0f}$".replace(",", "{,}")
        ms = f"{g2.loc[m,'sec']*1000:.2f}" if m in g2.index else "--"
        L.append(f"{PRETTY[m]} & {ax} & {ob} & {fa} & {wn} & {qq} & {ms} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "tables" / "tab_xai.tex").write_text("\n".join(L))
    print("wrote tab_xai.tex")

    # detection-AUC summary (for the caption / text)
    print("detection AUC (WAND via witness row, ECOD):")
    for k in ["axis", "oblique"]:
        print(f"  {k}: WAND={det.loc['witness',k]:.3f}  ECOD={det.loc['ecod',k]:.3f}")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Large fonts + compact figsize so labels stay legible after the figure
    # is down-scaled to a single column (~3.4in) in the paper.
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11,
                         "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 9})
    col = {"witness": "#2c7fb8", "gradient": "#41b6c4", "ecod": "#74c476",
           "shap": "#d95f0e", "lime": "#fec44f", "random": "#999999"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2))

    # (a) grouped bars: attribution-AUC, axis vs oblique
    methods = ["witness", "gradient", "ecod", "shap", "lime"]
    xpos = np.arange(2)        # axis, oblique
    w = 0.16
    for i, m in enumerate(methods):
        vals = [attr.loc[m, "axis"], attr.loc[m, "oblique"]]
        ax[0].bar(xpos + (i - 2) * w, vals, w, color=col[m],
                  label=PRETTY[m].replace(r"\textsc{Wand}", "WAND").replace(" (post-hoc)", "").replace(" (native)", ""))
    ax[0].set_xticks(xpos); ax[0].set_xticklabels(["axis-aligned", "oblique (corr.)"])
    ax[0].set_ylabel("attribution-AUC"); ax[0].set_ylim(0.45, 1.02)
    ax[0].set_title("(a) ground-truth recovery by regime", fontsize=11)
    ax[0].legend(fontsize=9, ncol=2, loc="lower left")
    ax[0].grid(alpha=0.3, axis="y")

    # (b) faithfulness vs query cost
    for m in ORDER:
        if m not in g2.index:
            continue
        q = max(g2.loc[m, "q2"], 0.5)
        ax[1].scatter(q, g2.loc[m, "faith"], s=70, color=col[m], zorder=3,
                      edgecolor="k", linewidth=0.5)
        ax[1].annotate(PRETTY[m].replace(r"\textsc{Wand}-", "").replace(r"\textsc{Wand}", "WAND").replace(" (post-hoc)", "").replace(" (native)", ""),
                       (q, g2.loc[m, "faith"]), fontsize=9, xytext=(3, 3),
                       textcoords="offset points")
    ax[1].set_xscale("log"); ax[1].set_xlabel("detector queries per explanation")
    ax[1].set_ylabel("faithfulness (mean)")
    ax[1].set_title(f"(b) faithfulness vs cost ({nd} datasets)", fontsize=11)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "xai_scaling.pdf", bbox_inches="tight")
    print("wrote xai_scaling.pdf")


if __name__ == "__main__":
    main()
