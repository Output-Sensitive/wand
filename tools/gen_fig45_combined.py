"""Combined Fig 4+5 as a single SINGLE-COLUMN 2x2 figure:
  (a) attribution-AUC by regime          (b) faithfulness vs query cost
  (c) detection under heavy tails        (d) explanation quality, heavy tails

Reads the same CSVs as gen_xai_assets.py / exp_e5_heavytail.py.
Output: figures/fig45_combined.pdf  (width ~= \columnwidth, so the
font sizes set here render at ~the same pt in the paper -- no down-scaling).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

PRETTY = {"witness": "WAND-wit.", "gradient": "WAND-grad.",
          "ecod": "ECOD", "shap": "SHAP", "lime": "LIME", "random": "Random"}
ORDER = ["witness", "gradient", "ecod", "shap", "lime", "random"]
COL = {"witness": "#2c7fb8", "gradient": "#41b6c4", "ecod": "#74c476",
       "shap": "#d95f0e", "lime": "#fec44f", "random": "#999999"}


def main():
    e1 = pd.read_csv(RES / "e1_synthetic.csv")
    e2 = pd.read_csv(RES / "e2_faithfulness.csv")
    ht = pd.read_csv(RES / "e5_heavytail.csv")
    nd = e2.dataset.nunique()
    attr = e1.groupby(["method", "kind"])["attr_auc"].mean().unstack()
    g2 = e2.groupby("method").agg(faith=("faithful", "mean"),
                                  q2=("queries", "mean"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6})
    fig, ax = plt.subplots(2, 2, figsize=(3.4, 3.5))

    # (a) attribution-AUC by regime (grouped bars)
    methods = ["witness", "gradient", "ecod", "shap", "lime"]
    xpos = np.arange(2)
    w = 0.16
    for i, m in enumerate(methods):
        vals = [attr.loc[m, "axis"], attr.loc[m, "oblique"]]
        ax[0, 0].bar(xpos + (i - 2) * w, vals, w, color=COL[m], label=PRETTY[m])
    ax[0, 0].set_xticks(xpos)
    ax[0, 0].set_xticklabels(["axis", "oblique"])
    ax[0, 0].set_ylabel("attr-AUC")
    ax[0, 0].set_ylim(0.45, 1.02)
    ax[0, 0].set_title("(a) ground-truth recovery")
    ax[0, 0].grid(alpha=0.3, axis="y")

    # (b) faithfulness vs query cost (scatter); colours match the top legend
    for m in ORDER:
        if m not in g2.index:
            continue
        q = max(g2.loc[m, "q2"], 0.5)
        ax[0, 1].scatter(q, g2.loc[m, "faith"], s=34, color=COL[m], zorder=3,
                         edgecolor="k", linewidth=0.4, label=PRETTY[m])
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_xlabel("queries / expl.")
    ax[0, 1].set_ylabel("faithful.")
    ax[0, 1].set_title("(b) faithfulness vs cost")
    ax[0, 1].grid(alpha=0.3)

    # shared method legend across the top row, above the panels
    handles, labels = ax[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center",
               fontsize=6, columnspacing=0.8, handletextpad=0.3,
               borderpad=0.3, bbox_to_anchor=(0.5, 1.02))

    # (c) detection under heavy tails
    x = np.arange(len(ht))
    labels_t = ["inf" if not np.isfinite(float(v)) else str(int(float(v)))
                for v in ht["df"]]
    for m, c, lab in [("wand", "#2c7fb8", "WAND"),
                      ("iforest", "#41b6c4", "IForest"),
                      ("ecod", "#d95f0e", "ECOD")]:
        ax[1, 0].plot(x, ht[m], marker="o", ms=3, color=c, label=lab)
    ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels(labels_t)
    ax[1, 0].set_xlabel("Student-$t$ d.o.f. (← heavier)")
    ax[1, 0].set_ylabel("ROC-AUC")
    ax[1, 0].set_title("(c) detection, heavy tails")
    ax[1, 0].legend(loc="lower right", fontsize=6, handlelength=1.3,
                    borderpad=0.25, labelspacing=0.25, handletextpad=0.4)
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].invert_xaxis()

    # (d) explanation quality under heavy tails
    ax[1, 1].plot(x, ht["witness_attr_auc"], marker="s", ms=3, color="#2c7fb8")
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(labels_t)
    ax[1, 1].set_xlabel("Student-$t$ d.o.f. (← heavier)")
    ax[1, 1].set_ylabel("attr-AUC")
    ax[1, 1].set_title("(d) explanation, heavy tails")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].invert_xaxis()

    fig.tight_layout(pad=0.3, h_pad=0.7, w_pad=0.8, rect=(0, 0, 1, 0.93))
    out = ROOT / "figures" / "fig45_combined.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}  (datasets={nd})")


if __name__ == "__main__":
    main()
