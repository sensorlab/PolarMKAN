"""Figures for the Polar MKAN letter.

    plot_tradeoff_scatter -> tradeoff_scatter_heldout.pdf
    plot_phase_wrap       -> phase_wrap_consistency.pdf

Both write a PDF for LaTeX plus a PNG preview, and return the Matplotlib figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ACCENT = "#c0392b"

# Display name -> canonical key used to join the two result tables.
TRADEOFF_POINTS = {
    "CNN": "cnn",
    "KAN": "kan",
    "MKAN": "mkan",
    "Polar CNN": "polarcnn",
    "Polar KAN": "polarkan",
    "Polar MKAN": "polarmkan",
}

# Per-point annotation offsets, tuned so labels do not overlap the markers.
TRADEOFF_LABEL_OFFSETS = {
    "CNN": ((0, -12), "center"),
    "KAN": ((6, -3), "left"),
    "MKAN": ((-6, -3), "right"),
    "Polar CNN": ((0, -12), "center"),
    "Polar KAN": ((6, 3), "left"),
    "Polar MKAN": ((0, 7), "center"),
}

PHASE_WRAP_LABELS = ["0%", "0--2%", "2--5%", "5--10%", ">10%"]


def _canon(name):
    return str(name).lower().replace(" ", "").replace("_", "").replace("ae", "")


def _parse_mean_ci(v):
    """Parse ``0.572``, ``'0.572 +/- 0.060'`` or ``'0.572 ± 0.060'``.

    Returns the mean and 95% CI half-width, both in percentage points.
    """
    if not isinstance(v, str):
        return float(v) * 100, 0.0

    s = v.strip().replace("±", "+/-")
    if "+/-" in s:
        mean, ci = s.split("+/-", 1)
        return float(mean.strip()) * 100, float(ci.strip()) * 100
    return float(s) * 100, 0.0


def plot_tradeoff_scatter(dci_csv, ued_csv,
                          out_pdf="tradeoff_scatter_heldout.pdf",
                          out_png="tradeoff_scatter_heldout_preview.png",
                          show=True):
    """Interpretability vs. detection trade-off (letter Fig. 5).

    Both axes come from the synthetic benchmark, so the two coordinates of each
    point belong to one coherent experiment:
        y : held-out DCI Disentanglement, with its 95% CI
        x : synthetic UED ROC-AUC, uncompensated
    Real-data detection cost lives separately in the WiSig table.

    Args:
        dci_csv: path to a synthetic_dci_*.csv.
        ued_csv: path to a synthetic_ued_*.csv.
    """
    dci = pd.read_csv(dci_csv)
    d_mean, d_ci = {}, {}
    for _, r in dci.iterrows():
        key = _canon(r["architecture"])
        d_mean[key], d_ci[key] = _parse_mean_ci(r["disentanglement"])

    ued = pd.read_csv(ued_csv)
    if "cfo_compensate" in ued.columns:
        ued = ued[ued["cfo_compensate"].astype(str).str.lower().isin(["false", "0"])]
    auc_col = "roc_auc" if "roc_auc" in ued.columns else "roc_auc_mean"

    auc = {}
    for _, r in ued.iterrows():
        auc[_canon(r["architecture"])], _ = _parse_mean_ci(r[auc_col])

    data = {
        name: (auc[key], d_mean[key], d_ci[key],
               "polar" if key.startswith("polar") else "raw")
        for name, key in TRADEOFF_POINTS.items()
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for name, (x, y, yerr, kind) in data.items():
        marker = "s" if kind == "polar" else "o"
        if name == "Polar MKAN":
            face, edge, size, z = ACCENT, "black", 60, 6
        else:
            face, edge, size, z = "white", "#444", 34, 5

        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=edge, elinewidth=0.9,
                    capsize=2.5, capthick=0.9, zorder=z - 1)
        ax.scatter(x, y, s=size, marker=marker, facecolor=face, edgecolor=edge,
                   linewidth=1.0, zorder=z)

        offset, ha = TRADEOFF_LABEL_OFFSETS[name]
        ax.annotate(name, (x, y), textcoords="offset points", xytext=offset,
                    fontsize=7, ha=ha, color="#222")

    ax.set_xlabel("UED detection (synthetic ROC-AUC, %)", fontsize=8)
    ax.set_ylabel("Disentanglement D (%)", fontsize=8)
    ax.set_xlim(47, 87)
    ax.set_ylim(-2, 70)
    ax.tick_params(labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, linewidth=0.4, alpha=0.35)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#444", markersize=6, label="raw I/Q"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor="#444", markersize=6, label="polar input"),
    ]
    ax.legend(handles=legend, fontsize=6.3, loc="center left",
              bbox_to_anchor=(0.0, 0.66), frameon=False, labelspacing=0.35)

    fig.tight_layout(pad=0.4)
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    if out_png:
        fig.savefig(out_png, dpi=250, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_phase_wrap(means, ci95, labels=None,
                    out_pdf="phase_wrap_consistency.pdf",
                    out_png="phase_wrap_consistency.png",
                    show=True):
    """Sign consistency of the phase features vs. wrap fraction (letter Fig. 4).

    Args:
        means: array (2, n_bins) - mean consistency for F_phi1 and F_phi2.
        ci95:  array (2, n_bins) - 95% CI half-widths.
        labels: x tick labels; defaults to PHASE_WRAP_LABELS.

    `means` and `ci95` come from ``experiments.run_branch_cut_validation``.
    """
    labels = PHASE_WRAP_LABELS if labels is None else labels
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    ax.errorbar(x, means[0], yerr=ci95[0], marker="o", capsize=3, linewidth=1.2,
                label=r"$F_{\phi_1}$")
    ax.errorbar(x, means[1], yerr=ci95[1], marker="s", capsize=3, linewidth=1.2,
                label=r"$F_{\phi_2}$")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Samples crossing branch cut (%)")
    ax.set_ylabel("Sign-consistent response rate")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout(pad=0.4)
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    if out_png:
        fig.savefig(out_png, dpi=250, bbox_inches="tight")
    if show:
        plt.show()
    return fig
