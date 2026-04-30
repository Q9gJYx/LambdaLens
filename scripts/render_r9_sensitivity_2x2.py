"""Render R9-D5 2x2 sensitivity figure.

Reads the four CSVs produced by run_r9_sensitivity.py and assembles a
single 2x2 PGFPlots-equivalent figure (matplotlib .pdf, single-column
width). One curve per dataset (Cora, PBMC, MNIST-kNN), shaded band over
seeds, dashed vertical line at the \\ours default per panel.

Output: output/figures/sensitivity_2x2.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_ORDER = ["cora", "pbmc", "mnist_knn"]
DATASET_LABELS = {"cora": "Cora", "pbmc": "PBMC-8k", "mnist_knn": "MNIST-kNN"}
# tab10 indices to roughly match teaser palette (Cora warm, PBMC blue-ish, MNIST cool)
DATASET_COLORS = {"cora": "#d62728", "pbmc": "#1f77b4", "mnist_knn": "#2ca02c"}

DEFAULTS = {
    "perplexity": 30,    # openTSNE default
    "n_iter": 1000,      # sgtsnepi default
    "k_knn": 15,         # mnist_knn default in load_dataset
    "alpha": 12,         # sgtsnepi default early-exaggeration
}

PANEL_TITLES = {
    "perplexity": "(a) perplexity",
    "n_iter": "(b) n_iter",
    "k_knn": "(c) k_kNN  (MNIST-kNN only)",
    "alpha": r"(d) $\alpha$ (early-exaggeration)",
}

PANEL_XLABELS = {
    "perplexity": "perplexity",
    "n_iter": "n_iter",
    "k_knn": "k_kNN",
    "alpha": r"$\alpha$",
}


def _load(tables: Path, name: str) -> pd.DataFrame:
    p = tables / f"sensitivity_{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _agg(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    sub = df[df["dataset"] == ds].dropna(subset=["label_T"])
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("value")["label_T"].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g.sort_values("value")


def _plot_panel(ax, df: pd.DataFrame, sweep: str, log_x: bool):
    if df.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return
    has_curve = False
    for ds in DATASET_ORDER:
        g = _agg(df, ds)
        if g.empty:
            continue
        x = g["value"].values
        y = g["mean"].values
        s = g["std"].values
        c = DATASET_COLORS[ds]
        ax.plot(x, y, marker="o", markersize=4, lw=1.5,
                color=c, label=DATASET_LABELS[ds])
        ax.fill_between(x, y - s, y + s, color=c, alpha=0.18, linewidth=0)
        has_curve = True
    if not has_curve:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return
    if log_x:
        ax.set_xscale("log")
    default = DEFAULTS.get(sweep)
    if default is not None:
        ax.axvline(default, ls="--", lw=0.8, color="#888888", zorder=0.5)
    ax.set_title(PANEL_TITLES[sweep], fontsize=10)
    ax.set_xlabel(PANEL_XLABELS[sweep], fontsize=9)
    ax.set_ylabel("Label-T&C", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6, zorder=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="output")
    args = ap.parse_args()

    tables = Path(args.out_root) / "tables"
    figures = Path(args.out_root) / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    df_perp = _load(tables, "perplexity")
    df_niter = _load(tables, "niter")
    df_kknn = _load(tables, "k_knn")
    df_alpha = _load(tables, "alpha")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), constrained_layout=True)
    _plot_panel(axes[0, 0], df_perp, "perplexity", log_x=True)
    _plot_panel(axes[0, 1], df_niter, "n_iter", log_x=True)
    _plot_panel(axes[1, 0], df_kknn, "k_knn", log_x=True)
    _plot_panel(axes[1, 1], df_alpha, "alpha", log_x=True)

    handles, labels = [], []
    for ax in axes.ravel():
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in labels:
                handles.append(h)
                labels.append(lbl)
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(handles), fontsize=9,
                   bbox_to_anchor=(0.5, -0.04), frameon=False)

    pdf = figures / "sensitivity_2x2.pdf"
    png = figures / "sensitivity_2x2.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[R9-D5] wrote {pdf}", flush=True)
    print(f"[R9-D5] wrote {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
