"""R6-D1: 3-row teaser (Cora/PBMC/MNIST × λ ∈ {1,5,20,80}) PCA-init seed=42.

Reads N=5 mean LT&C from output/tables/teaser_lens_inset_means.json (already
computed by aggregate_insets.py). Renders to teaser_three_regimes.{pdf,png}.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lens.data import load_dataset

LAMBDAS = (1.0, 5.0, 20.0, 80.0)
AUTO_LAMBDA = {"cora": 20.0, "pbmc": 5.0, "mnist_knn": 1.0}  # corrected R5-A1
ROW_DATASETS = ("cora", "pbmc", "mnist_knn")
ROW_LABELS = {
    "cora":      "Cora\n(CV(d) 1.34)",
    "pbmc":      "PBMC\n(CV(d) 0.75)",
    "mnist_knn": "MNIST-kNN\n(CV(d) 0.30)",
}
CMAP_LARGE = plt.cm.tab10
CMAP_SMALL = plt.cm.tab20


def _emb_path(ds: str, lam: float) -> Path:
    if ds == "pbmc":
        return Path(f"output/embeddings/{ds}_lam{lam}_seed42_init=pca_uw=False.npy")
    return Path(f"output/embeddings/{ds}_lam{lam}_seed42_init=pca.npy")


def _load_labels(ds: str):
    if ds == "pbmc":
        labels_path = Path("data/processed/pbmc/labels.npy")
        return np.load(labels_path) if labels_path.exists() else None
    _, _, labels = load_dataset(ds)
    return labels


def _draw(ax, Y, c, lam, lt_lc, is_auto, palette_n):
    cmap = CMAP_LARGE if palette_n <= 10 else CMAP_SMALL
    ax.scatter(Y[:, 0], Y[:, 1], s=0.7, c=c, cmap=cmap,
               vmin=0, vmax=palette_n - 1, alpha=0.7, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    txt = f"$\\lambda{{=}}{int(lam)}$"
    if is_auto:
        txt += "$\\star$"
    ax.text(0.5, 1.02, txt, transform=ax.transAxes,
            fontsize=9, ha="center", va="bottom")
    if lt_lc is not None:
        lt, lc = lt_lc
        ax.text(0.97, 0.97, f"LT&C\n{lt:.2f}/{lc:.2f}", transform=ax.transAxes,
                fontsize=6, ha="right", va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="black", lw=0.4, alpha=0.85))


def main() -> int:
    means_path = Path("output/tables/teaser_lens_inset_means.json")
    means = json.load(open(means_path)) if means_path.exists() else {}

    fig, axes = plt.subplots(3, 4, figsize=(8.0, 6.2))
    for row_idx, ds in enumerate(ROW_DATASETS):
        labels = _load_labels(ds)
        palette_n = int(labels.max()) + 1 if labels is not None else 1
        plot_c = np.where(labels < 0, palette_n, labels) if labels is not None else None
        palette_n_for_cmap = palette_n + (1 if (labels is not None and (labels < 0).any()) else 0)

        for col_idx, lam in enumerate(LAMBDAS):
            ax = axes[row_idx][col_idx]
            emb_p = _emb_path(ds, lam)
            if not emb_p.exists():
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            Y = np.load(emb_p)
            v = means.get(ds, {}).get(str(lam), {}) or {}
            lt = v.get("mean_label_T", float("nan"))
            lc = v.get("mean_label_C", float("nan"))
            lt_lc = (lt, lc) if not (np.isnan(lt) or np.isnan(lc)) else None
            _draw(ax, Y, plot_c if plot_c is not None else 0,
                  lam, lt_lc,
                  is_auto=(lam == AUTO_LAMBDA.get(ds)),
                  palette_n=palette_n_for_cmap)

    # row labels on left side
    for row_idx, ds in enumerate(ROW_DATASETS):
        # y-position centered on this row of axes
        axes[row_idx][0].set_ylabel(ROW_LABELS[ds], fontsize=9,
                                    rotation=0, labelpad=40, ha="center", va="center")

    fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.02,
                        wspace=0.06, hspace=0.18)
    out = Path("output/figures/teaser_three_regimes")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[D1] wrote {out.with_suffix('.pdf')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
