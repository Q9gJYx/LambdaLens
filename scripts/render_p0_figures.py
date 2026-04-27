"""P0-1a + P0-1b: hero teaser (PBMC, 1x4) + regime-contrast fig:lens (Cora+MNIST, 2x4).

Reads cached PCA-init seed=42 embeddings from output/embeddings/. Insets show
Label-T&C from the per-cell parquets (labeled datasets) or recomputed via
lens.metrics.compute_metrics with adj+labels (PBMC graph-only). Auto-lambda
star marker placed on the auto-lambda column (PBMC=20 default; cora/mnist=20
from auto_lambda_summary).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lens.data import load_dataset
from lens.metrics import compute_metrics

PBMC_LAMBDAS = (1.0, 5.0, 20.0, 80.0)
CORA_LAMBDAS = (1.0, 5.0, 20.0, 80.0)
MNIST_LAMBDAS = (1.0, 5.0, 20.0, 80.0)
AUTO_LAMBDA_PBMC = 20.0
AUTO_LAMBDA_CORA = 20.0
AUTO_LAMBDA_MNIST = 20.0

CMAP_LARGE = plt.cm.tab10
CMAP_SMALL = plt.cm.tab20


def _emb_path(ds: str, lam: float, uw_false: bool) -> Path:
    suffix = "_uw=False" if uw_false else ""
    return Path(f"output/embeddings/{ds}_lam{lam}_seed42_init=pca{suffix}.npy")


INSET_MEANS_PATH = Path("output/tables/teaser_lens_inset_means.json")
_inset_means_cache: dict | None = None


def _load_inset_means() -> dict:
    global _inset_means_cache
    if _inset_means_cache is None:
        if INSET_MEANS_PATH.exists():
            with open(INSET_MEANS_PATH) as f:
                _inset_means_cache = json.load(f)
        else:
            _inset_means_cache = {}
    return _inset_means_cache


def _label_tc_from_cell(ds: str, lam: float, uw_false: bool) -> tuple[float, float] | None:
    # Prefer N=5 mean from aggregate_insets.py when available (R4-A pass)
    means = _load_inset_means()
    if ds in means and str(lam) in means[ds]:
        v = means[ds][str(lam)]
        if v and not (v.get("mean_label_T") is None):
            lt = v["mean_label_T"]
            lc = v["mean_label_C"]
            if not (np.isnan(lt) or np.isnan(lc)):
                return float(lt), float(lc)
    # Fall back to single seed=42 cell
    suffix = "_uw=False" if uw_false else ""
    p = Path(f"output/tables/cells/{ds}_lam{lam}_seed42_init=pca{suffix}.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    lt = float(df["label_trustworthiness"].iloc[0])
    lc = float(df["label_continuity"].iloc[0])
    if np.isnan(lt) or np.isnan(lc):
        return None
    return lt, lc


def _label_tc_pbmc(adj, labels, lam: float) -> tuple[float, float]:
    Y = np.load(_emb_path("pbmc", lam, uw_false=True))
    m = compute_metrics(features=None, adj=adj, Y=Y, labels=labels, max_n=5000, seed=42)
    return float(m["label_trustworthiness"]), float(m["label_continuity"])


def _draw_panel(ax, Y, c, lam, lt_lc, is_auto, palette_n):
    cmap = CMAP_LARGE if palette_n <= 10 else CMAP_SMALL
    ax.scatter(Y[:, 0], Y[:, 1], s=1.0, c=c, cmap=cmap,
               vmin=0, vmax=palette_n - 1, alpha=0.7, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    txt = f"$\\lambda{{=}}{int(lam)}$"
    if is_auto:
        txt += "$\\star$"
    ax.text(0.5, 1.02, txt, transform=ax.transAxes,
            fontsize=10, ha="center", va="bottom")
    if lt_lc is not None:
        lt, lc = lt_lc
        ax.text(0.97, 0.97, f"LT&C\n{lt:.2f}/{lc:.2f}", transform=ax.transAxes,
                fontsize=6, ha="right", va="top",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="black", lw=0.4, alpha=0.85))


def _render_pbmc_hero(out_path: Path) -> dict:
    adj, _, labels = load_dataset("pbmc")
    if labels is None:
        labels_path = Path("data/processed/pbmc/labels.npy")
        if not labels_path.exists():
            raise RuntimeError(f"PBMC labels missing: {labels_path}")
        labels = np.load(labels_path)
    palette_n = int(labels.max()) + 1
    plot_c = np.where(labels < 0, palette_n, labels)
    palette_n_for_cmap = palette_n + (1 if (labels < 0).any() else 0)

    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.0))
    insets: dict = {}
    for ax, lam in zip(axes, PBMC_LAMBDAS):
        Y = np.load(_emb_path("pbmc", lam, uw_false=True))
        lt_lc = _label_tc_pbmc(adj, labels, lam)
        insets[str(lam)] = lt_lc
        _draw_panel(ax, Y, plot_c, lam, lt_lc,
                    is_auto=(lam == AUTO_LAMBDA_PBMC),
                    palette_n=palette_n_for_cmap)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.02, wspace=0.06)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"insets": {k: list(v) if v else None for k, v in insets.items()}}


def _render_lens_regimes(out_path: Path) -> dict:
    adj_c, _, lbl_c = load_dataset("cora")
    adj_m, _, lbl_m = load_dataset("mnist_knn")
    palette_c = int(lbl_c.max()) + 1
    palette_m = int(lbl_m.max()) + 1

    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.0))
    insets: dict[str, dict[str, list]] = {"cora": {}, "mnist_knn": {}}

    for ax, lam in zip(axes[0], CORA_LAMBDAS):
        Y = np.load(_emb_path("cora", lam, uw_false=False))
        lt_lc = _label_tc_from_cell("cora", lam, uw_false=False)
        insets["cora"][str(lam)] = list(lt_lc) if lt_lc else None
        _draw_panel(ax, Y, lbl_c, lam, lt_lc, is_auto=(lam == AUTO_LAMBDA_CORA),
                    palette_n=palette_c)

    for ax, lam in zip(axes[1], MNIST_LAMBDAS):
        Y = np.load(_emb_path("mnist_knn", lam, uw_false=False))
        lt_lc = _label_tc_from_cell("mnist_knn", lam, uw_false=False)
        insets["mnist_knn"][str(lam)] = list(lt_lc) if lt_lc else None
        _draw_panel(ax, Y, lbl_m, lam, lt_lc, is_auto=(lam == AUTO_LAMBDA_MNIST),
                    palette_n=palette_m)

    fig.text(0.005, 0.74, "Cora", rotation=90, va="center", fontsize=9)
    fig.text(0.005, 0.30, "MNIST-kNN", rotation=90, va="center", fontsize=9)
    fig.subplots_adjust(left=0.025, right=0.995, top=0.92, bottom=0.02, wspace=0.06, hspace=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return insets


def main() -> int:
    teaser_meta = _render_pbmc_hero(Path("output/figures/teaser_pbmc_hero"))
    print(f"[p0-1a] wrote output/figures/teaser_pbmc_hero.pdf  insets={teaser_meta['insets']}", flush=True)
    lens_meta = _render_lens_regimes(Path("output/figures/lens_idiom_regimes"))
    print(f"[p0-1b] wrote output/figures/lens_idiom_regimes.pdf  insets={lens_meta}", flush=True)
    Path("output/meta").mkdir(parents=True, exist_ok=True)
    with open("output/meta/p0_inset_source.json", "w") as f:
        json.dump({
            "source": "single-seed PCA-init seed=42 (Pass A); refresh in Pass C with N=5 mean if available",
            "teaser_pbmc": teaser_meta["insets"],
            "lens_regimes": lens_meta,
        }, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
