"""R4-H: publication-ready figures H1 (Pareto), H2 (qualitative 2x3), H3 (cores scaling).

H1 pareto_quality_runtime.pdf  -- quality vs runtime scatter with Pareto frontier
H2 qualitative_3method_2dataset.pdf  -- Cora + PubMed x {pysgtsnepi, UMAP, node2vec+UMAP}
H3 cores_scaling.pdf  -- pysgtsnepi runtime vs n_workers on Cora + ca_astroph (run separate)

Usage (after D/E cells complete):
  uv run python scripts/render_r4h_figures.py --figures h1 h2
  uv run python scripts/render_r4h_figures.py --figures h3 --max-workers 1,2,4,8,16,32

H3 runs new compute (scaling experiment) and renders inline.
"""
from __future__ import annotations
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS_ORDER = ["cora", "citeseer", "pubmed", "mnist_knn", "pbmc", "ca_astroph",
                  "coauthor_cs", "ogbn_arxiv"]
METHOD_COLORS = {
    "pysgtsnepi": "#e41a1c",
    "umap": "#377eb8",
    "opentsne": "#4daf4a",
    "phate": "#984ea3",
    "node2vec_umap": "#ff7f00",
    "drgraph": "#a65628",
}
METHOD_LABELS = {
    "pysgtsnepi": r"$\lambda$-Lens (\ours)",
    "umap": "UMAP",
    "opentsne": "openTSNE",
    "phate": "PHATE",
    "node2vec_umap": "node2vec+UMAP",
}
DS_MARKERS = {
    "cora": "o", "citeseer": "s", "pubmed": "^", "mnist_knn": "D",
    "pbmc": "v", "ca_astroph": "P", "coauthor_cs": "X", "ogbn_arxiv": "*",
}


def _load_all_agg(tables: Path):
    rows = []
    for ds in DATASETS_ORDER:
        p = tables / f"{ds}_comparison_agg.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["dataset"] = ds
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def render_h1(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    pareto_pts = []

    for _, row in df.iterrows():
        method = row.get("method", "?")
        ds = row.get("dataset", "?")
        rt = row.get("runtime_s_mean", float("nan"))
        # use label_T if available, else T
        quality = row.get("label_trustworthiness_mean", float("nan"))
        if np.isnan(quality):
            quality = row.get("trustworthiness_mean", float("nan"))
        if np.isnan(rt) or np.isnan(quality):
            continue

        color = METHOD_COLORS.get(method, "#888888")
        marker = DS_MARKERS.get(ds, "o")
        size = 60 if method == "pysgtsnepi" else 30
        ax.scatter(np.log10(rt + 1e-3), quality, c=color, marker=marker,
                   s=size, alpha=0.85, linewidths=0.4, edgecolors="white",
                   zorder=3 if method == "pysgtsnepi" else 2)
        pareto_pts.append((np.log10(rt + 1e-3), quality, method, ds))

    # Pareto frontier (minimize runtime, maximize quality)
    if pareto_pts:
        pareto_pts_sorted = sorted(pareto_pts, key=lambda x: x[0])
        frontier = []
        best_q = -1.0
        for pt in pareto_pts_sorted:
            if pt[1] > best_q:
                best_q = pt[1]
                frontier.append(pt)
        if frontier:
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            ax.step(fx, fy, where="post", color="#999999", linewidth=0.8,
                    linestyle="--", zorder=1, label="Pareto frontier")

    # legend by method
    for method, color in METHOD_COLORS.items():
        if method in df.get("method", pd.Series()).values:
            ax.scatter([], [], c=color, s=30, label=METHOD_LABELS.get(method, method))

    ax.set_xlabel(r"$\log_{10}$(runtime + 1\,ms) [s]", fontsize=8)
    ax.set_ylabel("Label-$T$ (or $T$ if unlabeled)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, markerscale=0.8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "pareto_quality_runtime.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "pareto_quality_runtime.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[H1] wrote {out_dir}/pareto_quality_runtime.pdf", flush=True)


def render_h2(tables: Path, emb_baselines: Path, out_dir: Path):
    methods = ["pysgtsnepi", "umap", "node2vec_umap"]
    datasets = ["cora", "pubmed"]
    method_labels = {
        "pysgtsnepi": r"\ours (auto-$\lambda$)",
        "umap": "UMAP",
        "node2vec_umap": "node2vec+UMAP",
    }

    from lens.data import load_dataset
    fig, axes = plt.subplots(len(datasets), len(methods),
                             figsize=(6.5, 4.5))

    for row_idx, ds in enumerate(datasets):
        _, _, labels = load_dataset(ds)
        n_cls = int(labels.max()) + 1 if labels is not None else 1
        cmap = plt.cm.tab10 if n_cls <= 10 else plt.cm.tab20

        # find median-label_T seed per (method, ds)
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx][col_idx]
            # find available seed parquets
            best_seed = 42
            best_lt = -1.0
            for seed in range(42, 52):
                p = tables / "cells_baselines" / f"{ds}_{method}_seed{seed}.parquet"
                if p.exists():
                    df = pd.read_parquet(p)
                    lt = float(df["label_trustworthiness"].iloc[0]) if not df.empty else float("nan")
                    if not np.isnan(lt) and lt > best_lt:
                        best_lt, best_seed = lt, seed

            emb_p = emb_baselines / f"{ds}_{method}_seed{best_seed}.npy"
            if not emb_p.exists():
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
            else:
                Y = np.load(str(emb_p))
                c = labels if labels is not None else np.zeros(Y.shape[0])
                ax.scatter(Y[:, 0], Y[:, 1], s=0.5, c=c, cmap=cmap,
                           vmin=0, vmax=n_cls - 1, alpha=0.7, linewidths=0)
                if best_lt > 0:
                    ax.text(0.97, 0.97, f"LT={best_lt:.2f}", transform=ax.transAxes,
                            fontsize=6, ha="right", va="top", family="monospace",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="black", lw=0.4, alpha=0.85))

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if row_idx == 0:
                ax.set_title(method_labels.get(method, method), fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(ds, fontsize=8)

    fig.tight_layout(pad=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "qualitative_3method_2dataset.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "qualitative_3method_2dataset.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[H2] wrote {out_dir}/qualitative_3method_2dataset.pdf", flush=True)


def render_h3(workers_list: list[int], out_dir: Path, out_root: str):
    from lens.data import load_dataset
    from lens.init import pca_init

    datasets = ["cora", "ca_astroph"]
    results: dict[str, list[float]] = {ds: [] for ds in datasets}

    for ds in datasets:
        adj, _, _ = load_dataset(ds)
        Y0 = pca_init(adj, d=2, scale=1e-4, random_state=42)
        # warm-up single run
        from pysgtsnepi import sgtsnepi
        _ = sgtsnepi(adj, d=2, lambda_=20.0, random_state=42, Y0=Y0)

        for nw in workers_list:
            cells = [(42 + i,) for i in range(nw)]
            t0 = time.perf_counter()
            with ProcessPoolExecutor(max_workers=nw) as ex:
                futs = [ex.submit(_scaling_cell, ds, 20.0, 42 + i, out_root)
                        for i in range(nw)]
                for fut in as_completed(futs):
                    fut.result()
            wall = time.perf_counter() - t0
            # report per-cell throughput as cells/s, plot as wall/cell normalized
            results[ds].append(wall / nw)
            print(f"[H3] {ds} nw={nw} wall={wall:.1f}s per-cell={wall/nw:.1f}s", flush=True)

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    for ds, color in [("cora", "#e41a1c"), ("ca_astroph", "#377eb8")]:
        vals = results[ds]
        normalized = [vals[0] / v for v in vals]
        ax.plot(workers_list, normalized, "o-", color=color, markersize=4,
                linewidth=1.2, label=ds)
    ax.plot(workers_list, workers_list, "k--", linewidth=0.8, alpha=0.4, label="ideal")
    ax.set_xlabel("Workers", fontsize=8)
    ax.set_ylabel("Speedup (relative to 1 worker)", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "cores_scaling.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "cores_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[H3] wrote {out_dir}/cores_scaling.pdf", flush=True)


def _scaling_cell(ds: str, lam: float, seed: int, out_root: str) -> float:
    os.environ["OMP_NUM_THREADS"] = "1"
    from lens.data import load_dataset
    from lens.init import pca_init
    from pysgtsnepi import sgtsnepi
    adj, _, _ = load_dataset(ds)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    t0 = time.perf_counter()
    _ = sgtsnepi(adj, d=2, lambda_=lam, random_state=seed, Y0=Y0)
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", nargs="+", choices=["h1", "h2", "h3"],
                    default=["h1", "h2"])
    ap.add_argument("--max-workers", default="1,2,4,8,16,32",
                    help="Comma-separated worker counts for H3")
    ap.add_argument("--out-root", default="output")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    tables = out_root / "tables"
    emb_baselines = out_root / "embeddings_baselines"
    figures = out_root / "figures"

    if "h1" in args.figures:
        df = _load_all_agg(tables)
        if df.empty:
            print("[H1] no agg parquets found, skipping", flush=True)
        else:
            render_h1(df, figures)

    if "h2" in args.figures:
        render_h2(tables, emb_baselines, figures)

    if "h3" in args.figures:
        workers_list = [int(x) for x in args.max_workers.split(",")]
        render_h3(workers_list, figures, args.out_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
