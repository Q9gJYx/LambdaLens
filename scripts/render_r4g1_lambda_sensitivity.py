"""R4-G1: lambda-sensitivity overlay plot.

Reads PCA-init seed=42 cells across the 16-pt lambda grid for each dataset,
plots Label-T (or T for unlabeled) vs log10(lambda). Color by CV(d). Marks
the auto-lambda per dataset. Highlights lambda<1 region with shaded band.

Output: output/figures/lambda_sensitivity.pdf
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAMBDAS_16PT = (0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80, 100, 150)
DATASETS = {
    "cora":       {"uw": True,  "labeled": True},
    "citeseer":   {"uw": True,  "labeled": True},
    "mnist_knn":  {"uw": True,  "labeled": True},
    "pubmed":     {"uw": True,  "labeled": True},
    "pbmc":       {"uw": False, "labeled": True,  "graph_only": True},
    "ca_astroph": {"uw": True,  "labeled": False, "graph_only": True},
    "ogbn_arxiv": {"uw": True,  "labeled": True},
}


def _read_cells(cells_dir: Path, ds: str, uw: bool) -> dict[float, dict]:
    suf = "" if uw else "_uw=False"
    out: dict[float, dict] = {}
    for p in sorted(cells_dir.glob(f"{ds}_lam*_seed42_init=pca{suf}.parquet")):
        try:
            df = pd.read_parquet(p)
            if df.empty:
                continue
            r = df.iloc[0].to_dict()
            if str(r.get("init", "random")) != "pca":
                continue
            if bool(r.get("unweighted_to_weighted", True)) != uw:
                continue
            out[float(r["lambda_"])] = r
        except Exception:
            pass
    return out


def main() -> int:
    cells_dir = Path("output/tables/cells")
    cv = json.load(open("output/tables/cv_d_summary.json"))
    autolam = pd.read_parquet("output/tables/auto_lambda_summary.parquet")
    autolam_map = {row["dataset"]: row["auto_lambda"] for _, row in autolam.iterrows()}

    # color by CV(d) (cool->warm)
    cv_values = [cv[ds]["cv_d"] for ds in DATASETS if ds in cv]
    norm = plt.Normalize(vmin=min(cv_values), vmax=max(cv_values))
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    # shaded band for lambda<1
    ax.axvspan(np.log10(0.05), 0, alpha=0.08, color="gray", zorder=0)
    ax.text(np.log10(0.3), ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] else 0.7,
            r"$\lambda<1$", fontsize=7, color="gray", ha="center", va="top")

    for ds, cfg in DATASETS.items():
        if ds not in cv:
            continue
        cells = _read_cells(cells_dir, ds, cfg["uw"])
        if not cells:
            continue
        lams = sorted(cells.keys())
        # Use label_T when labeled, else T; for graph-only PBMC use the
        # recomputed values from teaser_lens_inset_means.json if available
        ys = []
        for lam in lams:
            r = cells[lam]
            lt = float(r.get("label_trustworthiness", float("nan")))
            t = float(r.get("trustworthiness", float("nan")))
            v = lt if cfg["labeled"] and not np.isnan(lt) else t
            ys.append(v)
        ys = np.array(ys, dtype=float)
        # filter nan
        valid = ~np.isnan(ys)
        if not valid.any():
            continue
        x = np.log10(np.array(lams)[valid])
        y = ys[valid]
        cv_d = cv[ds]["cv_d"]
        color = cmap(norm(cv_d))
        line, = ax.plot(x, y, "-o", markersize=3, linewidth=1.0,
                         color=color, label=f"{ds} (CV={cv_d:.2f})")
        # auto-lambda marker
        al = autolam_map.get(ds)
        if al is not None and not (isinstance(al, float) and np.isnan(al)):
            ax.axvline(np.log10(al), color=color, linewidth=0.5, alpha=0.5,
                       linestyle=":", zorder=1)

    ax.set_xlabel(r"$\log_{10}(\lambda)$", fontsize=9)
    ax.set_ylabel("Label-$T$ (or $T$ if unlabeled)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6, loc="lower center", ncol=2, columnspacing=0.8,
              labelspacing=0.3, borderpad=0.4)
    fig.tight_layout()
    out_dir = Path("output/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "lambda_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "lambda_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[G1] wrote {out_dir}/lambda_sensitivity.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
