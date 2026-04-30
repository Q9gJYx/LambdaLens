"""R8: render 12 individual teaser panel PDFs (3 datasets × 4 λ).

Paper-side assembles the 3×4 grid, inset boxes, and ⋆ markers in
TikZ (images/01_teaser.tex). We just produce uniformly-styled scatter
PDFs from pre-existing embeddings.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lens.data import load_dataset

LAMBDAS = (1.0, 5.0, 20.0, 80.0)
POINT_SIZE = {"cora": 6.0, "pbmc": 2.0, "mnist_knn": 0.5}
OUT_DIR = Path("output/figures/teaser_panels")


def _emb_path(ds: str, lam: float) -> Path:
    if ds == "pbmc":
        return Path(f"output/embeddings/{ds}_lam{lam}_seed42_init=pca_uw=False.npy")
    return Path(f"output/embeddings/{ds}_lam{lam}_seed42_init=pca.npy")


def _load_labels(ds: str) -> np.ndarray:
    if ds == "pbmc":
        return np.load("data/processed/pbmc/labels.npy")
    _, _, labels = load_dataset(ds)
    return labels


def _render(ds: str, lam: float, labels: np.ndarray) -> Path:
    Y = np.load(_emb_path(ds, lam))
    has_noise = bool((labels < 0).any())
    shift = 1 if has_noise else 0
    palette_n = int(labels.max()) + 1 + shift
    cmap = plt.cm.tab10 if palette_n <= 10 else plt.cm.tab20
    c = labels + shift

    fig, ax = plt.subplots(figsize=(2.0, 2.0), facecolor="white")
    ax.scatter(Y[:, 0], Y[:, 1], s=POINT_SIZE[ds], c=c, cmap=cmap,
               vmin=0, vmax=palette_n - 1, alpha=0.7, linewidths=0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    out = OUT_DIR / f"{ds}_lambda{int(lam)}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=150,
                format="pdf", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in ("cora", "pbmc", "mnist_knn"):
        labels = _load_labels(ds)
        for lam in LAMBDAS:
            out = _render(ds, lam, labels)
            print(f"wrote {out} ({out.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
