"""Compare candidate pseudo-label algorithms for PBMC teaser coloring.

Renders a 5x4 grid: rows = algorithms, cols = λ ∈ {1,5,20,80}. Each row
uses the same label array for all 4 panels (color stability across λ).
Output: output/figures/teaser_panels/_pbmc_label_candidates.{pdf,png}.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import (
    SpectralClustering, KMeans, AgglomerativeClustering,
)

from lens.data import load_pbmc

LAMBDAS = (1.0, 5.0, 20.0, 80.0)
EMB_TPL = "output/embeddings/pbmc_lam{lam}_seed42_init=pca_uw=False.npy"
ANCHOR_LAM = 20.0  # embedding used for embedding-space clusterers
SEED = 42
OUT = Path("output/figures/teaser_panels/_pbmc_label_candidates")


def relabel_consecutive(y: np.ndarray) -> np.ndarray:
    """Map labels to 0..k-1 by descending size; -1 (noise) preserved as -1."""
    y = np.asarray(y, dtype=int).copy()
    pos_mask = y >= 0
    uniq, counts = np.unique(y[pos_mask], return_counts=True)
    order = uniq[np.argsort(-counts)]
    remap = {old: new for new, old in enumerate(order)}
    out = y.copy()
    for old, new in remap.items():
        out[y == old] = new
    return out


def cluster_louvain(adj) -> np.ndarray:
    G = nx.from_scipy_sparse_array(adj)
    comms = nx.community.louvain_communities(G, seed=SEED, resolution=1.0)
    y = np.full(adj.shape[0], -1, dtype=int)
    for ci, nodes in enumerate(comms):
        y[list(nodes)] = ci
    return relabel_consecutive(y)


def cluster_spectral(adj, k: int) -> np.ndarray:
    sc = SpectralClustering(
        n_clusters=k, affinity="precomputed", random_state=SEED,
        assign_labels="kmeans", n_init=10,
    )
    y = sc.fit_predict(adj)
    return relabel_consecutive(y)


def cluster_kmeans_emb(Y: np.ndarray, k: int) -> np.ndarray:
    y = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(Y)
    return relabel_consecutive(y)


def cluster_agglo_emb(Y: np.ndarray, k: int) -> np.ndarray:
    y = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Y)
    return relabel_consecutive(y)


def laplacian_spectral_embedding(adj, n_components: int) -> np.ndarray:
    """Top-(n_components) non-trivial eigenvectors of normalized Laplacian.

    L_sym = I - D^{-1/2} A D^{-1/2}; we use the smallest eigenvalues.
    Equivalent to scanpy's spectral basis for clustering, λ-independent.
    """
    A = (adj + adj.T) / 2.0  # symmetrize the stochastic kNN
    A = A.tocsr()
    A.setdiag(0); A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D = sp.diags(deg_inv_sqrt)
    L = sp.eye(A.shape[0], format="csr") - D @ A @ D
    # smallest n_components+1 eigvals; drop the trivial constant one
    vals, vecs = eigsh(L.astype(np.float64), k=n_components + 1, sigma=0,
                       which="LM")
    order = np.argsort(vals)
    vecs = vecs[:, order][:, 1:n_components + 1]
    # Row-normalize (Ng-Jordan-Weiss convention) to stabilize Ward
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def cluster_agglo_spectral(adj, k: int, n_components: int = 15) -> np.ndarray:
    Z = laplacian_spectral_embedding(adj, n_components)
    y = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    return relabel_consecutive(y)


def main() -> None:
    adj, _, _ = load_pbmc()
    Y_anchor = np.load(EMB_TPL.format(lam=ANCHOR_LAM))

    hdbscan_y = np.load("data/processed/pbmc/labels.npy")

    print("running clusterers ...")
    candidates: list[tuple[str, np.ndarray]] = []
    candidates.append(("HDBSCAN (current)", relabel_consecutive(hdbscan_y)))

    print("  louvain (networkx, graph)")
    candidates.append(("Louvain (graph)", cluster_louvain(adj)))

    print("  spectral k=7 (graph)")
    candidates.append(("Spectral k=7 (graph)", cluster_spectral(adj, 7)))

    print("  agglomerative k=7 (emb λ=20)")
    candidates.append(("Agglomerative k=7 (emb λ=20)", cluster_agglo_emb(Y_anchor, 7)))

    print("  kmeans k=7 (emb λ=20)")
    candidates.append(("KMeans k=7 (emb λ=20)", cluster_kmeans_emb(Y_anchor, 7)))

    print("  agglomerative k=7 (Laplacian spectral, n_comp=15)  ← Path A")
    candidates.append(
        ("Agglo k=7 (Laplacian spec) ★ Path A",
         cluster_agglo_spectral(adj, 7, n_components=15))
    )

    n_rows = len(candidates)
    n_cols = len(LAMBDAS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols + 2.0, 2.0 * n_rows),
        facecolor="white",
    )

    for r, (name, y) in enumerate(candidates):
        has_noise = bool((y < 0).any())
        shift = 1 if has_noise else 0
        palette_n = int(y.max()) + 1 + shift
        cmap = plt.cm.tab10 if palette_n <= 10 else plt.cm.tab20
        c = y + shift
        n_clusters = palette_n - shift
        n_noise = int((y < 0).sum())

        for col, lam in enumerate(LAMBDAS):
            ax = axes[r, col]
            Y = np.load(EMB_TPL.format(lam=lam))
            ax.scatter(Y[:, 0], Y[:, 1], s=2.0, c=c, cmap=cmap,
                       vmin=0, vmax=palette_n - 1,
                       alpha=0.7, linewidths=0)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if r == 0:
                ax.set_title(f"λ={int(lam)}", fontsize=10)

        suffix = f"\nk={n_clusters}" + (f", noise={n_noise}" if n_noise else "")
        axes[r, 0].set_ylabel(name + suffix, fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=10)

    fig.suptitle(
        "PBMC pseudo-label candidates  —  rows: algorithms (color stable across λ)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
