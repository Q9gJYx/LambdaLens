"""Derive PBMC cell-type pseudo-labels for the n=8381 fcdimitr split.

The pbmc-graph.tar.gz from fcdimitr/sgtsnepi ships only the stochastic
kNN graph (k=30); no expression vectors and no labels. The canonical
Zheng et al. 2017 PBMC-8k cell-type assignments for this exact n=8381
split are not publicly mirrored.

**Method (λ-independent):** Agglomerative-Ward k=7 on the top-15
non-trivial eigenvectors of the symmetric normalized Laplacian
L_sym = I - D^{-1/2} A D^{-1/2}, row-normalized
(Ng-Jordan-Weiss convention). This is the standard graph-spectral
clustering basis used by Seurat/scanpy pipelines and is independent
of any single λ embedding, so Label-T&C scored against these labels
does not favor any particular λ by construction.

Replaces the earlier HDBSCAN-on-PCA-init-λ=20 derivation, which was
partially circular (labels defined by the embedding being scored)
and produced a single mega-cluster swallowing ~half the cells.

Output: data/processed/pbmc/labels.npy (int8, shape (8381,)).
Provenance: data/processed/pbmc/labels_provenance.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import AgglomerativeClustering

from lens.data import load_pbmc

LABELS_OUT = Path("data/processed/pbmc/labels.npy")
PROV_OUT = Path("data/processed/pbmc/labels_provenance.json")
N_CLUSTERS = 7
N_COMPONENTS = 15


def _laplacian_spectral_embedding(adj, n_components: int) -> np.ndarray:
    A = ((adj + adj.T) / 2.0).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D = sp.diags(deg_inv_sqrt)
    L = sp.eye(A.shape[0], format="csr") - D @ A @ D
    vals, vecs = eigsh(L.astype(np.float64), k=n_components + 1,
                       sigma=0, which="LM")
    order = np.argsort(vals)
    vecs = vecs[:, order][:, 1:n_components + 1]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def _relabel_by_size(y: np.ndarray) -> np.ndarray:
    uniq, counts = np.unique(y, return_counts=True)
    order = uniq[np.argsort(-counts)]
    remap = {old: new for new, old in enumerate(order)}
    out = y.copy()
    for old, new in remap.items():
        out[y == old] = new
    return out


def main() -> int:
    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    adj, _, _ = load_pbmc()
    print(f"[pbmc-labels] loaded graph: n={adj.shape[0]}", flush=True)

    print(f"[pbmc-labels] computing top-{N_COMPONENTS} Laplacian eigvecs ...", flush=True)
    Z = _laplacian_spectral_embedding(adj, n_components=N_COMPONENTS)

    print(f"[pbmc-labels] Agglomerative-Ward k={N_CLUSTERS} on spectral basis", flush=True)
    y = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward").fit_predict(Z)
    y = _relabel_by_size(y).astype(np.int8)
    np.save(LABELS_OUT, y)

    sizes = [int((y == c).sum()) for c in range(N_CLUSTERS)]
    info = {
        "method": "agglomerative-ward-on-laplacian-spectral",
        "n_clusters": N_CLUSTERS,
        "n_spectral_components": N_COMPONENTS,
        "laplacian": "I - D^{-1/2} A D^{-1/2} (symmetric normalized)",
        "row_normalize_eigvecs": True,
        "linkage": "ward",
        "cluster_sizes_descending": sizes,
        "n_noise": 0,
    }
    PROV_OUT.write_text(json.dumps({
        "source_graph": "data/processed/pbmc/pbmc-graph.mtx (fcdimitr/sgtsnepi)",
        "n": int(y.size),
        "dtype": str(y.dtype),
        "info": info,
        "supersedes": (
            "Earlier HDBSCAN-on-PCA-init-λ=20 derivation (mildly circular: "
            "labels from the same embedding being scored). New derivation uses "
            "the input graph's spectral basis only, independent of any λ."
        ),
        "caption_note": (
            "PBMC pseudo-labels: Agglomerative-Ward k=7 on the top-15 "
            "eigenvectors of the symmetric normalized Laplacian of the input "
            "kNN graph (Ng-Jordan-Weiss row-normalization). Canonical Zheng "
            f"et al. 2017 PBMC-8k labels for the n={int(y.size)} fcdimitr "
            "split are not publicly mirrored."
        ),
    }, indent=2))
    print(
        f"[pbmc-labels] wrote {LABELS_OUT} (n={y.size}, dtype={y.dtype}, sizes={sizes})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
