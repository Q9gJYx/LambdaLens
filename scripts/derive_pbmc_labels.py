"""P0-2: derive PBMC cell-type labels.

The pbmc-graph.tar.gz from fcdimitr/sgtsnepi ships only the stochastic kNN
graph (k=30); no expression vectors and no labels. The canonical Zheng
et al. 2017 PBMC-8k cell-type assignments for the n=8381 split are not
publicly mirrored. Per round-3 spec escape valve: "If Zheng-2017 PBMC labels
are not directly available, HDBSCAN-derive at PCA-init lambda=20 is the
fallback. Acknowledge in caption."

We HDBSCAN on the cached deterministic PCA-init lambda=20 SG-t-SNE-Pi
embedding (auto-lambda choice for PBMC); target 8-12 clusters (canonical
PBMC-8k has ~10 cell types). Falls back to KMeans(k=10) if HDBSCAN gives
something pathological.

Output: data/processed/pbmc/labels.npy (int8, shape (8381,)).
Provenance written to: data/processed/pbmc/labels_provenance.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EMB_PATH = Path("output/embeddings/pbmc_lam20.0_seed42_init=pca_uw=False.npy")
LABELS_OUT = Path("data/processed/pbmc/labels.npy")
PROV_OUT = Path("data/processed/pbmc/labels_provenance.json")


def _hdbscan_label(Y: np.ndarray, min_cluster_size: int) -> tuple[np.ndarray, dict]:
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        cluster_selection_method="eom",
    )
    raw = clusterer.fit_predict(Y)
    nz = raw[raw >= 0]
    n_clusters = int(nz.max()) + 1 if nz.size else 0
    n_noise = int((raw == -1).sum())
    info = {
        "method": "hdbscan",
        "min_cluster_size": min_cluster_size,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": round(100 * n_noise / raw.size, 2),
    }
    return raw, info


def _kmeans_fallback(Y: np.ndarray, k: int = 10) -> tuple[np.ndarray, dict]:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    raw = km.fit_predict(Y)
    return raw, {"method": "kmeans_fallback", "k": k, "n_clusters": k, "n_noise": 0}


def main() -> int:
    Y = np.load(EMB_PATH)
    print(f"[p0-2] loaded embedding: {Y.shape}", flush=True)

    labels, info = _hdbscan_label(Y, min_cluster_size=20)
    print(f"[p0-2] hdbscan(mcs=20): clusters={info['n_clusters']} noise={info['noise_pct']}%", flush=True)

    if not (5 <= info["n_clusters"] <= 20):
        for mcs in (50, 30, 100, 75):
            labels, info = _hdbscan_label(Y, min_cluster_size=mcs)
            print(f"[p0-2] hdbscan(mcs={mcs}): clusters={info['n_clusters']} noise={info['noise_pct']}%", flush=True)
            if 5 <= info["n_clusters"] <= 20:
                break

    if not (5 <= info["n_clusters"] <= 20):
        labels, info = _kmeans_fallback(Y, k=10)
        print(f"[p0-2] fell back to kmeans(k=10): clusters={info['n_clusters']}", flush=True)

    if labels.dtype != np.int8:
        max_lbl = int(labels.max()) if labels.size else 0
        if max_lbl < 127:
            labels = labels.astype(np.int8)
        else:
            labels = labels.astype(np.int32)

    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(LABELS_OUT, labels)
    with open(PROV_OUT, "w") as f:
        json.dump({
            "source_embedding": str(EMB_PATH),
            "n": int(labels.size),
            "dtype": str(labels.dtype),
            "info": info,
            "caption_note": (
                f"Cell-type labels derived via {info['method']} clustering on the "
                f"PCA-init SG-t-SNE-Pi lambda=20 embedding "
                f"(seed=42, deterministic given init). Canonical Zheng et al. 2017 "
                f"PBMC-8k labels for the n={int(labels.size)} fcdimitr split are not "
                f"publicly mirrored."
            ),
        }, f, indent=2)
    print(f"[p0-2] wrote {LABELS_OUT} (n={labels.size}, dtype={labels.dtype}, "
          f"unique={len(np.unique(labels))}, noise={int((labels == -1).sum())})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
