"""One-off: run pubmed node2vec_umap single-seed with workers=4 (fast).

The default script forces workers=1 to avoid thread oversubscription in
ProcessPool. Since this is a single cell run alone, we can crank workers up.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from lens.data import load_dataset
from lens.metrics import compute_metrics


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")
    import networkx as nx
    from umap import UMAP
    adj, features, labels = load_dataset("pubmed")
    print(f"[pubmed-n2v] adj {adj.shape}", flush=True)
    t0 = time.perf_counter()
    try:
        from node2vec import Node2Vec
        G = nx.from_scipy_sparse_array(adj)
        n2v = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200,
                       p=1, q=1, workers=4, seed=42, quiet=True)
        model = n2v.fit(window=10, min_count=1, batch_words=4)
        emb = np.array([model.wv[str(i)] for i in range(adj.shape[0])])
        lib = "node2vec"
    except ImportError:
        from karateclub import Node2Vec  # type: ignore
        G = nx.from_scipy_sparse_array(adj)
        model = Node2Vec(walk_number=10, walk_length=30, dimensions=64, workers=4, seed=42)
        model.fit(G)
        emb = model.get_embedding()
        lib = "karateclub"
    Y = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=1000,
             init="pca", random_state=42).fit_transform(emb)
    rt = time.perf_counter() - t0
    metrics = compute_metrics(features=features, adj=adj, Y=Y, labels=labels, max_n=5000, seed=42)
    print(f"[pubmed-n2v] runtime={rt:.1f}s label_T&C=({metrics['label_trustworthiness']:.3f},"
          f"{metrics['label_continuity']:.3f}) T&C=({metrics['trustworthiness']:.3f},"
          f"{metrics['continuity']:.3f})", flush=True)
    np.save(Path("output/embeddings_baselines/pubmed_node2vec_umap_seed42.npy"), Y)
    row = {
        "dataset": "pubmed", "method": "node2vec_umap", "seed": 42,
        "runtime_s": float(rt),
        **metrics,
        "init_strategy": "pca",
        "iteration_count": 1000,
        "info_lib": lib, "info_n2v_dim": 64, "info_n2v_workers": 4,
        "info_umap_init": "pca",
        "info_provenance": "fast_oneoff_workers4",
    }
    cell = Path("output/tables/cells_baselines/pubmed_node2vec_umap_seed42.parquet")
    cell.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(cell, index=False)
    print(f"[pubmed-n2v] wrote {cell}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
