"""R4-J1: ogbn-products scale demo (n=2.4M).

Single seed=42, PCA-init, lambda=20. pysgtsnepi only (UMAP/openTSNE/PHATE/
node2vec are known-prohibitive at this size). Watchdog wall cap inside.
"""
from __future__ import annotations
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def main() -> int:
    print("[J1] loading ogbn-products...", flush=True)
    # OGB has TWO non-interactive issues:
    #   1. stdin prompt for download confirmation (handled by `yes y |` pipe
    #      at the shell level when launched);
    #   2. torch.load(weights_only=True) default in torch >=2.6 rejects OGB's
    #      pickle protocol 4 cache files. Patch torch.load to opt out, same
    #      as cursor's load_ogbn_arxiv pattern in lens.data.
    import torch
    _orig_torch_load = torch.load
    def _torch_load_ogb_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _torch_load_ogb_compat
    import builtins
    _orig_input = builtins.input
    def _auto_yes(prompt=""):
        print(prompt + " [auto: y]", flush=True)
        return "y"
    builtins.input = _auto_yes
    try:
        from ogb.nodeproppred import NodePropPredDataset
        ds = NodePropPredDataset(name="ogbn-products",
                                 root="data/processed/ogbn_products")
    finally:
        builtins.input = _orig_input
        torch.load = _orig_torch_load
    graph, labels = ds[0]
    n = int(graph["num_nodes"])
    edge_index = graph["edge_index"]
    adj = sp.csr_matrix(
        (np.ones(edge_index.shape[1], np.float32),
         (edge_index[0], edge_index[1])),
        shape=(n, n),
    )
    adj = ((adj + adj.T) > 0).astype(np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()
    labels = np.asarray(labels, dtype=int).ravel()
    print(f"[J1] n={n} m={adj.nnz//2} loaded", flush=True)

    from lens.init import pca_init
    from pysgtsnepi import sgtsnepi

    print("[J1] computing PCA-init Y0 ...", flush=True)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=42)

    print("[J1] running pysgtsnepi auto-lambda=20 seed=42 PCA-init...", flush=True)
    t0 = time.perf_counter()
    Y = sgtsnepi(adj, d=2, lambda_=20.0, random_state=42, Y0=Y0)
    rt = time.perf_counter() - t0
    print(f"[J1] DONE in {rt:.1f}s ({rt/60:.1f} min)", flush=True)

    Path("output/embeddings_baselines").mkdir(parents=True, exist_ok=True)
    np.save("output/embeddings_baselines/ogbn_products_pysgtsnepi_seed42.npy",
            np.asarray(Y, np.float64))
    Path("output/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "dataset": "ogbn_products", "n": n, "m": int(adj.nnz // 2),
        "method": "pysgtsnepi", "seed": 42, "lambda_": 20.0,
        "runtime_s": rt,
    }]).to_parquet("output/tables/ogbn_products_scale.parquet", index=False)
    print("[J1] wrote ogbn_products_scale.parquet", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
