"""R4-D4: node2vec_umap multi-seed on PBMC, ca_astroph, PubMed.

Extends R3 single-seed node2vec results to N=5 seeds {42-46}.
node2vec workers=8 per process; ProcessPool(max_workers=5) so
5 * 8 = 40 threads on zjl's 64 cores. OMP_NUM_THREADS=1 assumed
in the environment.

Writes per-cell parquets to output/tables/cells_baselines/ and
embeddings to output/embeddings_baselines/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS_UW = ("pubmed",)
DATASETS_UWFALSE = ("pbmc", "ca_astroph")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def _cell_done(out_root: Path, ds: str, seed: int) -> bool:
    p = out_root / "tables" / "cells_baselines" / f"{ds}_node2vec_umap_seed{seed}.parquet"
    return p.exists()


def _worker(ds: str, seed: int, out_root: str, graph_only: bool) -> dict:
    import numpy as np
    import scipy.sparse as sp
    from node2vec import Node2Vec
    import umap
    from lens.data import load_dataset
    from lens.init import pca_init
    from lens.metrics import compute_metrics

    os.environ["OMP_NUM_THREADS"] = "1"
    out = Path(out_root)
    (out / "tables" / "cells_baselines").mkdir(parents=True, exist_ok=True)
    (out / "embeddings_baselines").mkdir(parents=True, exist_ok=True)

    adj, features, labels = load_dataset(ds)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)

    import networkx as nx
    G = nx.from_scipy_sparse_array(adj)

    t0 = time.perf_counter()
    nv = Node2Vec(G, dimensions=64, walk_length=80, num_walks=10, workers=8,
                  quiet=True, seed=seed)
    model = nv.fit(window=10, min_count=1, batch_words=4)
    n = adj.shape[0]
    X_nv = np.array([model.wv[str(i)] for i in range(n)], dtype=np.float32)

    reducer = umap.UMAP(n_components=2, init=Y0, random_state=seed,
                        n_epochs=1000, n_jobs=1)
    Y = reducer.fit_transform(X_nv)
    runtime_s = time.perf_counter() - t0

    feat_for_metrics = features if features is not None else None
    metrics = compute_metrics(features=feat_for_metrics, adj=adj, Y=Y,
                              labels=labels, max_n=5000, seed=seed)

    row = {
        "dataset": ds, "method": "node2vec_umap", "seed": int(seed),
        "n_seeds": 1, "runtime_s": runtime_s,
        "label_trustworthiness": metrics.get("label_trustworthiness", float("nan")),
        "label_continuity": metrics.get("label_continuity", float("nan")),
        "trustworthiness": metrics.get("trustworthiness", float("nan")),
        "continuity": metrics.get("continuity", float("nan")),
    }
    cell_path = out / "tables" / "cells_baselines" / f"{ds}_node2vec_umap_seed{seed}.parquet"
    pd.DataFrame([row]).to_parquet(cell_path, index=False)
    emb_path = out / "embeddings_baselines" / f"{ds}_node2vec_umap_seed{seed}.npy"
    np.save(str(emb_path), Y)
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["pbmc", "ca_astroph", "pubmed"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--max-workers", type=int, default=5)
    p.add_argument("--out-root", default="output")
    args = p.parse_args()

    out_root = Path(args.out_root)
    cells = []
    for ds in args.datasets:
        for seed in args.seeds:
            if _cell_done(out_root, ds, seed):
                print(f"[D4] skip {ds} seed={seed} (done)", flush=True)
                continue
            graph_only = ds in ("pbmc", "ca_astroph")
            cells.append((ds, seed, graph_only))

    if not cells:
        print("[D4] nothing to do", flush=True)
        return 0

    print(f"[D4] {len(cells)} cells on {args.max_workers} workers", flush=True)
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(_worker, ds, seed, args.out_root, go): (ds, seed)
            for ds, seed, go in cells
        }
        for fut in as_completed(futs):
            ds, seed = futs[fut]
            try:
                row = fut.result()
                done += 1
                print(f"[D4] [{done}/{len(cells)}] {ds} seed={seed} "
                      f"runtime={row['runtime_s']:.0f}s "
                      f"LT={row['label_trustworthiness']:.3f} "
                      f"elapsed={(time.perf_counter()-t0)/60:.1f}m",
                      flush=True)
            except Exception as e:
                print(f"[D4] FAIL {ds} seed={seed}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
