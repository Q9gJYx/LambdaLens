"""R4-D5: T4 hyperparameter sweep on Cora + PubMed (fair-comparison strengthening).

Each baseline's headline hyperparameter swept within its native grid at
seed=42, PCA-init. pysgtsnepi NOT swept (auto-lambda is the contribution).

Sweeps:
  openTSNE: perplexity in {10, 30, 50, 100}
  UMAP:     n_neighbors in {5, 15, 30, 50, 100}
  PHATE:    knn in {5, 15, 30}
  node2vec+UMAP: walk_length in {40, 80, 120}

Output: output/tables/{ds}_t4_sweep.parquet keyed
  (method, dataset, hyperparam, value, label_T, label_C, T, C, runtime_s).
Best per (method, dataset) for supplement/footnote only;
does NOT overwrite main comparison_agg.
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

SWEEP_CONFIGS = {
    "opentsne": ("perplexity", [10, 30, 50, 100]),
    "umap":     ("n_neighbors", [5, 15, 30, 50, 100]),
    "phate":    ("knn", [5, 15, 30]),
    "node2vec_umap": ("walk_length", [40, 80, 120]),
}


def _worker(method: str, ds: str, hyperparam: str, value: int,
            seed: int, out_root: str) -> dict:
    import numpy as np
    import scipy.sparse as sp
    from lens.data import load_dataset
    from lens.init import pca_init
    from lens.metrics import compute_metrics

    os.environ["OMP_NUM_THREADS"] = "1"
    adj, features, labels = load_dataset(ds)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    t0 = time.perf_counter()

    if method == "opentsne":
        from openTSNE import TSNE
        tsne = TSNE(perplexity=value, n_iter=1000, initialization=Y0,
                    random_state=seed, n_jobs=1)
        Y = tsne.fit(features)

    elif method == "umap":
        import umap as _umap
        reducer = _umap.UMAP(n_components=2, n_neighbors=value, init=Y0,
                             random_state=seed, n_epochs=1000, n_jobs=1)
        Y = reducer.fit_transform(features)

    elif method == "phate":
        import phate
        op = phate.PHATE(n_components=2, knn=value, random_state=seed,
                         n_jobs=1, verbose=False)
        Y = op.fit_transform(features)

    elif method == "node2vec_umap":
        import networkx as nx
        import umap as _umap
        from node2vec import Node2Vec
        G = nx.from_scipy_sparse_array(adj)
        nv = Node2Vec(G, dimensions=64, walk_length=value, num_walks=10,
                      workers=1, quiet=True, seed=seed)
        model = nv.fit(window=10, min_count=1, batch_words=4)
        n = adj.shape[0]
        X_nv = np.array([model.wv[str(i)] for i in range(n)], dtype=np.float32)
        reducer = _umap.UMAP(n_components=2, init=Y0, random_state=seed,
                             n_epochs=1000, n_jobs=1)
        Y = reducer.fit_transform(X_nv)

    else:
        raise ValueError(f"Unknown method {method}")

    runtime_s = time.perf_counter() - t0
    m = compute_metrics(features=features, adj=adj, Y=np.array(Y, dtype=np.float64),
                        labels=labels, max_n=5000, seed=seed)
    return {
        "dataset": ds, "method": method, "hyperparam": hyperparam, "value": value,
        "seed": seed, "runtime_s": runtime_s,
        "label_T": m.get("label_trustworthiness", float("nan")),
        "label_C": m.get("label_continuity", float("nan")),
        "T": m.get("trustworthiness", float("nan")),
        "C": m.get("continuity", float("nan")),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cora", "pubmed"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--out-root", default="output")
    args = p.parse_args()

    out_root = Path(args.out_root)
    cells = []
    for ds in args.datasets:
        for method, (hyperparam, values) in SWEEP_CONFIGS.items():
            for value in values:
                cells.append((method, ds, hyperparam, value, args.seed))

    existing_rows: list[dict] = []
    rows_lock_path = out_root / "tables"
    rows_lock_path.mkdir(parents=True, exist_ok=True)

    print(f"[D5] {len(cells)} sweep cells on {args.max_workers} workers", flush=True)
    t0 = time.perf_counter()
    rows = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(_worker, m, ds, hp, v, args.seed, args.out_root): (m, ds, hp, v)
            for m, ds, hp, v, _ in cells
        }
        for fut in as_completed(futs):
            m, ds, hp, v = futs[fut]
            try:
                row = fut.result()
                rows.append(row)
                done += 1
                print(f"[D5] [{done}/{len(cells)}] {m} {ds} {hp}={v} "
                      f"LT={row['label_T']:.3f} rt={row['runtime_s']:.1f}s "
                      f"elapsed={(time.perf_counter()-t0)/60:.1f}m",
                      flush=True)
            except Exception as e:
                print(f"[D5] FAIL {m} {ds} {hp}={v}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)

    if rows:
        df = pd.DataFrame(rows)
        for ds in df["dataset"].unique():
            out_path = out_root / "tables" / f"{ds}_t4_sweep.parquet"
            df[df["dataset"] == ds].to_parquet(out_path, index=False)
            print(f"[D5] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
