"""E3 Round-3: multi-seed baseline comparison with PCA-init + matched budgets + graph-only routing.

Runs N=5 seeds per (method, dataset) pair. Per-(method, seed) embeddings under
output/embeddings_baselines/<ds>_<method>_seed<s>.npy; per-cell parquets under
output/tables/cells_baselines/<ds>_<method>_seed<s>.parquet (resume-safe).
Aggregation into <ds>_comparison.parquet (multi-row) is done via
scripts/merge_e3_results.py after this script runs.

Tricks/treats applied (state in caption):
  T2 PCA-init for ALL stochastic methods (UMAP/openTSNE/PHATE/node2vec head).
  T3 n_epochs=1000 UMAP, n_iter=1000 openTSNE.
  T5 Graph-only inputs: UMAP metric='precomputed' on (1 - sym(adj));
     openTSNE PrecomputedAffinities on adj; PHATE knn_dist='precomputed_affinity'.
  T6 PHATE subsample to 10K for mnist_knn / ca_astroph / ogbn_arxiv.

Compute-prohibitive cells (auto-skipped per --node2vec-single-seed-graphs):
  node2vec_umap on pbmc / ca_astroph runs single-seed (seed=42) only;
  node2vec_umap on mnist_knn / ogbn_arxiv skipped entirely (>1 h single-process).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

GRAPH_ONLY = {"pbmc", "ca_astroph", "ogbn_arxiv"}
NODE2VEC_SINGLE_SEED = {"pbmc", "ca_astroph", "pubmed"}
NODE2VEC_SKIP = {"mnist_knn", "ogbn_arxiv"}
PHATE_SUBSAMPLE_DATASETS = {"mnist_knn", "ca_astroph", "ogbn_arxiv"}
DEFAULT_PHATE_SUBSAMPLE_N = 10000
DEFAULT_UNLABELED_LAMBDA = 10.0


def _pca_init_path(out_root: Path, dataset: str) -> Path:
    return out_root / "meta" / f"pca_init_y0_{dataset}.npy"


def _ensure_pca_init(out_root: Path, dataset: str) -> np.ndarray:
    """Cache pca_init Y0 once per dataset; return loaded array."""
    p = _pca_init_path(out_root, dataset)
    if p.exists():
        return np.load(p)
    from lens.data import load_dataset
    from lens.init import pca_init
    adj, _, _ = load_dataset(dataset)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=42)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, Y0)
    return Y0


def _emb_path(out_root: Path, dataset: str, method: str, seed: int) -> Path:
    return out_root / "embeddings_baselines" / f"{dataset}_{method}_seed{seed}.npy"


def _cell_path(out_root: Path, dataset: str, method: str, seed: int) -> Path:
    return out_root / "tables" / "cells_baselines" / f"{dataset}_{method}_seed{seed}.parquet"


def _run_pysgtsnepi(adj: sp.csr_matrix, lambda_: float, dataset: str, seed: int, Y0: np.ndarray) -> tuple[np.ndarray, float, dict]:
    from pysgtsnepi import sgtsnepi
    uw = dataset != "pbmc"
    t0 = time.perf_counter()
    Y = sgtsnepi(adj, d=2, lambda_=lambda_, random_state=seed, Y0=Y0,
                 unweighted_to_weighted=uw)
    return Y, time.perf_counter() - t0, {"lambda_used": float(lambda_), "init": "pca", "iterations": 1000}


def _run_umap(adj: sp.csr_matrix, features: np.ndarray | None, seed: int, Y0: np.ndarray, dataset: str) -> tuple[np.ndarray, float, dict]:
    from umap import UMAP
    info: dict = {"n_epochs": 1000, "init": "pca"}
    t0 = time.perf_counter()
    if features is None:
        sym = (adj + adj.T) * 0.5
        d = (1.0 - sym.toarray().astype(np.float32))
        np.fill_diagonal(d, 0.0)
        info["input_format"] = "precomputed_distance(1-sym(adj))"
        Y = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=1000,
                 metric="precomputed", random_state=seed, init=Y0).fit_transform(d)
    else:
        info["input_format"] = "features"
        Y = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=1000,
                 random_state=seed, init=Y0).fit_transform(features)
    return Y, time.perf_counter() - t0, info


def _run_opentsne(adj: sp.csr_matrix, features: np.ndarray | None, seed: int, Y0: np.ndarray) -> tuple[np.ndarray, float, dict]:
    from openTSNE import TSNE
    info: dict = {"n_iter": 1000, "init": "pca"}
    t0 = time.perf_counter()
    if features is None:
        from openTSNE.affinity import PrecomputedAffinities
        aff = PrecomputedAffinities(adj.tocsr().astype(np.float64))
        info["input_format"] = "precomputed_affinity(adj)"
        emb = TSNE(n_components=2, n_iter=1000, random_state=seed,
                   initialization=Y0.astype(np.float64), n_jobs=4).fit(affinities=aff)
    else:
        info["input_format"] = "features"
        emb = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=seed,
                   initialization=Y0.astype(np.float64), n_jobs=4).fit(features.astype(np.float64))
    return np.asarray(emb), time.perf_counter() - t0, info


def _run_phate(adj: sp.csr_matrix, features: np.ndarray | None, seed: int, dataset: str, subsample_n: int) -> tuple[np.ndarray, float, dict]:
    import phate
    info: dict = {"init": "phate_internal_pca"}
    t0 = time.perf_counter()
    if features is None:
        # graph-only: pass adj as precomputed affinity
        info["input_format"] = "precomputed_affinity(adj)"
        if dataset in PHATE_SUBSAMPLE_DATASETS and adj.shape[0] > subsample_n:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(adj.shape[0], size=subsample_n, replace=False))
            sub = adj[idx][:, idx].toarray()
            info["subsampled_to"] = int(subsample_n)
            op = phate.PHATE(n_components=2, knn_dist="precomputed_affinity",
                             random_state=seed, n_jobs=-1, knn=15, verbose=0)
            Y = op.fit_transform(sub)
        else:
            op = phate.PHATE(n_components=2, knn_dist="precomputed_affinity",
                             random_state=seed, n_jobs=-1, knn=15, verbose=0)
            Y = op.fit_transform(adj.toarray())
    else:
        info["input_format"] = "features"
        if dataset in PHATE_SUBSAMPLE_DATASETS and features.shape[0] > subsample_n:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(features.shape[0], size=subsample_n, replace=False))
            sub = features[idx]
            info["subsampled_to"] = int(subsample_n)
            op = phate.PHATE(n_components=2, random_state=seed, n_jobs=-1, knn=15, verbose=0)
            Y = op.fit_transform(sub)
        else:
            op = phate.PHATE(n_components=2, random_state=seed, n_jobs=-1, knn=15, verbose=0)
            Y = op.fit_transform(features)
    return Y, time.perf_counter() - t0, info


def _run_node2vec_umap(adj: sp.csr_matrix, seed: int, Y0: np.ndarray) -> tuple[np.ndarray, float, dict]:
    import networkx as nx
    from umap import UMAP
    info: dict = {"n2v_dim": 64, "n_epochs": 1000, "umap_init": "pca", "n2v_workers": 1}
    t0 = time.perf_counter()
    try:
        from node2vec import Node2Vec
        G = nx.from_scipy_sparse_array(adj)
        n2v = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200,
                       p=1, q=1, workers=1, seed=seed, quiet=True)
        model = n2v.fit(window=10, min_count=1, batch_words=4)
        emb = np.array([model.wv[str(i)] for i in range(adj.shape[0])])
        info["lib"] = "node2vec"
    except ImportError:
        from karateclub import Node2Vec  # type: ignore
        G = nx.from_scipy_sparse_array(adj)
        model = Node2Vec(walk_number=10, walk_length=30, dimensions=64, workers=1, seed=seed)
        model.fit(G)
        emb = model.get_embedding()
        info["lib"] = "karateclub"
    Y = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=1000,
             init="pca", random_state=seed).fit_transform(emb)
    return Y, time.perf_counter() - t0, info


def _worker(dataset: str, method: str, seed: int, lam: float, out_root_str: str, phate_subsample_n: int) -> dict:
    warnings.filterwarnings("ignore")
    out_root = Path(out_root_str)
    from lens.data import load_dataset
    from lens.metrics import compute_metrics
    adj, features, labels = load_dataset(dataset)

    Y0_path = _pca_init_path(out_root, dataset)
    Y0 = np.load(Y0_path)

    if method == "pysgtsnepi":
        Y, t, info = _run_pysgtsnepi(adj, lam, dataset, seed, Y0)
    elif method == "umap":
        Y, t, info = _run_umap(adj, features, seed, Y0, dataset)
    elif method == "opentsne":
        Y, t, info = _run_opentsne(adj, features, seed, Y0)
    elif method == "phate":
        Y, t, info = _run_phate(adj, features, seed, dataset, phate_subsample_n)
    elif method == "node2vec_umap":
        Y, t, info = _run_node2vec_umap(adj, seed, Y0)
    else:
        raise ValueError(f"unknown method {method}")

    if "subsampled_to" in info and labels is not None:
        metrics = {"trustworthiness": float("nan"), "continuity": float("nan"),
                   "label_trustworthiness": float("nan"), "label_continuity": float("nan")}
    else:
        metrics = compute_metrics(features=features, adj=adj, Y=Y, labels=labels,
                                  max_n=5000, seed=seed)

    np.save(_emb_path(out_root, dataset, method, seed), Y)
    row = {
        "dataset": dataset, "method": method, "seed": int(seed),
        "runtime_s": float(t),
        **metrics,
        "init_strategy": info.get("init", "default"),
        "iteration_count": info.get("iterations", info.get("n_epochs", info.get("n_iter", -1))),
        **{f"info_{k}": v for k, v in info.items() if k not in ("init", "iterations", "n_epochs", "n_iter")},
    }
    cell = _cell_path(out_root, dataset, method, seed)
    cell.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(cell, index=False)
    return row


def _read_auto_lambdas(out_root: Path) -> dict[str, float]:
    import math
    p = out_root / "tables" / "auto_lambda_summary.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    out: dict[str, float] = {}
    for r in df.itertuples():
        try:
            v = float(r.auto_lambda)
            if math.isfinite(v):
                out[r.dataset] = v
        except (ValueError, TypeError):
            continue
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["pbmc", "cora", "citeseer", "ca_astroph", "mnist_knn"])
    p.add_argument("--methods", nargs="+",
                   default=["pysgtsnepi", "umap", "opentsne", "node2vec_umap", "phate"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--out-root", default="output")
    p.add_argument("--phate-subsample-n", type=int, default=DEFAULT_PHATE_SUBSAMPLE_N)
    p.add_argument("--unlabeled-lambda", type=float, default=DEFAULT_UNLABELED_LAMBDA)
    p.add_argument("--force", action="store_true", help="overwrite cached cells")
    args = p.parse_args()

    out_root = Path(args.out_root)
    auto = _read_auto_lambdas(out_root)
    print(f"[e3] auto-lambdas (labeled): {auto}", flush=True)

    for ds in args.datasets:
        Y0 = _ensure_pca_init(out_root, ds)
        print(f"[e3] {ds}: cached PCA init Y0 shape={Y0.shape}", flush=True)

    cells: list[tuple[str, str, int, float]] = []
    for ds in args.datasets:
        lam = auto.get(ds, args.unlabeled_lambda)
        for method in args.methods:
            if method == "node2vec_umap" and ds in NODE2VEC_SKIP:
                print(f"[e3] SKIP {ds}/node2vec_umap (compute-prohibitive)", flush=True)
                continue
            seeds_for_cell = (42,) if (method == "node2vec_umap" and ds in NODE2VEC_SINGLE_SEED) else tuple(args.seeds)
            for seed in seeds_for_cell:
                cell = _cell_path(out_root, ds, method, seed)
                if cell.exists() and not args.force:
                    continue
                cells.append((ds, method, seed, lam))

    if not cells:
        print("[e3] all cells cached; nothing to run", flush=True)
        return 0

    print(f"[e3] running {len(cells)} cells on {args.max_workers} workers", flush=True)
    t0 = time.perf_counter()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_worker, ds, m, s, lam, args.out_root, args.phate_subsample_n): (ds, m, s, lam)
                for ds, m, s, lam in cells}
        for fut in as_completed(futs):
            ds, m, s, lam = futs[fut]
            try:
                row = fut.result()
                completed += 1
                el = time.perf_counter() - t0
                print(f"[e3] [{completed}/{len(cells)}] {ds}/{m}/seed={s} runtime={row['runtime_s']:.1f}s "
                      f"label_T&C=({row['label_trustworthiness']:.3f},{row['label_continuity']:.3f}) "
                      f"T&C=({row['trustworthiness']:.3f},{row['continuity']:.3f}) elapsed={el:.0f}s",
                      flush=True)
            except Exception as e:
                print(f"[e3] FAILED {ds}/{m}/seed={s}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
