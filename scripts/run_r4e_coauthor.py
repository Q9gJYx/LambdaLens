"""R4-E2/E3: full pipeline (E1 + E2 + E3) for Coauthor-CS and Coauthor-Physics.

Coauthor-CS: n=18,333, 15 classes, 6,805-d features (Shchur 2018 PitfallsGNN).
Coauthor-Physics: n=34,493, 5 classes (stretch dataset).

For each dataset runs:
  E1 16-pt lambda grid at seed=42, PCA-init -> lambda_grid.parquet
  E2 auto-lambda (argmax over {1,5,20,50} of label_T&C) -> auto_lambda_summary update
  E3 N=5 seeds: pysgtsnepi + UMAP + openTSNE + PHATE + node2vec_umap
  -> {ds}_comparison.parquet + {ds}_comparison_agg.parquet
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

LAMBDAS_16PT = [0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80, 100, 150]
AUTO_LAMBDA_PROBE = [1.0, 5.0, 20.0, 50.0]
E3_SEEDS = [42, 43, 44, 45, 46]


def _e1_worker(ds: str, lam: float, seed: int, out_root: str) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    from lens.data import load_dataset
    from lens.run import run_one_cell
    adj, features, labels = load_dataset(ds)
    return run_one_cell(adj, features, labels, lam, seed, ds, out_root, init="pca")


def run_e1(ds: str, out_root: Path, max_workers: int) -> pd.DataFrame:
    cells_dir = out_root / "tables" / "cells"
    done = set()
    for p in cells_dir.glob(f"{ds}_lam*_seed42_init=pca.parquet"):
        try:
            df = pd.read_parquet(p)
            for r in df.itertuples():
                done.add(float(r.lambda_))
        except Exception:
            pass
    todo = [(ds, lam, 42) for lam in LAMBDAS_16PT if lam not in done]
    if not todo:
        print(f"[E1 {ds}] all {len(LAMBDAS_16PT)} lambda cells done", flush=True)
    else:
        print(f"[E1 {ds}] {len(todo)} cells", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_e1_worker, ds, lam, 42, str(out_root)): lam for _, lam, _ in todo}
            for fut in as_completed(futs):
                lam = futs[fut]
                try:
                    row = fut.result()
                    print(f"[E1 {ds}] lam={lam} LT={row.get('label_trustworthiness', float('nan')):.3f} "
                          f"rt={row['runtime_s']:.1f}s elapsed={(time.perf_counter()-t0)/60:.1f}m",
                          flush=True)
                except Exception as e:
                    print(f"[E1 {ds}] FAIL lam={lam}: {e}", file=sys.stderr, flush=True)

    files = sorted(cells_dir.glob(f"{ds}_lam*_seed42_init=pca.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run_e2(ds: str, grid_df: pd.DataFrame, out_root: Path) -> float | None:
    if grid_df.empty or "lambda_" not in grid_df.columns:
        print(f"[E2 {ds}] grid_df empty or missing lambda_ column; defaulting auto-lambda=20", flush=True)
        return 20.0
    best_lam, best_val = None, -1.0
    for lam in AUTO_LAMBDA_PROBE:
        row = grid_df[grid_df["lambda_"] == lam]
        if row.empty:
            continue
        lt = float(row["label_trustworthiness"].iloc[0])
        lc = float(row["label_continuity"].iloc[0])
        if np.isnan(lt) or np.isnan(lc):
            continue
        val = 2 * lt * lc / (lt + lc) if lt + lc > 0 else 0.0
        if val > best_val:
            best_val, best_lam = val, lam
    print(f"[E2 {ds}] auto-lambda={best_lam} (probe best label_T&C={best_val:.4f})", flush=True)
    return best_lam


def _e3_worker(ds: str, method: str, seed: int, auto_lam: float, out_root: str) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    from lens.data import load_dataset
    from lens.init import pca_init
    from lens.metrics import compute_metrics

    out = Path(out_root)
    (out / "tables" / "cells_baselines").mkdir(parents=True, exist_ok=True)
    (out / "embeddings_baselines").mkdir(parents=True, exist_ok=True)

    adj, features, labels = load_dataset(ds)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    t0 = time.perf_counter()

    if method == "pysgtsnepi":
        from pysgtsnepi import sgtsnepi
        Y = sgtsnepi(adj, d=2, lambda_=auto_lam, random_state=seed, Y0=Y0)
    elif method == "umap":
        import umap as _umap
        reducer = _umap.UMAP(n_components=2, init=Y0, random_state=seed,
                             n_epochs=1000, n_jobs=1)
        Y = reducer.fit_transform(features)
    elif method == "opentsne":
        from openTSNE import TSNE
        tsne = TSNE(perplexity=30, n_iter=1000, initialization=Y0.astype(np.float64),
                    random_state=seed, n_jobs=1)
        Y = tsne.fit(features.astype(np.float64))
    elif method == "phate":
        import phate
        op = phate.PHATE(n_components=2, random_state=seed, n_jobs=1, verbose=False)
        Y = op.fit_transform(features)
    elif method == "node2vec_umap":
        import networkx as nx
        import umap as _umap
        from node2vec import Node2Vec
        G = nx.from_scipy_sparse_array(adj)
        nv = Node2Vec(G, dimensions=64, walk_length=80, num_walks=10,
                      workers=1, quiet=True, seed=seed)
        model = nv.fit(window=10, min_count=1, batch_words=4)
        n = adj.shape[0]
        X_nv = np.array([model.wv[str(i)] for i in range(n)], dtype=np.float32)
        reducer = _umap.UMAP(n_components=2, init=Y0, random_state=seed,
                             n_epochs=1000, n_jobs=1)
        Y = reducer.fit_transform(X_nv)
    else:
        raise ValueError(f"Unknown method: {method}")

    runtime_s = time.perf_counter() - t0
    m = compute_metrics(features=features, adj=adj, Y=np.array(Y, dtype=np.float64),
                        labels=labels, max_n=5000, seed=seed)
    row = {
        "dataset": ds, "method": method, "seed": seed, "runtime_s": runtime_s,
        "lambda_used": auto_lam if method == "pysgtsnepi" else float("nan"),
        "label_trustworthiness": m.get("label_trustworthiness", float("nan")),
        "label_continuity": m.get("label_continuity", float("nan")),
        "trustworthiness": m.get("trustworthiness", float("nan")),
        "continuity": m.get("continuity", float("nan")),
    }
    cell_p = out / "tables" / "cells_baselines" / f"{ds}_{method}_seed{seed}.parquet"
    pd.DataFrame([row]).to_parquet(cell_p, index=False)
    emb_p = out / "embeddings_baselines" / f"{ds}_{method}_seed{seed}.npy"
    np.save(str(emb_p), np.array(Y, dtype=np.float64))
    return row


def run_e3(ds: str, auto_lam: float, out_root: Path, max_workers: int) -> pd.DataFrame:
    methods = ["pysgtsnepi", "umap", "opentsne", "phate", "node2vec_umap"]
    cells = []
    for method in methods:
        for seed in E3_SEEDS:
            p = out_root / "tables" / "cells_baselines" / f"{ds}_{method}_seed{seed}.parquet"
            if not p.exists():
                cells.append((method, seed))

    if not cells:
        print(f"[E3 {ds}] all cells done", flush=True)
    else:
        print(f"[E3 {ds}] {len(cells)} cells on {max_workers} workers", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(_e3_worker, ds, m, s, auto_lam, str(out_root)): (m, s)
                for m, s in cells
            }
            for fut in as_completed(futs):
                m, s = futs[fut]
                try:
                    row = fut.result()
                    print(f"[E3 {ds}] {m} seed={s} LT={row['label_trustworthiness']:.3f} "
                          f"rt={row['runtime_s']:.1f}s elapsed={(time.perf_counter()-t0)/60:.1f}m",
                          flush=True)
                except Exception as e:
                    print(f"[E3 {ds}] FAIL {m} seed={s}: {e}", file=sys.stderr, flush=True)

    # aggregate
    rows = []
    for method in methods:
        for seed in E3_SEEDS:
            p = out_root / "tables" / "cells_baselines" / f"{ds}_{method}_seed{seed}.parquet"
            if p.exists():
                rows.append(pd.read_parquet(p))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out_root / "tables" / f"{ds}_comparison.parquet", index=False)

    # compute agg
    agg_rows = []
    for method in methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        arow = {"method": method, "n_seeds": len(sub)}
        for col in ["label_trustworthiness", "label_continuity",
                    "trustworthiness", "continuity", "runtime_s"]:
            vals = sub[col].dropna()
            arow[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            arow[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else float("nan")
            arow[f"{col}_median"] = float(vals.median()) if len(vals) else float("nan")
        agg_rows.append(arow)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_parquet(out_root / "tables" / f"{ds}_comparison_agg.parquet", index=False)
    return agg_df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["coauthor_cs", "coauthor_physics"])
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--out-root", default="output")
    args = p.parse_args()

    out_root = Path(args.out_root)
    for ds in args.datasets:
        print(f"\n=== {ds} ===", flush=True)
        grid_df = run_e1(ds, out_root, max_workers=args.max_workers)
        auto_lam = run_e2(ds, grid_df, out_root) or 20.0
        agg_df = run_e3(ds, auto_lam, out_root, max_workers=args.max_workers)
        if not agg_df.empty:
            print(f"[{ds}] summary:", flush=True)
            cols = ["method", "label_trustworthiness_mean", "runtime_s_mean"]
            avail = [c for c in cols if c in agg_df.columns]
            print(agg_df[avail].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
