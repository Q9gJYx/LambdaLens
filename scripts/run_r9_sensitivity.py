"""R9-D5: 4-panel sensitivity sweeps for sensitivity_2x2 figure.

Each sweep varies one parameter while holding others at \\ours defaults
(auto-lambda + PCA-init). N=3 seeds {42,43,44}.

Panels:
  perplexity:  {10,20,30,50,80,120}  on Cora + MNIST-kNN  (PBMC graph-only -> N/A)
               openTSNE PerplexityBasedNN affinity -> pysgtsnepi(auto-lambda)
  n_iter:      {500,1000,2000,4000,8000}  on Cora + PBMC + MNIST-kNN
               sgtsnepi(max_iter=...)
  k_kNN:       {5,10,15,30,50}  on MNIST-kNN  (rebuild kNN graph each k)
               sgtsnepi on rebuilt adjacency
  alpha:       {1,5,12,20,50}  on Cora + PBMC + MNIST-kNN
               sgtsnepi(alpha=...)

Outputs (long-form CSVs):
  output/tables/sensitivity_perplexity.csv
  output/tables/sensitivity_niter.csv
  output/tables/sensitivity_k_knn.csv
  output/tables/sensitivity_alpha.csv
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

DATASETS = ["cora", "pbmc", "mnist_knn"]
SEEDS = [42, 43, 44]
AUTO_LAM = {"cora": 20.0, "pbmc": 5.0, "mnist_knn": 1.0}  # R5-A1 corrected

PERPLEXITY_GRID = [10, 20, 30, 50, 80, 120]
NITER_GRID = [500, 1000, 2000, 4000, 8000]
K_KNN_GRID = [5, 10, 15, 30, 50]
ALPHA_GRID = [1, 5, 12, 20, 50]


def _common_imports():
    os.environ["OMP_NUM_THREADS"] = "1"


def _w_perplexity(ds: str, perplexity: int, seed: int) -> dict:
    _common_imports()
    import numpy as _np
    from openTSNE.affinity import PerplexityBasedNN
    from pysgtsnepi import sgtsnepi
    from lens.data import load_dataset
    from lens.init import pca_init
    from lens.metrics import compute_metrics

    adj_default, features, labels = load_dataset(ds)
    if features is None:
        return {"dataset": ds, "sweep": "perplexity", "value": perplexity, "seed": seed,
                "label_T": float("nan"), "label_C": float("nan"), "runtime_s": float("nan"),
                "note": "no_features"}

    aff = PerplexityBasedNN(features.astype(_np.float32),
                            perplexity=int(perplexity),
                            n_jobs=1, random_state=seed)
    P = aff.P  # sparse
    Y0 = pca_init(P, d=2, scale=1e-4, random_state=seed)
    auto_lam = AUTO_LAM.get(ds, 20.0)
    t0 = time.perf_counter()
    Y = sgtsnepi(P, d=2, lambda_=auto_lam, random_state=seed, Y0=Y0)
    runtime_s = time.perf_counter() - t0
    Y_arr = _np.asarray(Y, dtype=_np.float64)
    m = compute_metrics(features=features, adj=adj_default, Y=Y_arr,
                        labels=labels, max_n=5000, seed=seed)
    return {"dataset": ds, "sweep": "perplexity", "value": int(perplexity), "seed": int(seed),
            "label_T": float(m.get("label_trustworthiness", float("nan"))),
            "label_C": float(m.get("label_continuity", float("nan"))),
            "runtime_s": float(runtime_s)}


def _w_niter(ds: str, n_iter: int, seed: int) -> dict:
    _common_imports()
    import numpy as _np
    from pysgtsnepi import sgtsnepi
    from lens.data import load_dataset
    from lens.init import pca_init
    from lens.metrics import compute_metrics

    adj, features, labels = load_dataset(ds)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    auto_lam = AUTO_LAM.get(ds, 20.0)
    uw = ds != "pbmc"
    t0 = time.perf_counter()
    Y = sgtsnepi(adj, d=2, lambda_=auto_lam, random_state=seed, Y0=Y0,
                 max_iter=int(n_iter), unweighted_to_weighted=uw)
    runtime_s = time.perf_counter() - t0
    Y_arr = _np.asarray(Y, dtype=_np.float64)
    m = compute_metrics(features=features, adj=adj, Y=Y_arr,
                        labels=labels, max_n=5000, seed=seed)
    return {"dataset": ds, "sweep": "n_iter", "value": int(n_iter), "seed": int(seed),
            "label_T": float(m.get("label_trustworthiness", float("nan"))),
            "label_C": float(m.get("label_continuity", float("nan"))),
            "runtime_s": float(runtime_s)}


def _w_k_knn(ds: str, k: int, seed: int) -> dict:
    """Rebuild kNN graph at given k on MNIST-kNN raw features, then run sgtsnepi.

    Uses a 10K subsample (matching R4-G's G4 protocol) so wall stays bounded.
    """
    _common_imports()
    import numpy as _np
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors
    from pysgtsnepi import sgtsnepi
    from lens.data import load_dataset
    from lens.init import pca_init
    from lens.metrics import compute_metrics

    adj0, features, labels = load_dataset(ds)
    n = adj0.shape[0]
    subsample = 10000
    rng = _np.random.default_rng(seed)
    idx = rng.choice(n, size=subsample, replace=False)
    feat_sub = features[idx]
    lbl_sub = labels[idx] if labels is not None else None

    nn = NearestNeighbors(n_neighbors=int(k), metric="cosine", n_jobs=1)
    nn.fit(feat_sub)
    _, inds = nn.kneighbors(feat_sub)
    rows = _np.repeat(_np.arange(subsample), int(k))
    cols = inds.ravel()
    adj_k = sp.csr_matrix(
        (_np.ones(len(rows), dtype=_np.float32), (rows, cols)),
        shape=(subsample, subsample),
    )
    adj_k = ((adj_k + adj_k.T) > 0).astype(_np.float32)
    adj_k.setdiag(0)
    adj_k.eliminate_zeros()

    Y0 = pca_init(adj_k, d=2, scale=1e-4, random_state=seed)
    auto_lam = AUTO_LAM.get(ds, 20.0)
    t0 = time.perf_counter()
    Y = sgtsnepi(adj_k, d=2, lambda_=auto_lam, random_state=seed, Y0=Y0)
    runtime_s = time.perf_counter() - t0
    Y_arr = _np.asarray(Y, dtype=_np.float64)
    m = compute_metrics(features=feat_sub, adj=adj_k, Y=Y_arr,
                        labels=lbl_sub, max_n=5000, seed=seed)
    return {"dataset": ds, "sweep": "k_knn", "value": int(k), "seed": int(seed),
            "subsample_n": subsample,
            "label_T": float(m.get("label_trustworthiness", float("nan"))),
            "label_C": float(m.get("label_continuity", float("nan"))),
            "runtime_s": float(runtime_s)}


def _w_alpha(ds: str, alpha: float, seed: int) -> dict:
    _common_imports()
    import numpy as _np
    from pysgtsnepi import sgtsnepi
    from lens.data import load_dataset
    from lens.init import pca_init
    from lens.metrics import compute_metrics

    adj, features, labels = load_dataset(ds)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    auto_lam = AUTO_LAM.get(ds, 20.0)
    uw = ds != "pbmc"
    t0 = time.perf_counter()
    try:
        Y = sgtsnepi(adj, d=2, lambda_=auto_lam, random_state=seed, Y0=Y0,
                     alpha=float(alpha), unweighted_to_weighted=uw)
    except TypeError as e:
        return {"dataset": ds, "sweep": "alpha", "value": float(alpha), "seed": int(seed),
                "label_T": float("nan"), "label_C": float("nan"), "runtime_s": float("nan"),
                "note": f"alpha kwarg not supported: {e}"}
    runtime_s = time.perf_counter() - t0
    Y_arr = _np.asarray(Y, dtype=_np.float64)
    m = compute_metrics(features=features, adj=adj, Y=Y_arr,
                        labels=labels, max_n=5000, seed=seed)
    return {"dataset": ds, "sweep": "alpha", "value": float(alpha), "seed": int(seed),
            "label_T": float(m.get("label_trustworthiness", float("nan"))),
            "label_C": float(m.get("label_continuity", float("nan"))),
            "runtime_s": float(runtime_s)}


def _pool_run(name: str, cells, worker_fn, max_workers, sweep_label):
    rows = []
    print(f"[R9-D5/{name}] {len(cells)} cells", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(worker_fn, *c): c for c in cells}
        done = 0
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                r = fut.result()
                rows.append(r)
                done += 1
                el = (time.perf_counter() - t0) / 60
                print(
                    f"[R9-D5/{name}] [{done:>3}/{len(cells)}] {c}  "
                    f"LT={r.get('label_T', float('nan')):.4f}  rt={r.get('runtime_s', float('nan')):.1f}s  elapsed={el:.1f}m",
                    flush=True,
                )
            except Exception as e:
                print(f"[R9-D5/{name}] FAIL {c}: {e}", file=sys.stderr, flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+",
                    choices=["perplexity", "niter", "k_knn", "alpha"],
                    default=["perplexity", "niter", "k_knn", "alpha"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--out-root", default="output")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    tables = out_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    if "perplexity" in args.sweeps:
        cells = [(ds, p, sd) for ds in args.datasets if ds != "pbmc"
                 for p in PERPLEXITY_GRID for sd in args.seeds]
        rows = _pool_run("perplexity", cells, _w_perplexity, args.max_workers, "perplexity")
        if rows:
            pd.DataFrame(rows).to_csv(tables / "sensitivity_perplexity.csv", index=False)
            print(f"[R9-D5/perplexity] wrote {tables / 'sensitivity_perplexity.csv'}", flush=True)

    if "niter" in args.sweeps:
        cells = [(ds, n, sd) for ds in args.datasets
                 for n in NITER_GRID for sd in args.seeds]
        rows = _pool_run("niter", cells, _w_niter, args.max_workers, "n_iter")
        if rows:
            pd.DataFrame(rows).to_csv(tables / "sensitivity_niter.csv", index=False)
            print(f"[R9-D5/niter] wrote {tables / 'sensitivity_niter.csv'}", flush=True)

    if "k_knn" in args.sweeps:
        # k_kNN well-defined only on MNIST-kNN (Cora citation graph; PBMC ships graph-only)
        ds_list = [d for d in args.datasets if d == "mnist_knn"]
        cells = [(ds, k, sd) for ds in ds_list for k in K_KNN_GRID for sd in args.seeds]
        rows = _pool_run("k_knn", cells, _w_k_knn, args.max_workers, "k_knn")
        if rows:
            pd.DataFrame(rows).to_csv(tables / "sensitivity_k_knn.csv", index=False)
            print(f"[R9-D5/k_knn] wrote {tables / 'sensitivity_k_knn.csv'}", flush=True)

    if "alpha" in args.sweeps:
        cells = [(ds, a, sd) for ds in args.datasets
                 for a in ALPHA_GRID for sd in args.seeds]
        rows = _pool_run("alpha", cells, _w_alpha, args.max_workers, "alpha")
        if rows:
            pd.DataFrame(rows).to_csv(tables / "sensitivity_alpha.csv", index=False)
            print(f"[R9-D5/alpha] wrote {tables / 'sensitivity_alpha.csv'}", flush=True)

    print("[R9-D5] all sweeps done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
