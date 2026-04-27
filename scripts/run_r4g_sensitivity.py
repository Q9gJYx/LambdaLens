"""R4-G: sensitivity analysis sweeps (G1 lambda plot + G2-G6 method-internal).

G2  openTSNE perplexity in {10,30,50,100} on Cora + PubMed
G3  pysgtsnepi n_iter in {100,300,500,1000,2000} on Cora + PubMed
G4  k-in-kNN in {5,10,15,30,50,100} on MNIST-raw features -> kNN graph
G5  pysgtsnepi u (sparsification) in {10,20,30,50,100} on Cora + PubMed
G6  early-exaggeration alpha in {4,8,12,16,24} on Cora (optional)

G1 (lambda sensitivity plot) runs separately from the lambda_grid.parquet
produced by R4-B; this script handles G2-G6 and emits parquets.
G1 figure rendered by render_r4h_figures.py.

Usage:
  uv run python scripts/run_r4g_sensitivity.py --sweeps g2 g3 g4 g5
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

from lens.data import load_dataset
from lens.init import pca_init
from lens.metrics import compute_metrics


def _run_opentsne(features, labels, adj, Y0, perplexity, seed):
    from openTSNE import TSNE
    tsne = TSNE(perplexity=perplexity, n_iter=1000,
                initialization=Y0.astype(np.float64),
                random_state=seed, n_jobs=1)
    return tsne.fit(features.astype(np.float64))


def _run_sgtsne_niter(adj, Y0, lam, n_iter, seed):
    from pysgtsnepi import sgtsnepi
    return sgtsnepi(adj, d=2, lambda_=lam, random_state=seed, Y0=Y0,
                    max_iter=n_iter)


def _run_sgtsne_u(adj, Y0, lam, u, seed):
    from pysgtsnepi import sgtsnepi
    return sgtsnepi(adj, d=2, lambda_=lam, random_state=seed, Y0=Y0, u=u)


def _run_sgtsne_alpha(adj, Y0, lam, alpha, seed):
    from pysgtsnepi import sgtsnepi
    return sgtsnepi(adj, d=2, lambda_=lam, random_state=seed, Y0=Y0,
                    early_exag=alpha)


def _g2_worker(ds, perplexity, seed, out_root):
    os.environ["OMP_NUM_THREADS"] = "1"
    adj, features, labels = load_dataset(ds)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    t0 = time.perf_counter()
    Y = _run_opentsne(features, labels, adj, Y0, perplexity, seed)
    rt = time.perf_counter() - t0
    m = compute_metrics(features=features, adj=adj, Y=np.array(Y, np.float64),
                        labels=labels, max_n=5000, seed=seed)
    return {"dataset": ds, "sweep": "perplexity", "value": perplexity, "seed": seed,
            "runtime_s": rt, **{k: m.get(k, float("nan")) for k in
            ["label_trustworthiness","label_continuity","trustworthiness","continuity"]}}


def _g3_worker(ds, n_iter, seed, auto_lam, out_root):
    os.environ["OMP_NUM_THREADS"] = "1"
    adj, features, labels = load_dataset(ds)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    t0 = time.perf_counter()
    Y = _run_sgtsne_niter(adj, Y0, auto_lam, n_iter, seed)
    rt = time.perf_counter() - t0
    m = compute_metrics(features=features, adj=adj, Y=np.array(Y, np.float64),
                        labels=labels, max_n=5000, seed=seed)
    return {"dataset": ds, "sweep": "n_iter", "value": n_iter, "seed": seed,
            "runtime_s": rt, **{k: m.get(k, float("nan")) for k in
            ["label_trustworthiness","label_continuity","trustworthiness","continuity"]}}


def _g5_worker(ds, u, seed, auto_lam, out_root):
    os.environ["OMP_NUM_THREADS"] = "1"
    adj, features, labels = load_dataset(ds)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    t0 = time.perf_counter()
    try:
        Y = _run_sgtsne_u(adj, Y0, auto_lam, u, seed)
    except TypeError:
        # u kwarg may not be exposed; skip gracefully
        return {"dataset": ds, "sweep": "u", "value": u, "seed": seed,
                "runtime_s": float("nan"), "label_trustworthiness": float("nan"),
                "label_continuity": float("nan"), "trustworthiness": float("nan"),
                "continuity": float("nan"), "note": "u kwarg not supported"}
    rt = time.perf_counter() - t0
    m = compute_metrics(features=features, adj=adj, Y=np.array(Y, np.float64),
                        labels=labels, max_n=5000, seed=seed)
    return {"dataset": ds, "sweep": "u", "value": u, "seed": seed,
            "runtime_s": rt, **{k: m.get(k, float("nan")) for k in
            ["label_trustworthiness","label_continuity","trustworthiness","continuity"]}}


def _g6_worker(ds, alpha, seed, auto_lam, out_root):
    os.environ["OMP_NUM_THREADS"] = "1"
    adj, features, labels = load_dataset(ds)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    t0 = time.perf_counter()
    try:
        Y = _run_sgtsne_alpha(adj, Y0, auto_lam, alpha, seed)
    except TypeError:
        return {"dataset": ds, "sweep": "alpha", "value": alpha, "seed": seed,
                "runtime_s": float("nan"), "label_trustworthiness": float("nan"),
                "label_continuity": float("nan"), "trustworthiness": float("nan"),
                "continuity": float("nan"), "note": "early_exag kwarg not supported"}
    rt = time.perf_counter() - t0
    m = compute_metrics(features=features, adj=adj, Y=np.array(Y, np.float64),
                        labels=labels, max_n=5000, seed=seed)
    return {"dataset": ds, "sweep": "alpha", "value": alpha, "seed": seed,
            "runtime_s": rt, **{k: m.get(k, float("nan")) for k in
            ["label_trustworthiness","label_continuity","trustworthiness","continuity"]}}


def _auto_lam(ds: str, tables: Path) -> float:
    p = tables / "auto_lambda_summary.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        r = df[df["dataset"] == ds]
        if not r.empty:
            v = r["auto_lambda"].iloc[0]
            if v and not (isinstance(v, float) and np.isnan(v)):
                return float(v)
    return 20.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", choices=["g2","g3","g4","g5","g6"],
                    default=["g2","g3","g5"])
    ap.add_argument("--datasets", nargs="+", default=["cora", "pubmed"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--out-root", default="output")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    tables = out_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    if "g2" in args.sweeps:
        perplexities = [10, 30, 50, 100]
        cells = [(ds, pp) for ds in args.datasets for pp in perplexities]
        print(f"[G2] {len(cells)} perplexity cells", flush=True)
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(_g2_worker, ds, pp, args.seed, args.out_root): (ds, pp)
                    for ds, pp in cells}
            for fut in as_completed(futs):
                ds, pp = futs[fut]
                try:
                    r = fut.result()
                    all_rows.append(r)
                    print(f"[G2] {ds} perplexity={pp} LT={r['label_trustworthiness']:.3f}", flush=True)
                except Exception as e:
                    print(f"[G2] FAIL {ds} pp={pp}: {e}", file=sys.stderr, flush=True)
        pd.DataFrame([r for r in all_rows if r.get("sweep")=="perplexity"]).to_parquet(
            tables / "sensitivity_perplexity.parquet", index=False)

    if "g3" in args.sweeps:
        niters = [100, 300, 500, 1000, 2000]
        cells = [(ds, ni) for ds in args.datasets for ni in niters]
        print(f"[G3] {len(cells)} n_iter cells", flush=True)
        rows_g3 = []
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(_g3_worker, ds, ni, args.seed,
                              _auto_lam(ds, tables), args.out_root): (ds, ni)
                    for ds, ni in cells}
            for fut in as_completed(futs):
                ds, ni = futs[fut]
                try:
                    r = fut.result()
                    rows_g3.append(r)
                    all_rows.append(r)
                    print(f"[G3] {ds} n_iter={ni} LT={r['label_trustworthiness']:.3f} rt={r['runtime_s']:.1f}s", flush=True)
                except Exception as e:
                    print(f"[G3] FAIL {ds} ni={ni}: {e}", file=sys.stderr, flush=True)
        pd.DataFrame(rows_g3).to_parquet(tables / "sensitivity_niter.parquet", index=False)

    if "g4" in args.sweeps:
        # k-in-kNN sweep on MNIST raw features (rebuild kNN graph at each k)
        ks = [5, 10, 15, 30, 50, 100]
        print(f"[G4] k-in-kNN sweep: k in {ks}", flush=True)
        from lens.data import load_dataset as _ld
        from sklearn.neighbors import NearestNeighbors
        import scipy.sparse as sp
        rows_g4 = []
        adj0, features, labels = _ld("mnist_knn")
        n = adj0.shape[0]
        subsample = 10000
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n, size=subsample, replace=False)
        feat_sub = features[idx]
        lbl_sub = labels[idx] if labels is not None else None
        for k in ks:
            try:
                nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
                nn.fit(feat_sub)
                dists, inds = nn.kneighbors(feat_sub)
                rows_knn = np.repeat(np.arange(subsample), k)
                cols_knn = inds.ravel()
                adj_k = sp.csr_matrix(
                    (np.ones(len(rows_knn), dtype=np.float32), (rows_knn, cols_knn)),
                    shape=(subsample, subsample)
                )
                adj_k = ((adj_k + adj_k.T) > 0).astype(np.float32)
                adj_k.setdiag(0)

                from pysgtsnepi import sgtsnepi
                Y0_k = pca_init(adj_k, d=2, scale=1e-4, random_state=args.seed)
                al = 20.0
                t0 = time.perf_counter()
                Y = sgtsnepi(adj_k, d=2, lambda_=al, random_state=args.seed, Y0=Y0_k)
                rt = time.perf_counter() - t0
                m = compute_metrics(features=feat_sub, adj=adj_k, Y=np.array(Y, np.float64),
                                    labels=lbl_sub, max_n=5000, seed=args.seed)
                r = {"dataset": "mnist_knn", "sweep": "k_knn", "value": k,
                     "seed": args.seed, "subsample_n": subsample, "runtime_s": rt,
                     **{kk: m.get(kk, float("nan")) for kk in
                     ["label_trustworthiness","label_continuity","trustworthiness","continuity"]}}
                rows_g4.append(r)
                print(f"[G4] k={k} LT={r['label_trustworthiness']:.3f} rt={rt:.1f}s", flush=True)
            except Exception as e:
                print(f"[G4] FAIL k={k}: {e}", file=sys.stderr, flush=True)
        if rows_g4:
            pd.DataFrame(rows_g4).to_parquet(tables / "sensitivity_k_knn.parquet", index=False)
            all_rows.extend(rows_g4)

    if "g5" in args.sweeps:
        us = [10, 20, 30, 50, 100]
        cells = [(ds, u) for ds in args.datasets for u in us]
        print(f"[G5] {len(cells)} u (sparsification) cells", flush=True)
        rows_g5 = []
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {ex.submit(_g5_worker, ds, u, args.seed,
                              _auto_lam(ds, tables), args.out_root): (ds, u)
                    for ds, u in cells}
            for fut in as_completed(futs):
                ds, u = futs[fut]
                try:
                    r = fut.result()
                    rows_g5.append(r)
                    all_rows.append(r)
                    print(f"[G5] {ds} u={u} LT={r['label_trustworthiness']:.3f}", flush=True)
                except Exception as e:
                    print(f"[G5] FAIL {ds} u={u}: {e}", file=sys.stderr, flush=True)
        if rows_g5:
            pd.DataFrame(rows_g5).to_parquet(tables / "sensitivity_u.parquet", index=False)

    if "g6" in args.sweeps:
        alphas = [4, 8, 12, 16, 24]
        rows_g6 = []
        for alpha in alphas:
            try:
                r = _g6_worker("cora", alpha, args.seed,
                               _auto_lam("cora", tables), args.out_root)
                rows_g6.append(r)
                all_rows.append(r)
                print(f"[G6] alpha={alpha} LT={r['label_trustworthiness']:.3f}", flush=True)
            except Exception as e:
                print(f"[G6] FAIL alpha={alpha}: {e}", file=sys.stderr, flush=True)
        if rows_g6:
            pd.DataFrame(rows_g6).to_parquet(tables / "sensitivity_alpha.parquet", index=False)

    print(f"[G] done. {len(all_rows)} total results.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
