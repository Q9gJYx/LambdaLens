"""R4-E1 OGBN-arxiv: dedicated E3 runner with OGBN-specific policy.

The generic ``run_r4e_coauthor.py`` queues N=5 cells for every method, which is
wrong for ogbn_arxiv: PHATE on the full 169K-node graph with N=5 and no
subsample is infeasible, and node2vec_umap with N=5 dominates wall time.

This runner bakes the OGBN policy in:

    pysgtsnepi  : N=5 seeds {42-46} at auto-lambda=20, PCA-init (workers=2)
    UMAP        : N=5 seeds {42-46}; reuses 42/43/44 already on disk (workers=2)
    openTSNE    : N=5 seeds {42-46}                            (workers=5)
    PHATE       : single seed=42, FULL graph, mem+wall guarded (1 spawn proc)
    node2vec    : single seed=42, Node2Vec(workers=8), wall guarded

Stages run sequentially inside one tmux session ``r4e1_ogbn_e3``. Every stage is
idempotent: per-cell parquet + .npy under ``output/{tables/cells_baselines,
embeddings_baselines}`` is the unit of work. A crash mid-stage at most loses
the in-flight cell.

Stage 7 aggregates the cells into ``ogbn_arxiv_comparison.parquet`` and
``ogbn_arxiv_comparison_agg.parquet`` using the same column schema as
Coauthor-CS so ``emit_paper_table.py`` picks it up unchanged.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = "ogbn_arxiv"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
AUTO_LAMBDA_PROBE = (1.0, 5.0, 20.0, 50.0)
PHATE_TIME_LIMIT_S = 45 * 60
PHATE_MEM_LIMIT_GB = 150.0
NODE2VEC_TIME_LIMIT_S = 90 * 60


def _emb_path(out_root: Path, method: str, seed: int) -> Path:
    return out_root / "embeddings_baselines" / f"{DATASET}_{method}_seed{seed}.npy"


def _cell_path(out_root: Path, method: str, seed: int) -> Path:
    return out_root / "tables" / "cells_baselines" / f"{DATASET}_{method}_seed{seed}.parquet"


def _make_row(method: str, seed: int, runtime_s: float, lambda_used: float, metrics: dict) -> dict:
    return {
        "dataset": DATASET,
        "method": method,
        "seed": int(seed),
        "runtime_s": float(runtime_s),
        "lambda_used": float(lambda_used) if lambda_used is not None and not np.isnan(lambda_used) else float("nan"),
        "label_trustworthiness": float(metrics.get("label_trustworthiness", float("nan"))),
        "label_continuity": float(metrics.get("label_continuity", float("nan"))),
        "trustworthiness": float(metrics.get("trustworthiness", float("nan"))),
        "continuity": float(metrics.get("continuity", float("nan"))),
    }


def _atomic_write_parquet(df: pd.DataFrame, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dst)


# ---------- Stage 1: auto-lambda summary -----------------------------------

def _harmonic(lt: float, lc: float) -> float:
    if not (np.isfinite(lt) and np.isfinite(lc)) or (lt + lc) <= 0:
        return float("nan")
    return 2.0 * lt * lc / (lt + lc)


def stage1_auto_lambda(out_root: Path) -> float:
    """Compute auto-lambda + gridsearch on E1 grid; upsert into auto_lambda_summary."""
    grid_p = out_root / "tables" / f"{DATASET}_lambda_grid.parquet"
    if not grid_p.exists():
        raise FileNotFoundError(f"missing E1 grid: {grid_p}")
    grid = pd.read_parquet(grid_p)
    grid = grid.sort_values("lambda_").reset_index(drop=True)

    grid = grid.assign(_harm=[_harmonic(lt, lc)
                              for lt, lc in zip(grid["label_trustworthiness"], grid["label_continuity"])])

    auto_lam = None
    auto_metric = float("nan")
    for lam in AUTO_LAMBDA_PROBE:
        sub = grid[np.isclose(grid["lambda_"], lam)]
        if sub.empty:
            continue
        v = float(sub["_harm"].iloc[0])
        if not np.isnan(v) and (auto_lam is None or v > auto_metric):
            auto_lam, auto_metric = float(lam), v

    valid = grid[~grid["_harm"].isna()]
    if valid.empty:
        raise RuntimeError(f"{DATASET} grid has no finite label-T&C values")
    g_idx = int(valid["_harm"].idxmax())
    grid_lam = float(valid.loc[g_idx, "lambda_"])
    grid_metric = float(valid.loc[g_idx, "_harm"])

    print(f"[S1] {DATASET} auto-lambda={auto_lam} (metric={auto_metric:.4f}) "
          f"gridsearch={grid_lam} (metric={grid_metric:.4f})", flush=True)

    summary_p = out_root / "tables" / "auto_lambda_summary.parquet"
    if summary_p.exists():
        sdf = pd.read_parquet(summary_p)
    else:
        sdf = pd.DataFrame(columns=["dataset", "auto_lambda", "auto_metric",
                                    "gridsearch_lambda", "gridsearch_metric",
                                    "match", "metric_used"])
    sdf = sdf[sdf["dataset"] != DATASET].copy()
    sdf = pd.concat([sdf, pd.DataFrame([{
        "dataset": DATASET,
        "auto_lambda": float(auto_lam) if auto_lam is not None else float("nan"),
        "auto_metric": float(auto_metric),
        "gridsearch_lambda": float(grid_lam),
        "gridsearch_metric": float(grid_metric),
        "match": bool(auto_lam is not None and np.isclose(auto_lam, grid_lam)),
        "metric_used": "label_T&C",
    }])], ignore_index=True)
    _atomic_write_parquet(sdf, summary_p)
    print(f"[S1] upserted {DATASET} into {summary_p}", flush=True)
    return float(auto_lam) if auto_lam is not None else 20.0


# ---------- Workers (run inside ProcessPoolExecutor children) --------------

def _worker_pysgtsnepi(seed: int, auto_lam: float, out_root: str) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    warnings.filterwarnings("ignore")
    out = Path(out_root)
    cell = _cell_path(out, "pysgtsnepi", seed)
    if cell.exists():
        return {"skipped": True, "method": "pysgtsnepi", "seed": int(seed)}
    from lens.data import load_dataset
    from lens.init import pca_init
    from lens.metrics import compute_metrics
    from pysgtsnepi import sgtsnepi

    adj, features, labels = load_dataset(DATASET)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    t0 = time.perf_counter()
    Y = sgtsnepi(adj, d=2, lambda_=auto_lam, random_state=seed, Y0=Y0)
    runtime_s = time.perf_counter() - t0
    Y = np.asarray(Y, dtype=np.float64)
    metrics = compute_metrics(features=features, adj=adj, Y=Y, labels=labels,
                              max_n=5000, seed=seed)
    np.save(_emb_path(out, "pysgtsnepi", seed), Y)
    row = _make_row("pysgtsnepi", seed, runtime_s, float(auto_lam), metrics)
    _atomic_write_parquet(pd.DataFrame([row]), cell)
    return row


def _worker_umap(seed: int, out_root: str) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    warnings.filterwarnings("ignore")
    out = Path(out_root)
    cell = _cell_path(out, "umap", seed)
    if cell.exists():
        return {"skipped": True, "method": "umap", "seed": int(seed)}
    from lens.data import load_dataset
    from lens.init import pca_init
    from lens.metrics import compute_metrics
    import umap as _umap

    adj, features, labels = load_dataset(DATASET)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    t0 = time.perf_counter()
    reducer = _umap.UMAP(n_components=2, init=Y0, random_state=seed,
                         n_epochs=1000, n_jobs=1)
    Y = reducer.fit_transform(features)
    runtime_s = time.perf_counter() - t0
    Y = np.asarray(Y, dtype=np.float64)
    metrics = compute_metrics(features=features, adj=adj, Y=Y, labels=labels,
                              max_n=5000, seed=seed)
    np.save(_emb_path(out, "umap", seed), Y)
    row = _make_row("umap", seed, runtime_s, float("nan"), metrics)
    _atomic_write_parquet(pd.DataFrame([row]), cell)
    return row


def _worker_opentsne(seed: int, out_root: str) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    warnings.filterwarnings("ignore")
    out = Path(out_root)
    cell = _cell_path(out, "opentsne", seed)
    if cell.exists():
        return {"skipped": True, "method": "opentsne", "seed": int(seed)}
    from lens.data import load_dataset
    from lens.init import pca_init
    from lens.metrics import compute_metrics
    from openTSNE import TSNE

    adj, features, labels = load_dataset(DATASET)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    t0 = time.perf_counter()
    tsne = TSNE(perplexity=30, n_iter=1000,
                initialization=Y0.astype(np.float64),
                random_state=seed, n_jobs=1)
    emb = tsne.fit(features.astype(np.float64))
    runtime_s = time.perf_counter() - t0
    Y = np.asarray(emb, dtype=np.float64)
    metrics = compute_metrics(features=features, adj=adj, Y=Y, labels=labels,
                              max_n=5000, seed=seed)
    np.save(_emb_path(out, "opentsne", seed), Y)
    row = _make_row("opentsne", seed, runtime_s, float("nan"), metrics)
    _atomic_write_parquet(pd.DataFrame([row]), cell)
    return row


def _worker_phate(seed: int, out_root: str) -> dict:
    """Single-seed PHATE on the full ogbn_arxiv feature matrix.

    Internal threading allowed (n_jobs=-1). Memory + wall time are policed
    by the parent watchdog via a spawn child process; this worker just runs
    PHATE and writes its cell on success.
    """
    warnings.filterwarnings("ignore")
    out = Path(out_root)
    cell = _cell_path(out, "phate", seed)
    if cell.exists():
        return {"skipped": True, "method": "phate", "seed": int(seed)}
    from lens.data import load_dataset
    from lens.metrics import compute_metrics
    import phate

    adj, features, labels = load_dataset(DATASET)
    if features is None:
        raise RuntimeError("phate stage expects features for ogbn_arxiv")
    t0 = time.perf_counter()
    op = phate.PHATE(n_components=2, knn=15, n_jobs=-1,
                     random_state=seed, verbose=0)
    Y = op.fit_transform(features)
    runtime_s = time.perf_counter() - t0
    Y = np.asarray(Y, dtype=np.float64)
    metrics = compute_metrics(features=features, adj=adj, Y=Y, labels=labels,
                              max_n=5000, seed=seed)
    np.save(_emb_path(out, "phate", seed), Y)
    row = _make_row("phate", seed, runtime_s, float("nan"), metrics)
    _atomic_write_parquet(pd.DataFrame([row]), cell)
    return row


def _worker_node2vec(seed: int, out_root: str) -> dict:
    """Single-seed node2vec(workers=8) -> UMAP head on full graph."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    warnings.filterwarnings("ignore")
    out = Path(out_root)
    cell = _cell_path(out, "node2vec_umap", seed)
    if cell.exists():
        return {"skipped": True, "method": "node2vec_umap", "seed": int(seed)}
    from lens.data import load_dataset
    from lens.init import pca_init
    from lens.metrics import compute_metrics
    import networkx as nx
    import umap as _umap
    from node2vec import Node2Vec

    adj, features, labels = load_dataset(DATASET)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    t0 = time.perf_counter()
    G = nx.from_scipy_sparse_array(adj)
    nv = Node2Vec(G, dimensions=64, walk_length=80, num_walks=10,
                  workers=8, quiet=True, seed=seed)
    model = nv.fit(window=10, min_count=1, batch_words=4)
    n = adj.shape[0]
    X_nv = np.array([model.wv[str(i)] for i in range(n)], dtype=np.float32)
    reducer = _umap.UMAP(n_components=2, init=Y0, random_state=seed,
                         n_epochs=1000, n_jobs=1)
    Y = reducer.fit_transform(X_nv)
    runtime_s = time.perf_counter() - t0
    Y = np.asarray(Y, dtype=np.float64)
    feat_for_metrics = features if features is not None else None
    metrics = compute_metrics(features=feat_for_metrics, adj=adj, Y=Y,
                              labels=labels, max_n=5000, seed=seed)
    np.save(_emb_path(out, "node2vec_umap", seed), Y)
    row = _make_row("node2vec_umap", seed, runtime_s, float("nan"), metrics)
    _atomic_write_parquet(pd.DataFrame([row]), cell)
    return row


# ---------- Watchdog (PHATE + node2vec single-process stages) --------------

def _spawn_target(target_name: str, seed: int, out_root: str) -> None:
    """Trampoline executed inside the spawn child process."""
    targets = {
        "phate": _worker_phate,
        "node2vec_umap": _worker_node2vec,
    }
    targets[target_name](seed=seed, out_root=out_root)


def _bound_subprocess(method: str, seed: int, out_root: Path,
                      time_limit_s: float, mem_limit_gb: float | None,
                      status_path: Path) -> dict:
    """Run the named worker in a spawn child with wall + (optional) RSS guard."""
    import psutil

    cell = _cell_path(out_root, method, seed)
    if cell.exists():
        print(f"[guard] {method} seed={seed} already on disk; skipping", flush=True)
        return {"status": "skipped", "method": method, "seed": int(seed)}

    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_spawn_target,
                       args=(method, int(seed), str(out_root)),
                       name=f"{method}_seed{seed}")
    t0 = time.perf_counter()
    proc.start()
    pid = proc.pid
    print(f"[guard] launched {method} seed={seed} pid={pid} "
          f"limits: wall={time_limit_s/60:.0f}m mem={mem_limit_gb}GB", flush=True)

    poll_interval_s = 30.0
    peak_rss_gb = 0.0
    reason: str | None = None

    while proc.is_alive():
        time.sleep(poll_interval_s)
        elapsed = time.perf_counter() - t0
        rss_gb = 0.0
        try:
            ps = psutil.Process(pid)
            rss_gb = ps.memory_info().rss / (1024 ** 3)
            for c in ps.children(recursive=True):
                try:
                    rss_gb += c.memory_info().rss / (1024 ** 3)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            rss_gb = 0.0
        peak_rss_gb = max(peak_rss_gb, rss_gb)
        if int(elapsed) % 300 < poll_interval_s:
            print(f"[guard] {method} seed={seed} elapsed={elapsed/60:.1f}m "
                  f"rss={rss_gb:.1f}GB (peak {peak_rss_gb:.1f}GB)", flush=True)
        if mem_limit_gb is not None and rss_gb > mem_limit_gb:
            reason = f"mem_exceeded_rss={rss_gb:.1f}GB"
            break
        if elapsed > time_limit_s:
            reason = f"time_exceeded_elapsed={elapsed/60:.1f}m"
            break

    if reason is not None:
        print(f"[guard] ABORT {method} seed={seed}: {reason}", flush=True)
        try:
            ps = psutil.Process(pid)
            for c in ps.children(recursive=True):
                try:
                    c.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            ps.terminate()
            try:
                ps.wait(timeout=5)
            except psutil.TimeoutExpired:
                for c in ps.children(recursive=True):
                    try:
                        c.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                ps.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        proc.join(10)
        info = {"status": "aborted", "method": method, "seed": int(seed),
                "reason": reason, "peak_rss_gb": float(peak_rss_gb),
                "elapsed_s": float(time.perf_counter() - t0)}
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(info, indent=2))
        return info

    proc.join()
    elapsed = time.perf_counter() - t0
    if proc.exitcode == 0 and cell.exists():
        info = {"status": "ok", "method": method, "seed": int(seed),
                "peak_rss_gb": float(peak_rss_gb), "elapsed_s": float(elapsed)}
        print(f"[guard] OK    {method} seed={seed} elapsed={elapsed/60:.1f}m "
              f"peak_rss={peak_rss_gb:.1f}GB", flush=True)
        return info

    info = {"status": "failed", "method": method, "seed": int(seed),
            "reason": f"exit_code={proc.exitcode} cell_exists={cell.exists()}",
            "peak_rss_gb": float(peak_rss_gb), "elapsed_s": float(elapsed)}
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(info, indent=2))
    print(f"[guard] FAIL  {method} seed={seed}: exit={proc.exitcode}", flush=True)
    return info


# ---------- Stage drivers --------------------------------------------------

def _run_pool(label: str, fn, items: list[tuple], max_workers: int) -> None:
    if not items:
        print(f"[{label}] all cells already on disk", flush=True)
        return
    print(f"[{label}] {len(items)} cells on {max_workers} workers", flush=True)
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, *args): args for args in items}
        for fut in as_completed(futs):
            args = futs[fut]
            try:
                row = fut.result()
                if row.get("skipped"):
                    print(f"[{label}] skip {args}", flush=True)
                    continue
                done += 1
                lt = row.get("label_trustworthiness", float("nan"))
                rt = row.get("runtime_s", float("nan"))
                print(f"[{label}] [{done}/{len(items)}] {args} "
                      f"LT={lt:.3f} rt={rt:.0f}s "
                      f"elapsed={(time.perf_counter()-t0)/60:.1f}m", flush=True)
            except Exception as e:
                print(f"[{label}] FAIL {args}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)


def stage2_pysgtsnepi(out_root: Path, auto_lam: float, seeds: tuple[int, ...]) -> None:
    items = [(s, auto_lam, str(out_root)) for s in seeds
             if not _cell_path(out_root, "pysgtsnepi", s).exists()]
    _run_pool("S2 pysgtsnepi", _worker_pysgtsnepi, items, max_workers=2)


def stage3_umap(out_root: Path, seeds: tuple[int, ...]) -> None:
    items = [(s, str(out_root)) for s in seeds
             if not _cell_path(out_root, "umap", s).exists()]
    _run_pool("S3 umap", _worker_umap, items, max_workers=2)


def stage4_opentsne(out_root: Path, seeds: tuple[int, ...]) -> None:
    items = [(s, str(out_root)) for s in seeds
             if not _cell_path(out_root, "opentsne", s).exists()]
    _run_pool("S4 opentsne", _worker_opentsne, items, max_workers=5)


def stage5_phate(out_root: Path) -> None:
    status = out_root / "meta" / "phate_status.json"
    info = _bound_subprocess(
        method="phate", seed=42, out_root=out_root,
        time_limit_s=PHATE_TIME_LIMIT_S, mem_limit_gb=PHATE_MEM_LIMIT_GB,
        status_path=status,
    )
    print(f"[S5] phate result: {info['status']}", flush=True)


def stage6_node2vec(out_root: Path) -> None:
    status = out_root / "meta" / "node2vec_status.json"
    info = _bound_subprocess(
        method="node2vec_umap", seed=42, out_root=out_root,
        time_limit_s=NODE2VEC_TIME_LIMIT_S, mem_limit_gb=None,
        status_path=status,
    )
    print(f"[S6] node2vec_umap result: {info['status']}", flush=True)


# ---------- Stage 7: aggregate ---------------------------------------------

def stage7_aggregate(out_root: Path) -> None:
    cells_dir = out_root / "tables" / "cells_baselines"
    files = sorted(cells_dir.glob(f"{DATASET}_*_seed*.parquet"))
    if not files:
        print(f"[S7] no cells found for {DATASET}", flush=True)
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["method", "seed"]).reset_index(drop=True)
    comp_p = out_root / "tables" / f"{DATASET}_comparison.parquet"
    _atomic_write_parquet(df, comp_p)
    print(f"[S7] wrote {len(df)} rows to {comp_p.name}", flush=True)

    methods = ["pysgtsnepi", "umap", "opentsne", "phate", "node2vec_umap"]
    metric_cols = ["label_trustworthiness", "label_continuity",
                   "trustworthiness", "continuity", "runtime_s"]
    agg_rows = []
    for method in methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        arow: dict = {"method": method, "n_seeds": int(sub["seed"].nunique())}
        for col in metric_cols:
            vals = sub[col].dropna()
            arow[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            arow[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else float("nan")
            arow[f"{col}_median"] = float(vals.median()) if len(vals) else float("nan")
        agg_rows.append(arow)
    agg_df = pd.DataFrame(agg_rows)
    agg_p = out_root / "tables" / f"{DATASET}_comparison_agg.parquet"
    _atomic_write_parquet(agg_df, agg_p)
    print(f"[S7] wrote {len(agg_df)} method rows to {agg_p.name}", flush=True)
    for r in agg_df.itertuples():
        lt = getattr(r, "label_trustworthiness_mean", float("nan"))
        rt = getattr(r, "runtime_s_mean", float("nan"))
        print(f"  {r.method}: n={r.n_seeds} label_T={lt:.3f} runtime={rt:.0f}s", flush=True)


# ---------- main -----------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="output")
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--skip-phate", action="store_true",
                   help="skip stage 5 (PHATE single-seed full graph)")
    p.add_argument("--skip-node2vec", action="store_true",
                   help="skip stage 6 (node2vec_umap single-seed)")
    p.add_argument("--only", nargs="+", default=None,
                   help="optional: subset of stages to run, e.g. --only s1 s2 s7")
    args = p.parse_args()

    out_root = Path(args.out_root)
    seeds = tuple(int(s) for s in args.seeds)
    only = set(args.only) if args.only else None

    def should_run(tag: str) -> bool:
        return only is None or tag in only

    print(f"=== R4-E1 OGBN-arxiv (seeds={seeds}, out={out_root}) ===", flush=True)

    auto_lam = 20.0
    if should_run("s1"):
        auto_lam = stage1_auto_lambda(out_root)
    else:
        print("[S1] skipped (using default auto_lam=20.0)", flush=True)

    if should_run("s2"):
        stage2_pysgtsnepi(out_root, auto_lam, seeds)
    if should_run("s3"):
        stage3_umap(out_root, seeds)
    if should_run("s4"):
        stage4_opentsne(out_root, seeds)
    if should_run("s5") and not args.skip_phate:
        stage5_phate(out_root)
    elif args.skip_phate:
        print("[S5] PHATE skipped (--skip-phate)", flush=True)
    if should_run("s6") and not args.skip_node2vec:
        stage6_node2vec(out_root)
    elif args.skip_node2vec:
        print("[S6] node2vec_umap skipped (--skip-node2vec)", flush=True)
    if should_run("s7"):
        stage7_aggregate(out_root)

    print("=== R4-E1 OGBN-arxiv DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
