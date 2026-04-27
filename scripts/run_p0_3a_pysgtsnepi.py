"""P0-3a: pysgtsnepi multi-seed pass at auto-lambda, PCA-init.

5 seeds × 5 datasets via run_one_cell. Reuses cached seed=42 cells from E2 if
present (resume-safe via per-cell parquet). Auto-lambda from
output/tables/auto_lambda_summary.parquet for labeled datasets; default
lambda=10 for unlabeled (matches E3 Phase 6 choice).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

DEFAULT_DATASETS = ("cora", "citeseer", "mnist_knn", "pbmc", "ca_astroph")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_UNLABELED_LAMBDA = 10.0


def _auto_lambdas(out_root: Path) -> dict[str, float]:
    """Returns {dataset: auto_lambda} only for datasets where the value is a finite float."""
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


def _existing_pca_cells(out_root: Path, dataset: str) -> set[tuple[float, int]]:
    cells_dir = out_root / "tables" / "cells"
    found: set[tuple[float, int]] = set()
    if not cells_dir.exists():
        return found
    for p in cells_dir.glob(f"{dataset}_*init=pca*.parquet"):
        try:
            df = pd.read_parquet(p)
            for r in df.itertuples():
                found.add((float(r.lambda_), int(r.seed)))
        except Exception:
            continue
    return found


def _worker(dataset: str, lambda_: float, seed: int, out_root: str) -> dict:
    from lens.data import load_dataset
    from lens.run import run_one_cell
    adj, features, labels = load_dataset(dataset)
    uw = dataset != "pbmc"
    return run_one_cell(
        adj, features, labels, lambda_, seed, dataset, out_root,
        init="pca", unweighted_to_weighted=uw,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--out-root", default="output")
    p.add_argument("--unlabeled-lambda", type=float, default=DEFAULT_UNLABELED_LAMBDA)
    args = p.parse_args()

    out_root = Path(args.out_root)
    auto = _auto_lambdas(out_root)

    cells: list[tuple[str, float, int]] = []
    for ds in args.datasets:
        lam = auto.get(ds, args.unlabeled_lambda)
        done = _existing_pca_cells(out_root, ds)
        for seed in args.seeds:
            if (lam, seed) not in done:
                cells.append((ds, lam, seed))
        print(f"[p0-3a] {ds}: lambda={lam} (auto={ds in auto}), "
              f"existing PCA cells={len(done)}, queued={sum(1 for c in cells if c[0]==ds)}",
              flush=True)

    if not cells:
        print("[p0-3a] nothing to run; all cells cached", flush=True)
        return 0

    print(f"[p0-3a] running {len(cells)} cells on {args.max_workers} workers", flush=True)
    t0 = time.perf_counter()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_worker, ds, lam, seed, args.out_root): (ds, lam, seed)
                for ds, lam, seed in cells}
        for fut in as_completed(futs):
            ds, lam, seed = futs[fut]
            try:
                row = fut.result()
                completed += 1
                el = time.perf_counter() - t0
                print(f"[p0-3a] [{completed}/{len(cells)}] {ds} lam={lam} seed={seed} "
                      f"runtime={row['runtime_s']:.1f}s elapsed={el:.1f}s", flush=True)
            except Exception as e:
                print(f"[p0-3a] FAILED {ds} lam={lam} seed={seed}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
