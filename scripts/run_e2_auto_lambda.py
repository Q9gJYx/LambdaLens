"""E2: auto-lambda heuristic validation.

For each dataset, ensure cells exist at lambda in {1, 2, 5, 10, 20, 50, 80}
(seed=42), then pick:

  auto_lambda     = argmax over {1, 5, 20, 50}   of label_T&C (or trustworthiness if no labels)
  gridsearch_lambda = argmax over {1, 2, 5, 10, 20, 50, 80} of same metric

Reports whether they match. Fills in missing cells via ProcessPoolExecutor.

Datasets: pbmc (uw=False), cora, citeseer, ca_astroph, mnist_knn (default uw=True).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ("pbmc", "cora", "citeseer", "ca_astroph", "mnist_knn", "pubmed")
PROBE_LAMBDAS = (1.0, 5.0, 20.0, 50.0)
GRIDSEARCH_LAMBDAS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 80.0)
SEED = 42
UW_BY_DATASET = {"pbmc": False}  # everything else default True


def _cell_filename(dataset: str, lam: float, seed: int, uw: bool) -> str:
    suffix = "_uw=False" if not uw else ""
    return f"{dataset}_lam{lam}_seed{seed}{suffix}.parquet"


def _existing_for_grid(out_root: Path, dataset: str, uw: bool) -> set[float]:
    cells_dir = out_root / "tables" / "cells"
    if not cells_dir.exists():
        return set()
    found: set[float] = set()
    for lam in GRIDSEARCH_LAMBDAS:
        p = cells_dir / _cell_filename(dataset, lam, SEED, uw)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                row = df[(df.lambda_ == lam) & (df.seed == SEED)]
                if not row.empty:
                    init_val = str(row.iloc[0].get("init", "random"))
                    uw_val = bool(row.iloc[0].get("unweighted_to_weighted", True))
                    if init_val == "random" and uw_val == uw:
                        found.add(lam)
            except Exception:
                continue
    return found


def _worker(dataset: str, lambda_: float, seed: int, uw: bool, out_root: str, zadu_subsample_n: int) -> dict:
    from lens.data import load_dataset
    from lens.run import run_one_cell

    adj, features, labels = load_dataset(dataset)
    return run_one_cell(
        adj, features, labels, lambda_, seed, dataset, out_root, zadu_subsample_n,
        init="random", unweighted_to_weighted=uw,
    )


def _read_metric(out_root: Path, dataset: str, lam: float, uw: bool) -> tuple[float, str]:
    """Return (metric_value, metric_name) for the (dataset, lam, seed=42, init=random, uw) cell."""
    p = out_root / "tables" / "cells" / _cell_filename(dataset, lam, SEED, uw)
    df = pd.read_parquet(p)
    row = df.iloc[0].to_dict()
    label_t = float(row.get("label_trustworthiness", float("nan")))
    label_c = float(row.get("label_continuity", float("nan")))
    if not np.isnan(label_t) and not np.isnan(label_c):
        return 0.5 * (label_t + label_c), "label_T&C"
    t = float(row.get("trustworthiness", float("nan")))
    c = float(row.get("continuity", float("nan")))
    if not np.isnan(t) and not np.isnan(c):
        return 0.5 * (t + c), "T&C"
    return float("nan"), "unavailable"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--out-root", default="output")
    p.add_argument("--zadu-subsample-n", type=int, default=5000)
    args = p.parse_args()

    out_root = Path(args.out_root)
    cells: list[tuple[str, float, int, bool]] = []
    for ds in args.datasets:
        uw = UW_BY_DATASET.get(ds, True)
        done = _existing_for_grid(out_root, ds, uw)
        for lam in GRIDSEARCH_LAMBDAS:
            if lam not in done:
                cells.append((ds, lam, SEED, uw))

    if cells:
        print(f"[e2] running {len(cells)} missing cells on {args.max_workers} workers", flush=True)
        t0 = time.perf_counter()
        completed = 0
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {
                ex.submit(_worker, ds, lam, seed, uw, args.out_root, args.zadu_subsample_n): (ds, lam, uw)
                for ds, lam, seed, uw in cells
            }
            for fut in as_completed(futs):
                ds, lam, uw = futs[fut]
                try:
                    row = fut.result()
                    completed += 1
                    elapsed = time.perf_counter() - t0
                    print(f"[e2] [{completed}/{len(cells)}] {ds} lam={lam} uw={uw} runtime={row['runtime_s']:.1f}s elapsed={elapsed/60:.1f}m",
                          flush=True)
                except Exception as e:
                    print(f"[e2] FAILED {ds} lam={lam}: {type(e).__name__}: {e}", flush=True, file=sys.stderr)
    else:
        print("[e2] all cells already present", flush=True)

    summary_rows = []
    for ds in args.datasets:
        uw = UW_BY_DATASET.get(ds, True)
        scores: dict[float, tuple[float, str]] = {}
        for lam in GRIDSEARCH_LAMBDAS:
            try:
                scores[lam] = _read_metric(out_root, ds, lam, uw)
            except Exception:
                scores[lam] = (float("nan"), "missing")

        valid_grid = {l: s for l, (s, _) in scores.items() if not np.isnan(s)}
        valid_probe = {l: scores[l][0] for l in PROBE_LAMBDAS if l in scores and not np.isnan(scores[l][0])}

        if valid_probe and valid_grid:
            auto_lam = max(valid_probe, key=valid_probe.get)
            grid_lam = max(valid_grid, key=valid_grid.get)
            auto_metric = valid_probe[auto_lam]
            grid_metric = valid_grid[grid_lam]
            metric_used = scores[auto_lam][1]
            match = bool(auto_lam == grid_lam)
        else:
            auto_lam = grid_lam = float("nan")
            auto_metric = grid_metric = float("nan")
            metric_used = "unavailable"
            match = False

        summary_rows.append({
            "dataset": ds,
            "auto_lambda": auto_lam,
            "auto_metric": auto_metric,
            "gridsearch_lambda": grid_lam,
            "gridsearch_metric": grid_metric,
            "match": match,
            "metric_used": metric_used,
        })
        print(f"[e2] {ds}: auto_lam={auto_lam} ({auto_metric:.4f}) | "
              f"gridsearch_lam={grid_lam} ({grid_metric:.4f}) | match={match} | metric={metric_used}",
              flush=True)

    summary_df = pd.DataFrame(summary_rows)
    out_path = out_root / "tables" / "auto_lambda_summary.parquet"
    summary_df.to_parquet(out_path, index=False)
    print(f"[e2] wrote {out_path} ({len(summary_df)} rows; {sum(r['match'] for r in summary_rows)}/{len(summary_rows)} match)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
