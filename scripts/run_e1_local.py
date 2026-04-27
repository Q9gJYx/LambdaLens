"""Drive the full E1 grid (datasets x lambdas x seeds) with ProcessPoolExecutor.

Resumable: skips (dataset, lambda, seed) cells whose per-cell parquet already
exists under output/tables/cells/. After all cells finish, merges per-cell
parquets into a single per-dataset parquet at output/tables/{ds}_lambda_grid.parquet.

Default lambda grid restricted to {0.5, 1, 2, 5, 10, 20} per literature audit
(Track A, 2026-04-26): lambda > ~k drives mass negative-sigma failures on
moderate-degree sparse graphs and is not operationally validated; published
SG-t-SNE-Pi work tops out at lambda=80 (Mobius edge case, not a benchmark).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

DEFAULT_DATASETS = ("cora", "citeseer", "mnist_knn", "ca_astroph")
DEFAULT_LAMBDAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
DEFAULT_SEEDS = (42, 43, 44)


def _existing_cells(
    out_root: Path, dataset: str, init: str, uw: bool
) -> set[tuple[float, int]]:
    """Return (lambda, seed) pairs already run for this (dataset, init, uw) combo.

    Cells are stored by run_one_cell with init / unweighted_to_weighted in the
    suffix; this resume key reads the parquet rows and matches all four fields
    so different (init, uw) variants for the same (lambda, seed) coexist.
    """
    cells_dir = out_root / "tables" / "cells"
    if not cells_dir.exists():
        return set()
    found: set[tuple[float, int]] = set()
    for p in cells_dir.glob(f"{dataset}_lam*_seed*.parquet"):
        try:
            df = pd.read_parquet(p)
            for r in df.itertuples():
                row_init = getattr(r, "init", "random")
                row_uw = bool(getattr(r, "unweighted_to_weighted", True))
                if row_init == init and row_uw == uw:
                    found.add((float(r.lambda_), int(r.seed)))
        except Exception:
            continue
    return found


def _merge_per_dataset(out_root: Path, datasets: list[str]) -> None:
    cells_dir = out_root / "tables" / "cells"
    if not cells_dir.exists():
        return
    for ds in datasets:
        files = sorted(cells_dir.glob(f"{ds}_lam*_seed*.parquet"))
        if not files:
            continue
        dfs = [pd.read_parquet(p) for p in files]
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["dataset", "lambda_", "seed"], keep="last"
        )
        combined = combined.sort_values(["lambda_", "seed"]).reset_index(drop=True)
        combined.to_parquet(out_root / "tables" / f"{ds}_lambda_grid.parquet", index=False)
        print(f"[merge] {ds}: {len(combined)} rows", flush=True)


def _worker(
    dataset: str,
    lambda_: float,
    seed: int,
    out_root: str,
    zadu_subsample_n: int,
    init: str,
    uw: bool,
) -> dict:
    from lens.data import load_dataset
    from lens.run import run_one_cell

    adj, features, labels = load_dataset(dataset)
    return run_one_cell(
        adj,
        features,
        labels,
        lambda_,
        seed,
        dataset,
        out_root,
        zadu_subsample_n,
        init=init,
        unweighted_to_weighted=uw,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--out-root", default="output")
    p.add_argument("--zadu-subsample-n", type=int, default=5000)
    p.add_argument("--init", choices=["random", "pca"], default="random")
    p.add_argument(
        "--unweighted-to-weighted",
        choices=["true", "false"],
        default="true",
        help="Pass to pysgtsnepi. PBMC (already stochastic) needs 'false'.",
    )
    p.add_argument(
        "--state-json",
        default="output/meta/run_e1_state.json",
        help="Per-cell completion log for monitoring.",
    )
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip the final per-dataset parquet merge step.",
    )
    args = p.parse_args()

    out_root = Path(args.out_root)
    state_path = Path(args.state_json)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    uw = args.unweighted_to_weighted == "true"
    cells: list[tuple[str, float, int]] = []
    for ds in args.datasets:
        done = _existing_cells(out_root, ds, args.init, uw)
        for lam in args.lambdas:
            for seed in args.seeds:
                if (lam, seed) in done:
                    continue
                cells.append((ds, lam, seed))

    total = len(cells)
    if total == 0:
        print("[grid] nothing to do — all cells already present.", flush=True)
        if not args.no_merge:
            _merge_per_dataset(out_root, list(args.datasets))
        return 0

    print(
        f"[grid] {total} cells to run "
        f"({len(args.datasets)} datasets x {len(args.lambdas)} lambdas x "
        f"{len(args.seeds)} seeds, minus already-done) "
        f"on {args.max_workers} workers",
        flush=True,
    )

    t0 = time.perf_counter()
    completed = 0
    state_path.write_text(json.dumps({"started_at": t0, "completed": 0, "total": total}))

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(
                _worker,
                ds,
                lam,
                seed,
                args.out_root,
                args.zadu_subsample_n,
                args.init,
                uw,
            ): (ds, lam, seed)
            for ds, lam, seed in cells
        }
        for fut in as_completed(futs):
            ds, lam, seed = futs[fut]
            try:
                row = fut.result()
                completed += 1
                elapsed = time.perf_counter() - t0
                eta = elapsed / completed * (total - completed)
                print(
                    f"[grid] [{completed}/{total}] "
                    f"{ds} lam={lam} seed={seed} "
                    f"runtime={row['runtime_s']:.1f}s "
                    f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
                state_path.write_text(
                    json.dumps(
                        {
                            "started_at": t0,
                            "completed": completed,
                            "total": total,
                            "elapsed_s": elapsed,
                            "eta_s": eta,
                            "last_cell": {"dataset": ds, "lambda_": lam, "seed": seed},
                        }
                    )
                )
            except Exception as e:
                print(
                    f"[grid] FAILED {ds} lam={lam} seed={seed}: {type(e).__name__}: {e}",
                    flush=True,
                    file=sys.stderr,
                )

    print(
        f"[grid] DONE: {completed}/{total} cells in "
        f"{(time.perf_counter() - t0)/60:.1f} min",
        flush=True,
    )

    if not args.no_merge:
        _merge_per_dataset(out_root, list(args.datasets))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
