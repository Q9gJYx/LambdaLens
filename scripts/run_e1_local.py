"""Drive the full E1 grid (datasets x lambdas x seeds) with ProcessPoolExecutor.

Resumable: skips (dataset, lambda, seed) cells already present in the parquet.
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
DEFAULT_LAMBDAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
DEFAULT_SEEDS = (42, 43, 44)


def _existing_cells(out_root: Path, dataset: str) -> set[tuple[float, int]]:
    p = out_root / "tables" / f"{dataset}_lambda_grid.parquet"
    if not p.exists():
        return set()
    df = pd.read_parquet(p)
    return {(float(r.lambda_), int(r.seed)) for r in df.itertuples()}


def _worker(
    dataset: str, lambda_: float, seed: int, out_root: str, zadu_subsample_n: int
) -> dict:
    # Import lazily inside the worker so each child process has a clean import tree.
    from lens.data import load_dataset
    from lens.run import run_one_cell

    adj, features, labels = load_dataset(dataset)
    row = run_one_cell(
        adj, features, labels, lambda_, seed, dataset, out_root, zadu_subsample_n
    )
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument(
        "--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS)
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--out-root", default="output")
    p.add_argument("--zadu-subsample-n", type=int, default=5000)
    p.add_argument(
        "--state-json",
        default="output/meta/run_e1_state.json",
        help="Per-cell completion log for monitoring.",
    )
    args = p.parse_args()

    out_root = Path(args.out_root)
    state_path = Path(args.state_json)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    cells: list[tuple[str, float, int]] = []
    for ds in args.datasets:
        done = _existing_cells(out_root, ds)
        for lam in args.lambdas:
            for seed in args.seeds:
                if (lam, seed) in done:
                    continue
                cells.append((ds, lam, seed))

    total = len(cells)
    if total == 0:
        print("[grid] nothing to do — all cells already present.", flush=True)
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
                _worker, ds, lam, seed, args.out_root, args.zadu_subsample_n
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
