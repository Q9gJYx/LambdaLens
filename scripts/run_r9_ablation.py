"""R9-D4: subtractive ablation on Cora + PBMC, N=5 seeds.

Variants:
  full          auto-lambda + PCA-init + Jaccard (uw=True) [PBMC: uw=False, no Jaccard step]
  no_pca        random init, otherwise full
  no_jaccard    uw=False, otherwise full (Cora only; PBMC structurally N/A)
  lam1          lambda=1, otherwise full
  lam20         lambda=20, otherwise full

Outputs:
  output/tables/ablation_v1.parquet     long form (dataset, variant, seed, label_T, label_C, runtime_s)
  output/tables/ablation_summary.csv    pre-aggregated mean/std per (dataset, variant)
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

DATASETS = ["cora", "pbmc"]
SEEDS = [42, 43, 44, 45, 46]
AUTO_LAM = {"cora": 20.0, "pbmc": 5.0}  # R5-A1 corrected auto-lambda

# (variant_name, init, lambda_override, uw_override) where None means "default for variant"
VARIANTS = [
    ("full",       "pca",    None, None),
    ("no_pca",     "random", None, None),
    ("no_jaccard", "pca",    None, False),  # Cora only
    ("lam1",       "pca",    1.0,  None),
    ("lam20",      "pca",    20.0, None),
]


def _worker(ds: str, variant: str, init: str, lam_override, uw_override, seed: int) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as _np
    from lens.data import load_dataset
    from lens.init import pca_init
    from lens.metrics import compute_metrics
    from pysgtsnepi import sgtsnepi

    adj, features, labels = load_dataset(ds)

    lam = float(AUTO_LAM[ds] if lam_override is None else lam_override)
    # default uw: True for Cora (Jaccard preprocessing), False for PBMC (already stochastic)
    uw_default = ds != "pbmc"
    uw = uw_default if uw_override is None else uw_override

    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed) if init == "pca" else None

    t0 = time.perf_counter()
    Y = sgtsnepi(
        adj, d=2, lambda_=lam, random_state=seed, Y0=Y0,
        unweighted_to_weighted=uw,
    )
    runtime_s = time.perf_counter() - t0

    Y_arr = _np.asarray(Y, dtype=_np.float64)
    m = compute_metrics(features=features, adj=adj, Y=Y_arr,
                        labels=labels, max_n=5000, seed=seed)
    return {
        "dataset": ds, "variant": variant, "seed": int(seed),
        "lambda_": lam, "init": init, "unweighted_to_weighted": bool(uw),
        "label_T": float(m.get("label_trustworthiness", float("nan"))),
        "label_C": float(m.get("label_continuity", float("nan"))),
        "trust": float(m.get("trustworthiness", float("nan"))),
        "cont": float(m.get("continuity", float("nan"))),
        "runtime_s": float(runtime_s),
    }


def _cells_to_run(datasets, seeds):
    cells = []
    for ds in datasets:
        for variant, init, lam_override, uw_override in VARIANTS:
            # PBMC -- Jaccard is N/A (no Jaccard step in pipeline)
            if variant == "no_jaccard" and ds == "pbmc":
                continue
            for seed in seeds:
                cells.append((ds, variant, init, lam_override, uw_override, seed))
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--out-root", default="output")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    tables = out_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    cells = _cells_to_run(args.datasets, args.seeds)
    print(f"[R9-D4] {len(cells)} cells | workers={args.max_workers}", flush=True)

    rows: list[dict] = []
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(_worker, ds, v, init, lo, uo, sd): (ds, v, sd)
            for ds, v, init, lo, uo, sd in cells
        }
        done = 0
        for fut in as_completed(futs):
            ds, v, sd = futs[fut]
            try:
                r = fut.result()
                rows.append(r)
                done += 1
                el = (time.perf_counter() - t_start) / 60
                print(
                    f"[{done:>3}/{len(cells)}] {ds:>5} {v:>11} seed={sd}  "
                    f"LT={r['label_T']:.4f} LC={r['label_C']:.4f} "
                    f"rt={r['runtime_s']:.1f}s  elapsed={el:.1f}m",
                    flush=True,
                )
            except Exception as e:
                print(f"[FAIL] {ds} {v} seed={sd}: {e}", file=sys.stderr, flush=True)

    df = pd.DataFrame(rows)
    parquet_path = tables / "ablation_v1.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"[R9-D4] wrote {parquet_path} ({len(df)} rows)", flush=True)

    agg = (
        df.groupby(["dataset", "variant"])
        .agg(
            label_T_mean=("label_T", "mean"),
            label_T_std=("label_T", "std"),
            label_C_mean=("label_C", "mean"),
            label_C_std=("label_C", "std"),
            runtime_s_mean=("runtime_s", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    csv_path = tables / "ablation_summary.csv"
    agg.to_csv(csv_path, index=False)
    print(f"[R9-D4] wrote {csv_path} ({len(agg)} rows)", flush=True)

    print("[R9-D4] summary:", flush=True)
    print(agg.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
