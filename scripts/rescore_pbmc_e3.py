"""Recompute label_T/label_C for cached PBMC E3 cell parquets against
the current data/processed/pbmc/labels.npy.

Cells are independent; we parallelize via ProcessPoolExecutor. Each
worker loads its own copy of the PBMC graph + new labels and rescores
one (method, seed) parquet. Compute_metrics is the bottleneck (~50 s
per cell on one core for n=8381 graph + max_n=5000 zadu T&C in 8381-d
feature space); 8 workers brings 45 cells to ~3-4 min wall.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os

import numpy as np
import pandas as pd

CELLS_DIR = Path("output/tables/cells_baselines")
EMB_DIR = Path("output/embeddings_baselines")
DATASET = "pbmc"
N_WORKERS = 8


def _rescore_one(parquet_str: str) -> dict:
    """Worker: load adj+labels lazily (per-process), rescore one parquet."""
    p = Path(parquet_str)
    df = pd.read_parquet(p)
    if df.empty:
        return {"name": p.name, "status": "empty"}
    row = df.iloc[0]
    method = row["method"]
    seed = int(row["seed"])
    emb = EMB_DIR / f"{DATASET}_{method}_seed{seed}.npy"
    if not emb.exists():
        return {"name": p.name, "status": "no_emb"}

    from lens.data import load_pbmc
    from lens.metrics import compute_metrics
    adj, _, labels = load_pbmc()
    Y = np.load(emb)

    m = compute_metrics(features=None, adj=adj, Y=Y, labels=labels,
                        max_n=5000, seed=seed)
    old_lt = float(row.get("label_trustworthiness", float("nan")))
    old_lc = float(row.get("label_continuity", float("nan")))
    new_lt = float(m["label_trustworthiness"])
    new_lc = float(m["label_continuity"])
    df.loc[df.index[0], "label_trustworthiness"] = new_lt
    df.loc[df.index[0], "label_continuity"] = new_lc
    df.to_parquet(p, index=False)
    return {
        "name": p.name, "status": "ok", "method": method, "seed": seed,
        "old_lt": old_lt, "new_lt": new_lt,
        "old_lc": old_lc, "new_lc": new_lc,
    }


def main() -> int:
    parquets = sorted(CELLS_DIR.glob(f"{DATASET}_*.parquet"))
    print(f"[rescore] {len(parquets)} parquets, {N_WORKERS} workers", flush=True)
    n_ok = n_skip = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_rescore_one, str(p)): p for p in parquets}
        for fut in as_completed(futs):
            r = fut.result()
            if r["status"] != "ok":
                n_skip += 1
                print(f"  [skip:{r['status']}] {r['name']}", flush=True)
                continue
            n_ok += 1
            print(
                f"  [ok] {r['method']:13s} seed={r['seed']}: "
                f"LT {r['old_lt']:.3f}→{r['new_lt']:.3f}  "
                f"LC {r['old_lc']:.3f}→{r['new_lc']:.3f}",
                flush=True,
            )
    print(f"[rescore] updated={n_ok} skipped={n_skip}", flush=True)
    return 0


if __name__ == "__main__":
    # avoid BLAS oversubscription with 8 worker procs each spawning threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    raise SystemExit(main())
