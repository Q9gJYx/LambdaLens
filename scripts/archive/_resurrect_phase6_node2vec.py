"""Bring Phase-6 single-seed node2vec_umap embeddings into round-3 cells_baselines/.

Used for ca_astroph node2vec_umap, where round-3 single-seed (workers=1, PCA-init UMAP head)
would take ~70 min on M3 Pro and adds little value over Phase-6 single-seed
(random init UMAP head, workers=4, runtime 1107 s).

Recomputes T/C via lens.metrics.compute_metrics on the Phase-6 embedding;
labels=None for ca_astroph (graph-only, no labels), so only T/C populated
(label_T&C remain NaN). Runtime carried over from Phase-6 record.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lens.data import load_dataset
from lens.metrics import compute_metrics

PHASE6_CELLS = [
    ("ca_astroph", "node2vec_umap", 1106.5,
     {"init_strategy": "random_phase6_carryover", "iteration_count": 200,
      "info_lib": "node2vec", "info_n2v_dim": 64, "info_n2v_workers": 4}),
    ("ca_astroph", "phate", 10.5,
     {"init_strategy": "phate_internal_pca", "iteration_count": -1,
      "info_input_format": "feature_like(adj_rows)_phase6"}),
]


def main() -> int:
    out_root = Path("output")
    written = 0
    for ds, method, rt, extra_info in PHASE6_CELLS:
        ph6_npy = out_root / "embeddings_baselines" / f"{ds}_{method}_seed42_phase6.npy"
        if not ph6_npy.exists():
            print(f"[resurrect] {ds}/{method}: no Phase-6 npy at {ph6_npy}", file=sys.stderr)
            continue
        Y = np.load(ph6_npy)
        adj, features, labels = load_dataset(ds)
        metrics = compute_metrics(features=features, adj=adj, Y=Y, labels=labels,
                                  max_n=5000, seed=42)
        new_npy = out_root / "embeddings_baselines" / f"{ds}_{method}_seed42.npy"
        if not new_npy.exists():
            shutil.copy(ph6_npy, new_npy)
        row = {
            "dataset": ds,
            "method": method,
            "seed": 42,
            "runtime_s": float(rt),
            **metrics,
            "info_provenance": "phase6_single_seed_carryover_to_round3",
            **extra_info,
        }
        cell_path = out_root / "tables" / "cells_baselines" / f"{ds}_{method}_seed42.parquet"
        cell_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_parquet(cell_path, index=False)
        print(f"[resurrect] {ds}/{method}/seed=42: T/C=({metrics['trustworthiness']:.3f},"
              f"{metrics['continuity']:.3f}) runtime={rt:.1f}s (Phase-6 carryover)", flush=True)
        written += 1
    print(f"[resurrect] wrote {written} cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
