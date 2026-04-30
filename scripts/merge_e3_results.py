"""Round-3: merge per-(method, seed) cell parquets into per-dataset comparison tables.

Reads `output/tables/cells_baselines/<ds>_<method>_seed<s>.parquet`, writes:
  - `output/tables/<ds>_comparison.parquet` — multi-row, one per (method, seed)
  - `output/tables/<ds>_comparison_agg.parquet` — one row per method with
    mean / std / median / best-of-3 of label_T&C and runtime.

For graph-only datasets (no labels), aggregates trustworthiness/continuity
instead of label_T&C.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATASETS = ("pbmc", "cora", "citeseer", "ca_astroph", "mnist_knn")


def _agg_method(group: pd.DataFrame) -> dict:
    def _mean_std_best3(col: str) -> tuple[float, float, float, float]:
        v = group[col].dropna().to_numpy()
        if v.size == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        mean = float(v.mean())
        std = float(v.std(ddof=0))
        median = float(np.median(v))
        if v.size >= 3:
            best3 = float(np.sort(v)[-3:].mean())
        else:
            best3 = mean
        return mean, std, median, best3

    iteration_count = -1
    if "iteration_count" in group:
        raw_iteration_count = group["iteration_count"].dropna()
        if not raw_iteration_count.empty:
            iteration_count = int(raw_iteration_count.iloc[0])

    out: dict = {
        "method": group["method"].iloc[0],
        "n_seeds": int(group["seed"].nunique()),
        "init_strategy": group["init_strategy"].iloc[0] if "init_strategy" in group else "default",
        "iteration_count": iteration_count,
    }
    for col in ("label_trustworthiness", "label_continuity",
                "trustworthiness", "continuity", "runtime_s"):
        if col not in group.columns:
            continue
        m, s, med, b3 = _mean_std_best3(col)
        out[f"{col}_mean"] = m
        out[f"{col}_std"] = s
        out[f"{col}_median"] = med
        out[f"{col}_best3"] = b3
    return out


def _merge_dataset(out_root: Path, dataset: str) -> None:
    cells_dir = out_root / "tables" / "cells_baselines"
    files = sorted(cells_dir.glob(f"{dataset}_*_seed*.parquet"))
    if not files:
        print(f"[merge] {dataset}: no cells", flush=True)
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["method", "seed"]).reset_index(drop=True)
    df.to_parquet(out_root / "tables" / f"{dataset}_comparison.parquet", index=False)
    print(f"[merge] {dataset}: wrote {len(df)} cell rows to {dataset}_comparison.parquet", flush=True)

    agg_rows = [_agg_method(g) for _, g in df.groupby("method", sort=False)]
    agg_df = pd.DataFrame(agg_rows).sort_values("method").reset_index(drop=True)
    agg_df.to_parquet(out_root / "tables" / f"{dataset}_comparison_agg.parquet", index=False)
    print(f"[merge] {dataset}: wrote agg ({len(agg_df)} methods) to {dataset}_comparison_agg.parquet", flush=True)
    for r in agg_df.itertuples():
        lt = getattr(r, "label_trustworthiness_mean", float("nan"))
        lt_std = getattr(r, "label_trustworthiness_std", float("nan"))
        rt = getattr(r, "runtime_s_mean", float("nan"))
        rt_std = getattr(r, "runtime_s_std", float("nan"))
        print(f"  {r.method}: n={r.n_seeds} label_T={lt:.3f}+-{lt_std:.3f} runtime={rt:.1f}+-{rt_std:.1f}s",
              flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--out-root", default="output")
    args = p.parse_args()
    out_root = Path(args.out_root)
    for ds in args.datasets:
        _merge_dataset(out_root, ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
