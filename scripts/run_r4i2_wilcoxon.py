"""R4-I2: Wilcoxon paired-rank at full N=10 on contested datasets.

Pairs ours vs the headline competitor per dataset (PBMC->UMAP, Citeseer->PHATE,
MNIST->UMAP) on label_T over the same seed pool. Honest report regardless of
significance. Output: output/tables/wilcoxon.parquet.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CONTESTED = {
    "pbmc": "umap",
    "citeseer": "phate",
    "mnist_knn": "umap",
}


def main() -> int:
    out: list[dict] = []
    for ds, vs in CONTESTED.items():
        seeds = list(range(42, 52))  # N=10 if all present
        ours_lts: list[float] = []
        other_lts: list[float] = []
        for s in seeds:
            po = Path(f"output/tables/cells_baselines/{ds}_pysgtsnepi_seed{s}.parquet")
            pt = Path(f"output/tables/cells_baselines/{ds}_{vs}_seed{s}.parquet")
            if not (po.exists() and pt.exists()):
                continue
            lo = float(pd.read_parquet(po)["label_trustworthiness"].iloc[0])
            lt = float(pd.read_parquet(pt)["label_trustworthiness"].iloc[0])
            if np.isnan(lo) or np.isnan(lt):
                continue
            ours_lts.append(lo)
            other_lts.append(lt)
        n = len(ours_lts)
        if n < 5:
            print(f"[I2] {ds}: only {n} paired cells; skipping", flush=True)
            continue
        ours_arr = np.array(ours_lts)
        other_arr = np.array(other_lts)
        diff = ours_arr - other_arr
        try:
            stat, p = wilcoxon(ours_arr, other_arr)
            stat_v, p_v = float(stat), float(p)
        except Exception as e:
            print(f"[I2] {ds}: wilcoxon failed ({e}); reporting NaN", flush=True)
            stat_v, p_v = float("nan"), float("nan")
        out.append({
            "dataset": ds, "method_a": "pysgtsnepi", "method_b": vs,
            "n_pairs": n,
            "ours_mean": float(ours_arr.mean()),
            "other_mean": float(other_arr.mean()),
            "mean_diff": float(diff.mean()),
            "wilcoxon_stat": stat_v, "wilcoxon_p": p_v,
        })
        print(f"[I2] {ds}: ours vs {vs} N={n} mean_diff={diff.mean():+.4f} p={p_v:.4f}",
              flush=True)

    if not out:
        print("[I2] no datasets had >=5 paired cells; nothing written", flush=True)
        return 1
    Path("output/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet("output/tables/wilcoxon.parquet", index=False)
    print(f"[I2] wrote wilcoxon.parquet ({len(out)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
