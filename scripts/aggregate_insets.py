"""R4-A: aggregate N=5 PCA-init cell parquets to per-(dataset, lambda) mean LT&C.

Reads cells/{ds}_lam{lam}_seed{s}_init=pca*.parquet for seeds in --seeds,
computes mean label_T, mean label_C, and harmonic mean Label-T&C per
(dataset, lambda). Writes output/tables/teaser_lens_inset_means.json.

Run after R4-A cell grid completes (run_e1_local.py for A datasets + seeds).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS_UW = ("cora", "mnist_knn")
DATASETS_UWFALSE = ("pbmc",)
LAMBDAS = (1.0, 5.0, 20.0, 80.0)


def _harmonic(lt: float, lc: float) -> float | None:
    if np.isnan(lt) or np.isnan(lc) or lt <= 0 or lc <= 0:
        return None
    return 2.0 * lt * lc / (lt + lc)


def aggregate(cells_dir: Path, datasets: tuple[str, ...], uw_false: bool,
              lambdas: tuple[float, ...], seeds: list[int],
              recompute: bool = False) -> dict:
    """Aggregate per (ds, lambda) mean LT&C over seeds.

    If recompute=True (graph-only datasets like PBMC), label_T&C is
    recomputed from saved embeddings + adj + labels via
    lens.metrics.compute_metrics for any cell with NaN metrics.
    """
    suf = "_uw=False" if uw_false else ""
    emb_dir = cells_dir.parent.parent / "embeddings"
    out: dict[str, dict] = {}
    cached_ds: dict[str, tuple] = {}
    for ds in datasets:
        out[ds] = {}
        for lam in lambdas:
            rows = []
            for seed in seeds:
                p = cells_dir / f"{ds}_lam{lam}_seed{seed}_init=pca{suf}.parquet"
                if not p.exists():
                    continue
                df = pd.read_parquet(p)
                if df.empty:
                    continue
                r = df.iloc[0]
                lt = float(r.get("label_trustworthiness", float("nan")))
                lc = float(r.get("label_continuity", float("nan")))
                if recompute and np.isnan(lt) and np.isnan(lc):
                    emb_path = emb_dir / f"{ds}_lam{lam}_seed{seed}_init=pca{suf}.npy"
                    if emb_path.exists():
                        Y = np.load(str(emb_path))
                        if ds not in cached_ds:
                            from lens.data import load_dataset
                            cached_ds[ds] = load_dataset(ds)
                        adj, features, labels = cached_ds[ds]
                        from lens.metrics import compute_metrics
                        m = compute_metrics(features=features, adj=adj, Y=Y,
                                            labels=labels, max_n=2000, seed=seed)
                        lt = float(m.get("label_trustworthiness", float("nan")))
                        lc = float(m.get("label_continuity", float("nan")))
                rows.append((lt, lc))
            if not rows:
                out[ds][str(lam)] = None
                continue
            lts = [r[0] for r in rows if not np.isnan(r[0])]
            lcs = [r[1] for r in rows if not np.isnan(r[1])]
            mean_lt = float(np.mean(lts)) if lts else float("nan")
            mean_lc = float(np.mean(lcs)) if lcs else float("nan")
            hm = _harmonic(mean_lt, mean_lc)
            out[ds][str(lam)] = {
                "mean_label_T": mean_lt,
                "mean_label_C": mean_lc,
                "harmonic_label_TC": hm,
                "n_seeds": len(rows),
                "seeds": seeds,
            }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--out-root", default="output")
    args = p.parse_args()

    cells_dir = Path(args.out_root) / "tables" / "cells"
    result = {}
    result.update(aggregate(cells_dir, DATASETS_UW, uw_false=False,
                            lambdas=LAMBDAS, seeds=args.seeds, recompute=False))
    result.update(aggregate(cells_dir, DATASETS_UWFALSE, uw_false=True,
                            lambdas=LAMBDAS, seeds=args.seeds, recompute=True))

    out_path = Path(args.out_root) / "tables" / "teaser_lens_inset_means.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[aggregate_insets] wrote {out_path}", flush=True)
    for ds, by_lam in result.items():
        for lam, v in by_lam.items():
            if v:
                print(f"  {ds} lam={lam}: LT={v['mean_label_T']:.3f} LC={v['mean_label_C']:.3f} n={v['n_seeds']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
