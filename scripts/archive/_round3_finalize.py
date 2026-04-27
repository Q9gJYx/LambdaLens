"""Round-3 finalization: merge cells, print summary table per dataset.

Run after `scripts/run_e3_baselines.py` completes for all 5 main datasets and
PubMed. Prints a paper-ready summary that paper-side can paste into §5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_DATASETS = ("pbmc", "cora", "citeseer", "ca_astroph", "mnist_knn", "pubmed")


def _print_dataset_summary(out_root: Path, dataset: str) -> dict:
    p = out_root / "tables" / f"{dataset}_comparison_agg.parquet"
    if not p.exists():
        print(f"\n=== {dataset}: NO comparison_agg.parquet — skipping ===", flush=True)
        return {}
    df = pd.read_parquet(p)
    print(f"\n=== {dataset} ===", flush=True)
    summary: dict = {"dataset": dataset, "methods": {}}
    for r in df.itertuples():
        lt_mean = getattr(r, "label_trustworthiness_mean", float("nan"))
        lt_std = getattr(r, "label_trustworthiness_std", float("nan"))
        lt_b3 = getattr(r, "label_trustworthiness_best3", float("nan"))
        lc_mean = getattr(r, "label_continuity_mean", float("nan"))
        lc_std = getattr(r, "label_continuity_std", float("nan"))
        t_mean = getattr(r, "trustworthiness_mean", float("nan"))
        t_std = getattr(r, "trustworthiness_std", float("nan"))
        c_mean = getattr(r, "continuity_mean", float("nan"))
        rt_mean = getattr(r, "runtime_s_mean", float("nan"))
        rt_std = getattr(r, "runtime_s_std", float("nan"))
        n = int(r.n_seeds)
        print(f"  {r.method:18s} n={n}  "
              f"label_T={lt_mean:.3f}+-{lt_std:.3f} (best3 {lt_b3:.3f})  "
              f"label_C={lc_mean:.3f}+-{lc_std:.3f}  "
              f"T={t_mean:.3f}+-{t_std:.3f}  C={c_mean:.3f}  "
              f"runtime={rt_mean:.1f}+-{rt_std:.1f}s", flush=True)
        summary["methods"][r.method] = {
            "n_seeds": n,
            "label_trustworthiness_mean": float(lt_mean),
            "label_trustworthiness_std": float(lt_std),
            "label_trustworthiness_best3": float(lt_b3),
            "label_continuity_mean": float(lc_mean),
            "label_continuity_std": float(lc_std),
            "trustworthiness_mean": float(t_mean),
            "trustworthiness_std": float(t_std),
            "continuity_mean": float(c_mean),
            "runtime_mean": float(rt_mean),
            "runtime_std": float(rt_std),
        }
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--out-root", default="output")
    args = p.parse_args()

    out_root = Path(args.out_root)
    summaries: list[dict] = []
    for ds in args.datasets:
        s = _print_dataset_summary(out_root, ds)
        if s:
            summaries.append(s)

    final = {"per_dataset": summaries}
    out_path = out_root / "meta" / "round3_final_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[finalize] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
